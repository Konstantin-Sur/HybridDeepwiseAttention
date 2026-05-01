import torch
import torch.nn as nn
import time

class AdaptiveCompressedAttention(nn.Module):
    def __init__(self, channels, heads=4, compress_ratio=4, top_k_frac=0.25):
        super().__init__()
        assert channels % heads == 0
        self.heads = heads
        self.dim = channels // heads
        self.scale = self.dim ** -0.5
        self.top_k_frac = top_k_frac
        self.r = compress_ratio

        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)

        self.compress_k = nn.Conv2d(channels, channels, self.r, stride=self.r, groups=channels, bias=False)
        self.compress_v = nn.Conv2d(channels, channels, self.r, stride=self.r, groups=channels, bias=False)

        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        k_s = self.compress_k(k)
        v_s = self.compress_v(v)
        n = k_s.shape[2] * k_s.shape[3]

        def reshape(t, seq):
            return t.reshape(B, self.heads, self.dim, seq).reshape(B * self.heads, self.dim, seq)

        q = reshape(q.reshape(B, C, N), N)
        k = reshape(k_s.reshape(B, C, n), n)
        v = reshape(v_s.reshape(B, C, n), n)

        attn = torch.einsum("b d i, b d j -> b i j", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        token_score = attn.sum(dim=1)
        top_k = max(1, int(n * self.top_k_frac))
        _, idx = token_score.topk(top_k, dim=-1)

        idx_exp = idx.unsqueeze(1).expand(-1, self.dim, -1)
        k_top = k.gather(2, idx_exp)
        v_top = v.gather(2, idx_exp)

        attn_full = torch.einsum("b d i, b d j -> b i j", q, k_top) * self.scale
        attn_full = attn_full.softmax(dim=-1)

        out = torch.einsum("b i j, b d j -> b d i", attn_full, v_top)
        out = out.reshape(B, self.heads, self.dim, N).reshape(B, C, H, W)
        return self.out(out)


def measure_memory_and_time(model, input_shape, device, num_warmup=3, num_iters=5, with_backward=False):
    B, C, H, W = input_shape
    x = torch.randn(B, C, H, W, device=device)

    # Сброс статистики памяти
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Forward (for memory measure)
    out = model(x)
    if with_backward:
        loss = out.sum()
        loss.backward()

    torch.cuda.synchronize(device)
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    # Warmup
    for _ in range(num_warmup):
        _ = model(x)
        if with_backward:
            _ = model(x).sum().backward()
    torch.cuda.synchronize(device)



    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        out = model(x)
        if with_backward:
            out.sum().backward()
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / num_iters * 1000  # в миллисекундах
    return peak_memory, avg_time


def test_memory_and_time():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        return

    # Params
    channels = 512
    heads = 4
    compress_ratio = 16
    top_k_frac = 0.5

    model = AdaptiveCompressedAttention(
        channels=channels,
        heads=heads,
        compress_ratio=compress_ratio,
        top_k_frac=top_k_frac
    ).to(device)
    model.eval()

    # Size of image(side size)
    sizes = [64, 128, 256, 512, 1024]
    batch = 1

    print(f"{'Size':<12} {'Peak Memory (MB)':<20} {'Time (ms)':<12}")
    print("-" * 45)

    for size in sizes:
        input_shape = (batch, channels, size, size)
        try:
            peak_mb, time_ms = measure_memory_and_time(
                model, input_shape, device,
                num_warmup=3, num_iters=5, with_backward=False
            )
            print(f"{size}x{size:<8} {peak_mb:.2f} MB            {time_ms:.2f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{size}x{size:<8} OOM (out of memory)    N/A")
                torch.cuda.empty_cache()
            else:
                raise


if __name__ == "__main__":
    test_memory_and_time()
