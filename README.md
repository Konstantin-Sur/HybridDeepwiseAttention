# HybridDeepwiseAttention

```python
class HybridSparseAttention(nn.Module):
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
            return t.reshape(B, self.heads, self.dim, seq).reshape(B*self.heads, self.dim, seq)
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
```
