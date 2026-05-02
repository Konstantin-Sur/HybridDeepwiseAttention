import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm          # auto selects notebook-friendly bar in Colab
import os

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

# ─── Hyperparameters (edit here instead of argparse) ─────────────────────────
CONFIG = dict(
    data_root    = "./data",        # dataset download directory
    batch_size   = 32,
    num_epochs   = 20,
    lr           = 3e-4,
    weight_decay = 1e-4,
    img_size     = 224,
    val_split    = 0.15,            # fraction of training set used for validation
    num_workers  = 2,
    # HybridSparseAttention settings
    attn_channels    = 512,         # must match the feature-map channels at injection point
    attn_heads       = 8,
    compress_ratio   = 14,
    top_k_frac       = 0.25,
    # Misc
    device = "cuda" if torch.cuda.is_available() else "cpu",
    save_path = "best_model.pth",
)

print(f"Running on: {CONFIG['device']}")


# ─── AdaptiveCompressedAttention ────────────────────────────────────────────────────

class AdaptiveCompressedAttention(nn.Module):
    """
    Two-stage sparse attention for feature maps.

    Stage 1 — global context with compressed keys/values (stride-r pooling).
    Stage 2 — attend only to the top-k most salient compressed tokens,
               selected by accumulated attention mass from stage 1.
    """

    def __init__(self, channels, heads=4, compress_ratio=4, top_k_frac=0.25):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.heads       = heads
        self.dim         = channels // heads
        self.scale       = self.dim ** -0.5
        self.top_k_frac  = top_k_frac
        self.r           = compress_ratio

        # Pointwise projections for query / key / value
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)

        # Depthwise strided conv to compress spatial resolution
        self.compress_k = nn.Conv2d(
            channels, channels, self.r, stride=self.r, groups=channels, bias=False
        )
        self.compress_v = nn.Conv2d(
            channels, channels, self.r, stride=self.r, groups=channels, bias=False
        )

        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Compress keys and values spatially
        k_s = self.compress_k(k)
        v_s = self.compress_v(v)
        n   = k_s.shape[2] * k_s.shape[3]      # compressed sequence length

        def reshape(t, seq):
            # (B, C, seq) → (B*heads, dim, seq)
            return (
                t.reshape(B, self.heads, self.dim, seq)
                 .reshape(B * self.heads, self.dim, seq)
            )

        q = reshape(q.reshape(B, C, N),  N)
        k = reshape(k_s.reshape(B, C, n), n)
        v = reshape(v_s.reshape(B, C, n), n)

        # Stage 1: sparse attention over compressed tokens
        attn = torch.einsum("b d i, b d j -> b i j", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # Select top-k compressed tokens by total attention mass
        token_score = attn.sum(dim=1)                           # (B*heads, n)
        top_k       = max(1, int(n * self.top_k_frac))
        _, idx      = token_score.topk(top_k, dim=-1)          # (B*heads, top_k)

        idx_exp = idx.unsqueeze(1).expand(-1, self.dim, -1)    # (B*heads, dim, top_k)
        k_top   = k.gather(2, idx_exp)
        v_top   = v.gather(2, idx_exp)

        # Stage 2: full attention only over selected top-k tokens
        attn_full = torch.einsum("b d i, b d j -> b i j", q, k_top) * self.scale
        attn_full = attn_full.softmax(dim=-1)

        out = torch.einsum("b i j, b d j -> b d i", attn_full, v_top)
        out = out.reshape(B, self.heads, self.dim, N).reshape(B, C, H, W)
        return self.out(out)


# ─── Model: ResNet-50 + HybridSparseAttention ────────────────────────────────

class ResNetWithSparseAttn(nn.Module):
    """
    ResNet-50 backbone with HybridSparseAttention injected after layer3.
    Channels at that point are 1024; we project down to attn_channels,
    apply attention, then project back before passing to layer4.
    """

    def __init__(self, num_classes, cfg):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Keep everything up to and including layer3
        self.stem   = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3       # output: (B, 1024, H/16, W/16)

        ch = cfg["attn_channels"]           # e.g. 512

        # Project 1024 → ch for attention, then back to 1024
        self.down_proj = nn.Conv2d(1024, ch, 1, bias=False)
        self.attn      = AdaptiveCompressedAttention(
            channels      = ch,
            heads         = cfg["attn_heads"],
            compress_ratio= cfg["compress_ratio"],
            top_k_frac    = cfg["top_k_frac"],
        )
        self.up_proj   = nn.Conv2d(ch, 1024, 1, bias=False)
        self.attn_norm = nn.BatchNorm2d(1024)

        self.layer4    = backbone.layer4    # output: (B, 2048, H/32, W/32)
        self.avgpool   = backbone.avgpool
        self.head      = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Residual attention block
        residual = x
        a = self.down_proj(x)
        a = self.attn(a)
        a = self.up_proj(a)
        x = self.attn_norm(residual + a)

        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_dataloaders(cfg):
    sz = cfg["img_size"]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(sz, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(sz * 1.14)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # OxfordIIITPet: 37 breeds, ~7 349 images total
    full_train = datasets.OxfordIIITPet(
        cfg["data_root"], split="trainval", download=True, transform=train_tf
    )
    test_ds = datasets.OxfordIIITPet(
        cfg["data_root"], split="test", download=True, transform=val_tf
    )

    # Split trainval → train / val
    n_val   = int(len(full_train) * cfg["val_split"])
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Apply val transform to the validation split
    val_ds.dataset = datasets.OxfordIIITPet(
        cfg["data_root"], split="trainval", download=False, transform=val_tf
    )

    kw = dict(batch_size=cfg["batch_size"],
              num_workers=cfg["num_workers"],
              pin_memory=cfg["device"] == "cuda")

    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


# ─── Training utilities ───────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    """Run one full pass over loader. Returns (avg_loss, accuracy)."""
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    desc = "Train" if training else "Val  "

    with ctx:
        bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for imgs, labels in bar:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs           = labels.size(0)
            total_loss  += loss.item() * bs
            correct     += (logits.argmax(1) == labels).sum().item()
            total       += bs

            bar.set_postfix(loss=f"{loss.item():.3f}",
                            acc=f"{correct / total:.3%}")

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    return run_epoch(model, loader, criterion, None, device, training=False)


# ─── Main training loop ───────────────────────────────────────────────────────

def train(cfg):
    device = torch.device(cfg["device"])

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    num_classes = 37   # Oxford Pets has 37 breed classes

    model = ResNetWithSparseAttn(num_classes, cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Use different LRs: small for pretrained backbone, larger for new layers
    attn_params   = list(model.down_proj.parameters()) + \
                    list(model.attn.parameters())      + \
                    list(model.up_proj.parameters())   + \
                    list(model.attn_norm.parameters()) + \
                    list(model.head.parameters())
    backbone_params = [p for p in model.parameters()
                       if not any(p is q for q in attn_params)]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["lr"] * 0.1},
        {"params": attn_params,     "lr": cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])

    # Cosine annealing over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["num_epochs"], eta_min=1e-6
    )

    best_val_acc = 0.0

    epoch_bar = tqdm(range(1, cfg["num_epochs"] + 1),
                     desc="Epochs", dynamic_ncols=True)

    for epoch in epoch_bar:
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_bar.set_postfix(
            tr_acc=f"{tr_acc:.3%}",
            val_acc=f"{val_acc:.3%}",
            val_loss=f"{val_loss:.3f}",
        )

        # Save checkpoint when validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg["save_path"])
            tqdm.write(
                f"  ✓ Epoch {epoch:02d} | "
                f"val_acc={val_acc:.3%} (best) | "
                f"checkpoint saved"
            )
        else:
            tqdm.write(
                f"  · Epoch {epoch:02d} | "
                f"tr_loss={tr_loss:.3f}  tr_acc={tr_acc:.3%} | "
                f"val_loss={val_loss:.3f}  val_acc={val_acc:.3%}"
            )

    # ── Final evaluation on held-out test split ──────────────────────────────
    print("\nLoading best checkpoint for test evaluation …")
    model.load_state_dict(torch.load(cfg["save_path"], map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test  loss={test_loss:.4f}  acc={test_acc:.3%}")
    print(f"Best val acc: {best_val_acc:.3%}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(CONFIG)
