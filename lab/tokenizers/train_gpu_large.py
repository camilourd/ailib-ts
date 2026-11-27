"""F11.4: comparación GPU a ~8M de parámetros con las familias de tokenizers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path("data/gpu")
OUT = DATA / "results-8m.json"
CONTEXT = 128
BATCH = 8
STEPS = 400
D_MODEL = 256
HEADS = 8
BLOCKS = 6
HIDDEN = 1024
SEED = 1
NAMES = [
    "fair-bpe-2048",
    "fair-sentencepiece-2048",
    "fair-wordpiece-2048",
    "fair-unigram-2048",
]


def rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, length, dimension = q.shape
    half = dimension // 2
    positions = torch.arange(length, device=q.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=q.device, dtype=torch.float32) / half
    )
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    first, second = q[..., :half], q[..., half:]
    q_out = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    first_k, second_k = k[..., :half], k[..., half:]
    k_out = torch.cat((first_k * cos - second_k * sin, first_k * sin + second_k * cos), dim=-1)
    return q_out, k_out


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, D_MODEL * 3)
        self.output = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, HEADS, D_MODEL // HEADS).transpose(1, 2)
        k = k.view(batch, length, HEADS, D_MODEL // HEADS).transpose(1, 2)
        v = v.view(batch, length, HEADS, D_MODEL // HEADS).transpose(1, 2)
        q, k = rope(q, k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(attended.transpose(1, 2).contiguous().view(batch, length, D_MODEL))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attention = Attention()
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, D_MODEL))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocabulary: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(BLOCKS))
        self.norm = nn.LayerNorm(D_MODEL)
        self.output = nn.Linear(D_MODEL, vocabulary)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


def evaluate(model: nn.Module, data: torch.Tensor, byte_count: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(data) - CONTEXT - 1, CONTEXT):
            x = data[start : start + CONTEXT][None, :]
            y = data[start + 1 : start + CONTEXT + 1][None, :]
            loss = F.cross_entropy(model(x).view(-1, model.output.out_features), y.view(-1))
            total += loss.item()
            count += 1
    model.train()
    return (total / max(1, count)) * len(data) / byte_count / math.log(2)


def train(name: str) -> dict:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    payload = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    model = TransformerLM(payload["vocabulary"]).to(device)
    data = torch.tensor(payload["trainTokens"], dtype=torch.long, device=device)
    test = torch.tensor(payload["testTokens"], dtype=torch.long, device=device)
    external = torch.tensor(payload["externalTokens"], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(SEED)
    for step in range(STEPS):
        starts = torch.randint(0, len(data) - CONTEXT - 1, (BATCH,), generator=generator).to(device)
        x = torch.stack([data[start : start + CONTEXT] for start in starts])
        y = torch.stack([data[start + 1 : start + CONTEXT + 1] for start in starts])
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"{name}: paso {step} · loss {loss.item():.4f}")
    result = {
        "name": name,
        "vocabulary": payload["vocabulary"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "bpb_test": evaluate(model, test, payload["bytes"]["test"]),
        "bpb_external": evaluate(model, external, payload["bytes"]["external"]),
        "device": str(device),
    }
    print(
        f"{name}: BPB test {result['bpb_test']:.3f} · externo {result['bpb_external']:.3f} · "
        f"params {result['parameters']}"
    )
    return result


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no disponible")
    results = []
    for name in NAMES:
        results.append(train(name))
        torch.cuda.empty_cache()
        OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"guardado: {OUT}")
