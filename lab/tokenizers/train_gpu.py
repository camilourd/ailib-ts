"""F9.4b: entrena el mismo transformer en GPU para BPE y BPE+sufijos.

Uso:
    python lab/tokenizers/train_gpu.py

Lee data/gpu/<nombre>.json (tokens exportados por export-gpu.ts) y escribe
data/gpu/results.json con BPB de Quijote test y Lazarillo.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DATA = Path("data/gpu")
CONTEXT = 128
BATCH = 16
STEPS = 400
LEARNING_RATE = 3e-4
D_MODEL = 160
HEADS = 8
BLOCKS = 4
HIDDEN = 640
SEED = 1


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
    first = q[..., :half]
    second = q[..., half:]
    q_out = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    first_k = k[..., :half]
    second_k = k[..., half:]
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
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, D_MODEL),
        )

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

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def windows(tokens: list[int], device: torch.device):
    data = torch.tensor(tokens, dtype=torch.long, device=device)
    return data


def batch_from(data: torch.Tensor, generator: torch.Generator):
    starts = torch.randint(0, len(data) - CONTEXT - 1, (BATCH,), generator=generator).to(data.device)
    x = torch.stack([data[start : start + CONTEXT] for start in starts])
    y = torch.stack([data[start + 1 : start + CONTEXT + 1] for start in starts])
    return x, y


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
    payload = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = TransformerLM(payload["vocabulary"]).to(device)
    train_data = windows(payload["trainTokens"], device)
    test_data = windows(payload["testTokens"], device)
    external_data = windows(payload["externalTokens"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED)
    for step in range(STEPS):
        x, y = batch_from(train_data, generator)
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
        "bpb_test": evaluate(model, test_data, payload["bytes"]["test"]),
        "bpb_external": evaluate(model, external_data, payload["bytes"]["external"]),
        "device": str(device),
    }
    print(
        f"{name}: BPB test {result['bpb_test']:.3f} · externo {result['bpb_external']:.3f} · "
        f"params {result['parameters']} · {result['device']}"
    )
    return result


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible; instala torch+cu126")
    names = sorted(
        path.stem for path in DATA.glob("*.json")
        if path.name != "results.json"
    )
    results = [train(name) for name in names]
    (DATA / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("guardado: data/gpu/results.json")
