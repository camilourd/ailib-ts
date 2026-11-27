"""F11.4b: comparación GPU configurable por semilla, bloques y tamaño.

Ejemplo:
    python lab/tokenizers/train_gpu_fair.py --seed 2 --out data/gpu/results-seed2.json
    python lab/tokenizers/train_gpu_fair.py --seed 1 --blocks 8 --hidden 1280 --out data/gpu/results-8m.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path("data/gpu")
CONTEXT = 128
BATCH = 8
STEPS = 400
D_MODEL = 256
HEADS = 8
LayerNorm = nn.LayerNorm


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
    def __init__(self, model: int, heads: int, position: str = "rope") -> None:
        super().__init__()
        self.heads = heads
        self.model = model
        self.position = position
        self.qkv = nn.Linear(model, model * 3)
        self.output = nn.Linear(model, model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.model // self.heads).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.model // self.heads).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.model // self.heads).transpose(1, 2)
        if self.position == "rope":
            q, k = rope(q, k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(attended.transpose(1, 2).contiguous().view(batch, length, self.model))


class RMSNorm(nn.Module):
    def __init__(self, model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, model: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(model, hidden)
        self.w3 = nn.Linear(model, hidden)
        self.w2 = nn.Linear(hidden, model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    def __init__(self, model: int, hidden: int, experts: int, top_k: int, random_router: bool = False) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(model, hidden), nn.ReLU(), nn.Linear(hidden, model))
            for _ in range(experts)
        )
        self.router = nn.Linear(model, experts)
        self.top_k = top_k
        self.random_router = random_router
        self.usage = [0 for _ in range(experts)]
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(self.router(x), dim=-1)
        if self.random_router:
            indices = torch.randint(0, len(self.experts), (x.shape[0], self.top_k), device=x.device)
            weights = torch.ones_like(indices, dtype=x.dtype)
        else:
            weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
            counts = torch.bincount(indices.flatten(), minlength=len(self.experts)).float()
            fractions = counts / max(1, x.shape[0])
            mean_probabilities = probabilities.mean(dim=0)
            self.aux_loss = len(self.experts) * (fractions * mean_probabilities).sum()
        output = torch.zeros_like(x)
        for expert in range(len(self.experts)):
            selected = (indices == expert).any(dim=-1)
            if selected.any():
                gate = (probabilities[selected, expert] * (indices[selected] == expert).sum(dim=-1)).unsqueeze(-1)
                output[selected] += self.experts[expert](x[selected]) * gate
                self.usage[expert] += int(selected.sum().item())
        return output

    def reset_usage(self) -> None:
        self.usage = [0 for _ in self.usage]


class Block(nn.Module):
    def __init__(
        self,
        model: int,
        heads: int,
        hidden: int,
        norm_mode: str = "pre",
        norm_kind: str = "layer",
        ffn_kind: str = "relu",
        parallel: bool = False,
        position: str = "rope",
        dropout: float = 0.0,
        layer_scale: bool = False,
        moe_experts: int = 0,
        moe_topk: int = 1,
        moe_random: bool = False,
    ) -> None:
        super().__init__()
        self.norm_mode = norm_mode
        self.parallel = parallel
        self.dropout = nn.Dropout(dropout)
        self.attention_scale = nn.Parameter(torch.ones(model)) if layer_scale else None
        self.ffn_scale = nn.Parameter(torch.ones(model)) if layer_scale else None
        norm = LayerNorm if norm_kind == "layer" else RMSNorm
        self.norm1 = norm(model)
        self.attention = Attention(model, heads, position)
        self.norm2 = norm(model)
        self.ffn = (
            MoEFFN(model, hidden, moe_experts, moe_topk, moe_random)
            if moe_experts > 0
            else (
                nn.Sequential(nn.Linear(model, hidden), nn.ReLU(), nn.Linear(hidden, model))
                if ffn_kind == "relu"
                else SwiGLU(model, hidden)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "post":
            attention = self.dropout(self.attention(x))
            if self.attention_scale is not None:
                attention = attention * self.attention_scale
            hidden = self.norm1(x + attention)
            ffn = self.dropout(self.ffn(hidden))
            if self.ffn_scale is not None:
                ffn = ffn * self.ffn_scale
            return self.norm2(hidden + ffn)
        attention = self.dropout(self.attention(self.norm1(x)))
        if self.attention_scale is not None:
            attention = attention * self.attention_scale
        ffn_input = x if self.parallel else x + attention
        ffn = self.dropout(self.ffn(self.norm2(ffn_input)))
        if self.ffn_scale is not None:
            ffn = ffn * self.ffn_scale
        if self.parallel:
            return x + attention + ffn
        return ffn_input + ffn


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        blocks: int,
        hidden: int,
        model: int = D_MODEL,
        heads: int = HEADS,
        norm_mode: str = "pre",
        norm_kind: str = "layer",
        ffn_kind: str = "relu",
        parallel: bool = False,
        position: str = "rope",
        dropout: float = 0.0,
        layer_scale: bool = False,
        tie_embedding: bool = False,
        moe_experts: int = 0,
        moe_topk: int = 1,
        moe_random: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, model)
        if tie_embedding:
            nn.init.normal_(self.embedding.weight, std=0.02)
        self.blocks = nn.ModuleList(
            Block(model, heads, hidden, norm_mode, norm_kind, ffn_kind, parallel, position, dropout, layer_scale, moe_experts, moe_topk, moe_random)
            for _ in range(blocks)
        )
        self.norm = LayerNorm(model) if norm_kind == "layer" else RMSNorm(model)
        self.output = None if tie_embedding else nn.Linear(model, vocabulary)
        self.position = position

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return F.linear(x, self.embedding.weight) if self.output is None else self.output(x)

    def moe_aux_loss(self) -> torch.Tensor:
        losses = [
            module.aux_loss
            for module in self.modules()
            if isinstance(module, MoEFFN) and module.aux_loss.requires_grad
        ]
        return torch.stack(losses).sum() if losses else torch.tensor(0.0, device=self.norm.weight.device)


class RecurrentLM(nn.Module):
    def __init__(self, vocabulary: int, model: int, hidden: int, kind: str) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, model)
        if kind == "lstm":
            self.recurrent = nn.LSTM(model, hidden, num_layers=2, batch_first=True)
        else:
            self.recurrent = nn.GRU(model, hidden, num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden, vocabulary)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x, _ = self.recurrent(x)
        return self.output(x)


def evaluate(model: nn.Module, data: torch.Tensor, byte_count: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(data) - CONTEXT - 1, CONTEXT):
            x = data[start : start + CONTEXT][None, :]
            y = data[start + 1 : start + CONTEXT + 1][None, :]
            loss = F.cross_entropy(model(x).view(-1, model.embedding.num_embeddings), y.view(-1))
            total += loss.item()
            count += 1
    model.train()
    return (total / max(1, count)) * len(data) / byte_count / math.log(2)


def train(
    name: str,
    seed: int,
    blocks: int,
    hidden: int,
    model_size: int = D_MODEL,
    heads: int = HEADS,
    arch: str = "transformer",
    variant: str = "classic",
    learning_rate: float = 3e-4,
    schedule: str = "none",
    weight_decay: float = 0.0,
    dropout: float = 0.0,
    layer_scale: bool = False,
    tie_embedding: bool = False,
    moe_experts: int = 0,
    moe_topk: int = 1,
    moe_random: bool = False,
    moe_balance: float = 0.0,
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    payload = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    if arch in {"gru", "lstm"}:
        model = RecurrentLM(payload["vocabulary"], model_size, 768 if arch == "gru" else 512, arch).to(device)
    else:
        norm_kind = "rms" if variant == "modern" else "layer"
        ffn_kind = "swiglu" if variant == "modern" else "relu"
        parallel = variant == "parallel"
        position = "none" if variant == "nope" else "rope"
        model = TransformerLM(
            payload["vocabulary"], blocks, hidden, model_size, heads,
            "post" if arch == "postnorm" else "pre",
            norm_kind, ffn_kind, parallel, position,
            dropout, layer_scale, tie_embedding, moe_experts, moe_topk, moe_random,
        ).to(device)
    data = torch.tensor(payload["trainTokens"], dtype=torch.long, device=device)
    test = torch.tensor(payload["testTokens"], dtype=torch.long, device=device)
    external = torch.tensor(payload["externalTokens"], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(STEPS):
        if schedule == "cosine":
            warmup = min(1.0, (step + 1) / 20)
            progress = step / max(1, STEPS - 1)
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * warmup * factor
        starts = torch.randint(0, len(data) - CONTEXT - 1, (BATCH,), generator=generator).to(device)
        x = torch.stack([data[start : start + CONTEXT] for start in starts])
        y = torch.stack([data[start + 1 : start + CONTEXT + 1] for start in starts])
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        if moe_balance > 0 and moe_experts > 0 and not moe_random:
            loss = loss + moe_balance * model.moe_aux_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"{name} seed{seed}: paso {step} · loss {loss.item():.4f}", flush=True)
    domain_bpb: dict[str, float] = {}
    domain_usage: dict[str, list[int]] = {}
    if "testSets" in payload:
        for domain, tokens in payload["testSets"].items():
            for module in model.modules():
                if isinstance(module, MoEFFN):
                    module.reset_usage()
            domain_bpb[domain] = evaluate(
                model,
                torch.tensor(tokens, dtype=torch.long, device=device),
                payload["bytes"][domain],
            )
            domain_usage[domain] = [
                usage
                for module in model.modules()
                if isinstance(module, MoEFFN)
                for usage in module.usage
            ]
    result = {
        "name": name,
        "seed": seed,
        "arch": arch,
        "variant": variant,
        "learningRate": learning_rate,
        "schedule": schedule,
        "weightDecay": weight_decay,
        "dropout": dropout,
        "layerScale": layer_scale,
        "tieEmbedding": tie_embedding,
        "moeExperts": moe_experts,
        "moeTopK": moe_topk,
        "moeRandom": moe_random,
        "moeBalance": moe_balance,
        "blocks": blocks,
        "hidden": hidden,
        "vocabulary": payload["vocabulary"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "bpb_test": evaluate(model, test, payload["bytes"]["test"]),
        "bpb_external": evaluate(model, external, payload["bytes"]["external"]),
        "moeUsage": [
            usage
            for module in model.modules()
            if isinstance(module, MoEFFN)
            for usage in module.usage
        ],
        "domainBpb": domain_bpb,
        "domainMoeUsage": domain_usage,
        "device": str(device),
    }
    print(
        f"{name} seed{seed}: BPB test {result['bpb_test']:.3f} · externo {result['bpb_external']:.3f} · "
        f"params {result['parameters']}",
        flush=True,
    )
    return result


def main() -> None:
    global CONTEXT, STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--model", type=int, default=D_MODEL)
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--arch", choices=["transformer", "postnorm", "gru", "lstm"], default="transformer")
    parser.add_argument("--variant", choices=["classic", "modern", "nope", "parallel"], default="classic")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--schedule", choices=["none", "cosine"], default="none")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layer-scale", action="store_true")
    parser.add_argument("--tie-embedding", action="store_true")
    parser.add_argument("--moe-experts", type=int, default=0)
    parser.add_argument("--moe-topk", type=int, default=1)
    parser.add_argument("--moe-random", action="store_true")
    parser.add_argument("--moe-balance", type=float, default=0.0)
    parser.add_argument("--out", default="data/gpu/results-fair.json")
    parser.add_argument("--names", default="fair-bpe-2048,fair-sentencepiece-2048,fair-wordpiece-2048,fair-unigram-2048")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no disponible")
    CONTEXT = args.context
    STEPS = args.steps
    results = []
    for name in args.names.split(","):
        results.append(train(name, args.seed, args.blocks, args.hidden, args.model, args.heads, args.arch, args.variant, args.lr, args.schedule, args.weight_decay, args.dropout, args.layer_scale, args.tie_embedding, args.moe_experts, args.moe_topk, args.moe_random, args.moe_balance))
        torch.cuda.empty_cache()
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"guardado: {args.out}", flush=True)


if __name__ == "__main__":
    main()
