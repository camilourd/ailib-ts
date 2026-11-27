# F14.4: variantes de crecimiento: freeze/unfreeze, más pasos y Net2Net de anchura.
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train_gpu_fair import CONTEXT, TransformerLM, evaluate

DATA = Path("data/gpu")
payload = json.loads((DATA / "f14-mixture-bpe-2k.json").read_text(encoding="utf-8"))
device = torch.device("cuda")
data = torch.tensor(payload["trainTokens"], dtype=torch.long, device=device)
test = torch.tensor(payload["testTokens"], dtype=torch.long, device=device)
external = torch.tensor(payload["externalTokens"], dtype=torch.long, device=device)


def make_model(blocks: int, hidden: int = 1536):
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    return TransformerLM(payload["vocabulary"], blocks, hidden, 384, 8, activation_checkpoint=True).to(device)


def train_steps(model, optimizer, steps: int, generator) -> None:
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(4):
            starts = torch.randint(0, len(data) - CONTEXT - 1, (2,), generator=generator).to(device)
            x = torch.stack([data[start : start + CONTEXT] for start in starts])
            y = torch.stack([data[start + 1 : start + CONTEXT + 1] for start in starts])
            loss = F.cross_entropy(model(x).view(-1, payload["vocabulary"]), y.view(-1))
            (loss / 4).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0:
            print(f"paso {step}: {loss.item():.4f}", flush=True)


def domains(model) -> dict[str, float]:
    return {
        name: evaluate(model, torch.tensor(tokens, device=device), payload["bytes"][name])
        for name, tokens in payload["testSets"].items()
    }


def base_pretrain(model) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_steps(model, optimizer, 50, torch.Generator().manual_seed(1))


rows = []
started = time.time()

# 1. Congelar las viejas, entrenar solo las nuevas, luego todo.
freeze = make_model(6)
base_pretrain(freeze)
freeze.grow_depth(2)
freeze.to(device)
for block in freeze.blocks[:6]:
    for parameter in block.parameters():
        parameter.requires_grad = False
train_steps(freeze, torch.optim.AdamW(filter(lambda p: p.requires_grad, freeze.parameters()), lr=3e-4), 50, torch.Generator().manual_seed(2))
for parameter in freeze.parameters():
    parameter.requires_grad = True
train_steps(freeze, torch.optim.AdamW(freeze.parameters(), lr=1e-4), 50, torch.Generator().manual_seed(3))
rows.append({"name": "freeze-unfreeze", "params": sum(p.numel() for p in freeze.parameters()), "test": evaluate(freeze, test, payload["bytes"]["test"]), "domains": domains(freeze)})
del freeze
torch.cuda.empty_cache()

# 2. Más pasos después de crecer.
more = make_model(6)
base_pretrain(more)
more.grow_depth(2)
more.to(device)
train_steps(more, torch.optim.AdamW(more.parameters(), lr=1e-4), 150, torch.Generator().manual_seed(2))
rows.append({"name": "more-steps", "params": sum(p.numel() for p in more.parameters()), "test": evaluate(more, test, payload["bytes"]["test"]), "domains": domains(more)})
del more
torch.cuda.empty_cache()

# 3. Net2Net: duplicar parte del FFN y repartir los pesos.
width = make_model(6)
base_pretrain(width)
before = width(torch.tensor(payload["testTokens"][:CONTEXT], device=device)[None, :]).detach()
width.grow_ffn_width(768)
after = width(torch.tensor(payload["testTokens"][:CONTEXT], device=device)[None, :]).detach()
width.to(device)
train_steps(width, torch.optim.AdamW(width.parameters(), lr=1e-4), 100, torch.Generator().manual_seed(2))
rows.append({"name": "net2net-width", "params": sum(p.numel() for p in width.parameters()), "insertionDiff": float((before - after).abs().max().item()), "test": evaluate(width, test, payload["bytes"]["test"]), "domains": domains(width)})

result = {"variants": rows, "seconds": time.time() - started, "peakMemoryMB": torch.cuda.max_memory_allocated() / (1024 * 1024)}
Path("data/gpu/results-f14-variants.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(result)
