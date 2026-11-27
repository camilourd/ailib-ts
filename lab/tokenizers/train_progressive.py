# F14.2: 6 capas -> insertar 2 identity -> 8 capas, comparado con 8 desde cero.
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train_gpu_fair import CONTEXT, TransformerLM, evaluate

DATA = Path("data/gpu")
payload = json.loads((DATA / "f14-mixture-bpe-2k.json").read_text(encoding="utf-8"))
device = torch.device("cuda")
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
model = TransformerLM(
    payload["vocabulary"], 6, 1536, 384, 8,
    activation_checkpoint=True,
).to(device)
data = torch.tensor(payload["trainTokens"], dtype=torch.long, device=device)
test = torch.tensor(payload["testTokens"], dtype=torch.long, device=device)
external = torch.tensor(payload["externalTokens"], dtype=torch.long, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
generator = torch.Generator().manual_seed(1)
torch.cuda.reset_peak_memory_stats()


def train_steps(steps: int) -> None:
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
            print(f"progresivo paso {step}: {loss.item():.4f}", flush=True)


def domain_bpb() -> dict[str, float]:
    return {
        name: evaluate(model, torch.tensor(tokens, device=device), payload["bytes"][name])
        for name, tokens in payload["testSets"].items()
    }


started = time.time()
train_steps(50)
bpb6 = domain_bpb()
probe = torch.tensor(payload["testTokens"][:CONTEXT], device=device)[None, :]
before = model(probe).detach()
model.grow_depth(2, zero_init=True)
model.to(device)
after = model(probe).detach()
insertion_diff = float((before - after).abs().max().item())
print(f"inserción: diff={insertion_diff} params={sum(p.numel() for p in model.parameters())}")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
torch.cuda.reset_peak_memory_stats()
train_steps(50)
final = {
    "name": "f14-progresivo-6a8",
    "seed": 1,
    "params": sum(parameter.numel() for parameter in model.parameters()),
    "insertionDiff": insertion_diff,
    "domainBpb6": bpb6,
    "domainBpb8": domain_bpb(),
    "bpb6": evaluate(model, external, payload["bytes"]["external"]),
    "bpb8_test": evaluate(model, test, payload["bytes"]["test"]),
    "seconds": time.time() - started,
    "peakMemoryMB": torch.cuda.max_memory_allocated() / (1024 * 1024),
}
Path("data/gpu/results-f14-progressive.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
print(final)
