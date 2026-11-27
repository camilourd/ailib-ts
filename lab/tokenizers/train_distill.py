# F14.3: teacher de 6 capas + crecimiento a 8 + destilación.
import json
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
data = torch.tensor(payload["trainTokens"], dtype=torch.long, device=device)
test = torch.tensor(payload["testTokens"], dtype=torch.long, device=device)
external = torch.tensor(payload["externalTokens"], dtype=torch.long, device=device)
generator = torch.Generator().manual_seed(1)


def train_steps(model, optimizer, steps: int, teacher=None, alpha: float = 0.0, temperature: float = 1.0) -> None:
    model.train()
    if teacher is not None:
        teacher.eval()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(4):
            starts = torch.randint(0, len(data) - CONTEXT - 1, (2,), generator=generator).to(device)
            x = torch.stack([data[start : start + CONTEXT] for start in starts])
            y = torch.stack([data[start + 1 : start + CONTEXT + 1] for start in starts])
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, payload["vocabulary"]), y.view(-1))
            if teacher is not None and alpha > 0:
                with torch.no_grad():
                    teacher_logits = teacher(x)
                loss = loss + alpha * F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature * temperature
            (loss / 4).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0:
            print(f"paso {step}: {loss.item():.4f}", flush=True)


def domain_bpb(model) -> dict[str, float]:
    return {
        name: evaluate(model, torch.tensor(tokens, device=device), payload["bytes"][name])
        for name, tokens in payload["testSets"].items()
    }


started = time.time()
teacher = TransformerLM(payload["vocabulary"], 6, 1536, 384, 8, activation_checkpoint=True).to(device)
optimizer = torch.optim.AdamW(teacher.parameters(), lr=3e-4)
train_steps(teacher, optimizer, 100)
teacher_bpb = evaluate(teacher, test, payload["bytes"]["test"])
teacher_domains = domain_bpb(teacher)
print("teacher", teacher_bpb, teacher_domains)

student = TransformerLM(payload["vocabulary"], 8, 1536, 384, 8, activation_checkpoint=True).to(device)
student.load_state_dict(teacher.state_dict(), strict=False)
for block in student.blocks[6:]:
    block.zero_output()
student.to(device)
probe = test[:CONTEXT][None, :]
insertion_diff = float((teacher(probe).detach() - student(probe).detach()).abs().max().item())
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
train_steps(student, optimizer, 100, teacher=teacher, alpha=0.5, temperature=1.0)

result = {
    "name": "f14-distill-6to8",
    "seed": 1,
    "insertionDiff": insertion_diff,
    "params": sum(p.numel() for p in student.parameters()),
    "teacherBpb": teacher_bpb,
    "teacherDomains": teacher_domains,
    "studentBpb": evaluate(student, test, payload["bytes"]["test"]),
    "studentDomains": domain_bpb(student),
    "seconds": time.time() - started,
    "peakMemoryMB": torch.cuda.max_memory_allocated() / (1024 * 1024),
}
Path("data/gpu/results-f14-distill.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(result)
