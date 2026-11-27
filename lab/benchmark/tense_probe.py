"""F10.2: sonda de tiempo verbal sobre el LM congelado.

Entrena el mismo transformer para cada tokenizer, congela y ajusta una regresión logística sobre el
estado oculto en la posición del verbo. Entrena con Quijote y evalúa en Quijote retenido y Lazarillo.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab" / "tokenizers"))
from train_gpu import TransformerLM  # noqa: E402

DATA = ROOT / "data" / "gpu"
RESULT = ROOT / "data" / "tense-probe.json"
CONTEXT = 128
STEPS = 400
PROBE_STEPS = 600
LABELS = ["pasado", "presente", "futuro"]
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}


def train_lm(name: str, device: torch.device) -> TransformerLM:
    payload = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    model = TransformerLM(payload["vocabulary"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    data = torch.tensor(payload["trainTokens"], dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(1)
    model.train()
    for step in range(STEPS):
        starts = torch.randint(0, len(data) - CONTEXT - 1, (16,), generator=generator).to(device)
        x = torch.stack([data[start : start + CONTEXT] for start in starts])
        y = torch.stack([data[start + 1 : start + CONTEXT + 1] for start in starts])
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def features(model: TransformerLM, records: list[dict], device: torch.device):
    model.eval()
    values = []
    labels = []
    with torch.no_grad():
        for record in records:
            tokens = record["tokens"]
            if len(tokens) < 2:
                continue
            position = record["position"]
            start = max(0, position - CONTEXT + 1)
            window = tokens[start : start + CONTEXT]
            if position - start >= len(window):
                continue
            x = torch.tensor(window, dtype=torch.long, device=device)[None, :]
            hidden = model.hidden(x)[0, position - start]
            values.append(hidden)
            labels.append(LABEL_INDEX[record["tense"]])
    return torch.stack(values), torch.tensor(labels, dtype=torch.long, device=device)


def accuracy(model: nn.Linear, values: torch.Tensor, labels: torch.Tensor) -> float:
    if len(labels) == 0:
        return 0.0
    with torch.no_grad():
        return (model(values).argmax(dim=-1) == labels).float().mean().item()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no disponible")
    device = torch.device("cuda")
    results = []
    for name in ("bpe-2048", "bpe-2048-suffix", "caracter"):
        records = json.loads((DATA / f"probe-{name}.json").read_text(encoding="utf-8"))
        records = [record for record in records if record["tense"] in LABEL_INDEX]
        train_records = [record for record in records if record["source"] == "quijote"]
        cut = int(len(train_records) * 0.8)
        fit_records, test_records = train_records[:cut], train_records[cut:]
        external_records = [record for record in records if record["source"] == "lazarillo"]
        model = train_lm(name, device)
        fit_values, fit_labels = features(model, fit_records, device)
        test_values, test_labels = features(model, test_records, device)
        external_values, external_labels = features(model, external_records, device)
        mean = fit_values.mean(dim=0, keepdim=True)
        std = fit_values.std(dim=0, keepdim=True).clamp_min(1e-5)
        fit_values = (fit_values - mean) / std
        test_values = (test_values - mean) / std
        external_values = (external_values - mean) / std
        probe = nn.Linear(fit_values.shape[1], len(LABELS)).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2)
        for _ in range(PROBE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(probe(fit_values), fit_labels)
            loss.backward()
            optimizer.step()
        result = {
            "name": name,
            "vocabulary": model.embedding.num_embeddings,
            "fit": len(fit_labels),
            "quijoteTest": len(test_labels),
            "external": len(external_labels),
            "accuracyQuijote": accuracy(probe, test_values, test_labels),
            "accuracyExternal": accuracy(probe, external_values, external_labels),
            "baselineMajority": max(
                int((fit_labels == index).sum()) for index in range(len(LABELS))
            ) / max(1, len(fit_labels)),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        results.append(result)
        print(
            f"{name}: Quijote {result['accuracyQuijote']:.3f} · externo "
            f"{result['accuracyExternal']:.3f} · fit {result['fit']} · ext {result['external']}"
        )
    RESULT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"guardado: {RESULT}")


if __name__ == "__main__":
    main()
