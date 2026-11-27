"""Comprueba la integridad de todos los artefactos F10."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
problems: list[str] = []


def load(name: str):
    path = ROOT / "data" / name
    if not path.exists():
        problems.append(f"falta {name}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    reference = load("morphology-reference-reviewed.json")
    if reference:
        counts = reference["counts"]
        if counts["total"] != 795:
            problems.append(f"reference total {counts['total']} != 795")
        if counts["reviewedDecisions"] != 100:
            problems.append("decisiones != 100")
        if counts["rejected"] != 5:
            problems.append("rechazadas != 5")
        if any(record["lower"] in {"adrede", "huelguen", "maldita", "-que", "yegua"} for record in reference["records"]):
            problems.append("quedó una forma rechazada")

    decisions = load("morphology-review-decisions.json")
    if decisions and len(decisions["decisions"]) != 100:
        problems.append("decisiones del CSV != 100")

    audit = load("morphology-review-audit.json")
    if audit and audit["issueCount"] != 27:
        problems.append("la auditoría original cambió de tamaño")

    benchmark = load("tokenizer-benchmark.json")
    if benchmark:
        names = {row["name"] for row in benchmark["rows"]}
        if names != {"bpe-2048", "bpe-2048-suffix", "bpe-2048-prefix-suffix", "wordpiece-2048", "unigram-2048", "sentencepiece-2048", "byte-256", "byte-patch-2", "byte-patch-3", "palabra", "caracter"}:
            problems.append(f"variantes inesperadas: {names}")
        for row in benchmark["rows"]:
            if not row["exact"]:
                problems.append(f"{row['name']} no es exacto")
            if not (0 <= row["boundary"]["f1"] <= 1):
                problems.append(f"{row['name']} F1 fuera de rango")
            if not math.isfinite(row["bpbProxy"]):
                problems.append(f"{row['name']} BPB no finito")

    probe = load("tense-probe.json")
    if probe:
        if len(probe) != 3:
            problems.append("sonda no tiene 3 variantes")
        for row in probe:
            if not (0 <= row["accuracyQuijote"] <= 1 and 0 <= row["accuracyExternal"] <= 1):
                problems.append(f"{row['name']} accuracy fuera de rango")

    evolution = load("tokenizer-evolution-f10.json")
    if evolution:
        for key in ("randomBest", "sge"):
            row = evolution[key]
            expected = row["tokensPerChar"] + 0.5 * (1 - row["boundaryF1"]) + 0.1 * row["editDistance"]
            if abs(expected - row["score"]) > 1e-9:
                problems.append(f"{key} score no coincide")
            if not row["exact"]:
                problems.append(f"{key} no es exacto")

    with (ROOT / "data" / "morphology-review-fixed.csv").open(encoding="utf-8-sig", newline="") as handle:
        fixed = list(csv.DictReader(handle))
    if len(fixed) != 95:
        problems.append(f"CSV corregido tiene {len(fixed)} filas")

    print(json.dumps({"problems": problems, "ok": len(problems) == 0}, ensure_ascii=True, indent=2))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
