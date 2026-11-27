"""Aplica las correcciones detectadas en la revisión y produce una versión auditada.

No sobrescribe el CSV original: escribe `morphology-review-fixed.csv` y
`morphology-reference-reviewed.json`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data" / "morphology-review.csv"
FIXED = ROOT / "data" / "morphology-review-fixed.csv"
REFERENCE = ROOT / "data" / "morphology-reference.json"
REVIEWED_REFERENCE = ROOT / "data" / "morphology-reference-reviewed.json"
DECISIONS = ROOT / "data" / "morphology-review-decisions.json"

REJECTED = {
    ("adrede", "lazarillo"),
    ("huelguen", "lazarillo"),
    ("maldita", "lazarillo"),
    ("-que", "quijote"),
    ("yegua", "quijote"),
}

CORRECTIONS = {
    ("salgamos", "lazarillo"): {
        "stem": "salg", "suffix": "amos", "tense": "presente", "confidence": "alta",
        "correction": "tallo corregido: salg + amos",
    },
    ("remediaran", "lazarillo"): {
        "stem": "remedi", "suffix": "aran", "tense": "pasado", "confidence": "alta",
        "correction": "sufijo corregido: aran; pretérito imperfecto de subjuntivo",
    },
    ("comenzamos", "quijote"): {
        "stem": "comenz", "suffix": "amos", "tense": "ambiguo", "confidence": "alta",
        "correction": "tallo corregido: comenz + amos; presente/pretérito ambiguo",
    },
    ("ponérsele", "quijote"): {
        "stem": "", "suffix": "ponérsele", "tense": "ambiguo", "confidence": "baja",
        "correction": "enclítico: no se separa tallo/sufijo",
    },
    ("pegan", "lazarillo"): {
        "stem": "peg", "suffix": "an", "tense": "presente", "confidence": "alta",
        "correction": "sufijo corregido: an",
    },
    ("ponme", "lazarillo"): {
        "stem": "", "suffix": "ponme", "tense": "ambiguo", "confidence": "baja",
        "correction": "enclítico: no se separa tallo/sufijo",
    },
    ("quedase", "lazarillo"): {
        "stem": "qued", "suffix": "ase", "tense": "pasado", "confidence": "alta",
        "correction": "tallo/sufijo corregidos: qued + ase",
    },
    ("entendiera", "quijote"): {
        "stem": "entend", "suffix": "iera", "tense": "pasado", "confidence": "alta",
        "correction": "sufijo corregido: iera",
    },
}


def key_of(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("lower", "")).casefold(),
        str(row.get("source", "")),
        str(row.get("sentence", "")).strip(),
    )


def main() -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    originals = reference["records"]
    with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_key = {key_of(record): record for record in originals}
    by_normalized: dict[tuple[str, str], dict] = {}
    for record in originals:
        normalized = record["lower"].lstrip("-–— ")
        by_normalized.setdefault((normalized, record["source"]), record)

    fixed_rows = []
    reviewed_records = []
    seen_originals: set[tuple[str, str, str]] = set()
    notes = [
        {"form": form, "decision": "rechazada"}
        for form, source in sorted(REJECTED)
    ]
    for row in rows:
        lower = (row.get("lower") or "").strip().lower()
        source = (row.get("source") or "").strip()
        current = dict(row)
        if (lower, source) in REJECTED:
            notes.append({"form": row.get("form"), "decision": "rechazada"})
            continue
        if (lower, source) in CORRECTIONS:
            current.update(CORRECTIONS[(lower, source)])
            notes.append({"form": current.get("form"), "decision": "corregida"})
        else:
            notes.append({"form": current.get("form"), "decision": "mantenida"})

        original = by_key.get(key_of(current))
        if original is None:
            original = by_normalized.get((lower.lstrip("-–— "), source))
        if original is not None:
            seen_originals.add(key_of(original))
            merged = dict(original)
            merged.update(
                {
                    "form": current.get("form"),
                    "lower": current.get("lower"),
                    "stem": current.get("stem"),
                    "suffix": current.get("suffix"),
                    "tense": current.get("tense"),
                    "confidence": current.get("confidence"),
                    "reviewed": True,
                    "correction": current.get("correction", ""),
                }
            )
            reviewed_records.append(merged)
        fixed_rows.append(current)

    with FIXED.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(fixed_rows)

    kept = [record for record in originals if key_of(record) not in seen_originals]
    accepted = [
        record for record in originals
        if (record["lower"], record["source"]) not in REJECTED
    ]
    # Los revisados reemplazan a su original; los no revisados se conservan.
    replacements = {key_of(record): record for record in reviewed_records}
    final_records = [replacements.get(key_of(record), record) for record in accepted]
    final_records.extend(record for record in reviewed_records if key_of(record) not in {key_of(item) for item in accepted})

    payload = {
        "kind": "morphology-reference-reviewed",
        "version": 1,
        "model": reference["model"],
        "spacy": reference["spacy"],
        "note": "Revisión humana de 100 decisiones: 95 formas conservadas y 5 rechazadas.",
        "counts": {
            "reviewedDecisions": len(notes),
            "kept": len(reviewed_records),
            "rejected": len([note for note in notes if note["decision"] == "rechazada"]),
            "corrected": len([note for note in notes if note["decision"] == "corregida"]),
            "total": len(final_records),
        },
        "records": final_records,
    }
    REVIEWED_REFERENCE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DECISIONS.write_text(json.dumps({"decisions": notes, "rejected": sorted(REJECTED)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
