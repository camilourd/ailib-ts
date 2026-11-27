"""Audita la revisión humana del CSV morfológico contra la referencia original."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data" / "morphology-review.csv"
REFERENCE = ROOT / "data" / "morphology-reference.json"
AUDIT = ROOT / "data" / "morphology-review-audit.json"
TENSES = {"pasado", "presente", "futuro", "ambiguo"}


def key_of(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("lower", "")).casefold(),
        str(record.get("source", "")),
        str(record.get("sentence", "")).strip(),
    )


def main() -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    originals = reference["records"]
    expected = random.Random(17).sample(originals, min(100, len(originals)))
    expected_keys = {key_of(record) for record in expected}

    with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    current_keys = [key_of(row) for row in rows]
    duplicates = sorted({key for key in current_keys if current_keys.count(key) > 1})

    by_key: dict[tuple[str, str], list[dict]] = {}
    for record in originals:
        by_key.setdefault((record["lower"], record["source"]), []).append(record)

    issues: list[dict] = []
    reviewed = 0
    corrections = 0
    for index, row in enumerate(rows, start=2):
        form = (row.get("form") or "").strip()
        lower = (row.get("lower") or "").strip().lower()
        source = (row.get("source") or "").strip()
        stem = (row.get("stem") or "").strip()
        suffix = (row.get("suffix") or "").strip()
        tense = (row.get("tense") or "").strip().lower()
        reviewed_value = (row.get("reviewed") or "").strip().lower()
        correction = (row.get("correction") or "").strip()
        problems = []

        if reviewed_value in {"si", "sí", "yes", "true", "1"}:
            reviewed += 1
        else:
            problems.append("sin revisar")
        if correction:
            corrections += 1
        if tense not in TENSES:
            problems.append(f"tiempo inválido: {tense!r}")
        if form and stem and suffix and (stem + suffix).casefold() != form.casefold():
            problems.append(f"tallo+sufijo no reconstruye: {stem!r}+{suffix!r} != {form!r}")
        if not stem or not suffix:
            problems.append("tallo o sufijo vacío")

        matches = by_key.get((lower, source), [])
        original = matches[0] if len(matches) == 1 else None
        if original is None and matches:
            sentence = (row.get("sentence") or "").strip()
            original = next((item for item in matches if item["sentence"] == sentence), None)
        if original is None:
            problems.append("no se encuentra en la referencia")
        else:
            changed = []
            for field in ("stem", "suffix", "tense"):
                current = (row.get(field) or "").strip().lower()
                before = str(original[field]).lower()
                if current != before:
                    changed.append({"field": field, "before": before, "after": current})
            if correction and not changed:
                problems.append("corrección escrita pero tallo/sufijo/tiempo sin cambiar")
            if changed and not correction:
                problems.append("campos cambiados sin nota")

        if problems:
            issues.append({"line": index, "form": form, "source": source, "problems": problems})

    current_set = set(current_keys)
    missing = [
        {"form": record["form"], "lower": record["lower"], "tense": record["tense"], "source": record["source"]}
        for record in expected
        if key_of(record) not in current_set
    ]
    unexpected = [
        {"form": row.get("form"), "lower": row.get("lower"), "source": row.get("source")}
        for row in rows
        if key_of(row) not in expected_keys
    ]

    payload = {
        "rows": len(rows),
        "expectedRows": len(expected),
        "reviewed": reviewed,
        "pending": len(rows) - reviewed,
        "corrections": corrections,
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": [list(key) for key in duplicates],
        "issues": issues,
        "issueCount": len(issues),
    }
    AUDIT.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("rows", "expectedRows", "reviewed", "corrections", "issueCount")}, indent=2))
    print(f"missing={len(missing)} unexpected={len(unexpected)} duplicates={len(duplicates)}")


if __name__ == "__main__":
    main()
