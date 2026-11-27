"""Audita la versión corregida del CSV y la referencia revisada."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXED = ROOT / "data" / "morphology-review-fixed.csv"
REFERENCE = ROOT / "data" / "morphology-reference-reviewed.json"
TENSES = {"pasado", "presente", "futuro", "ambiguo"}


def main() -> None:
    with FIXED.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    issues = []
    high = 0
    segmented = 0
    for line, row in enumerate(rows, start=2):
        form = (row.get("form") or "").strip()
        stem = (row.get("stem") or "").strip()
        suffix = (row.get("suffix") or "").strip()
        confidence = (row.get("confidence") or "").strip().lower()
        tense = (row.get("tense") or "").strip().lower()
        if (row.get("reviewed") or "").strip().lower() not in {"si", "sí", "yes", "true", "1"}:
            issues.append({"line": line, "form": form, "problem": "sin revisar"})
        if tense not in TENSES:
            issues.append({"line": line, "form": form, "problem": f"tiempo inválido {tense!r}"})
        if confidence == "alta":
            high += 1
            if stem and suffix:
                segmented += 1
            if not stem or not suffix:
                issues.append({"line": line, "form": form, "problem": "alta sin tallo/sufijo"})
            elif (stem + suffix).casefold() != form.casefold():
                issues.append({"line": line, "form": form, "problem": "tallo+sufijo no reconstruye"})
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reviewed = [record for record in reference["records"] if record.get("reviewed")]
    rejected = {"adrede", "huelguen", "maldita", "-que", "yegua"}
    still_rejected = [record for record in reference["records"] if record["lower"] in rejected]
    payload = {
        "rows": len(rows),
        "high": high,
        "segmented": segmented,
        "issues": issues,
        "referenceTotal": len(reference["records"]),
        "referenceReviewed": len(reviewed),
        "rejectedStillPresent": len(still_rejected),
        "counts": reference["counts"],
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
