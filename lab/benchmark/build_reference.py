"""F10.0: construye una referencia morfológica verbal para Quijote y Lazarillo.

La referencia es una hipótesis: spaCy aporta POS/tiempo/lema; el tallo y el sufijo se derivan del
lema y de la forma. Las formas irregulares quedan con confianza baja. Una muestra de 100 formas se
guarda aparte para revisión humana.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data" / "quijote.txt"
EXTERNAL = ROOT / "data" / "lazarillo.txt"
REFERENCE = ROOT / "data" / "morphology-reference.json"
REVIEW = ROOT / "data" / "morphology-review.csv"
TARGET = 800
PER_SOURCE = 400

TENSE = {"Past": "pasado", "Pres": "presente", "Fut": "futuro"}


def analyze(text: str, source: str, nlp, seen: set[tuple[str, str]], records: list[dict]) -> None:
    for document in nlp.pipe((text[index : index + 4000] for index in range(0, len(text), 4000)), batch_size=16):
        for token in document:
            if token.pos_ != "VERB":
                continue
            if (token.morph.get("VerbForm") or [""])[0] != "Fin":
                continue
            tense = (token.morph.get("Tense") or [""])[0]
            if tense not in TENSE:
                continue
            lemma = token.lemma_.lower()
            if not lemma.endswith(("ar", "er", "ir")):
                continue
            form = token.text
            lower = form.lower()
            stem = lemma[:-2]
            suffix = lower[len(stem) :] if lower.startswith(stem) else ""
            if not suffix:
                stem = ""
                suffix = lower
            confidence = "alta" if stem and suffix else "media"
            if not stem:
                confidence = "baja"
            key = (lower, TENSE[tense], source)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "form": form,
                    "lower": lower,
                    "lemma": lemma,
                    "stem": stem,
                    "suffix": suffix,
                    "tense": TENSE[tense],
                    "mood": (token.morph.get("Mood") or ["-"])[0],
                    "person": (token.morph.get("Person") or ["-"])[0],
                    "verb_form": (token.morph.get("VerbForm") or ["-"])[0],
                    "confidence": confidence,
                    "source": source,
                    "sentence": token.sent.text.strip(),
                }
            )
            if sum(record["source"] == source for record in records) >= PER_SOURCE:
                return


def main() -> None:
    nlp = spacy.load("es_core_news_sm")
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    analyze(TRAIN.read_text(encoding="utf-8"), "quijote", nlp, seen, records)
    analyze(EXTERNAL.read_text(encoding="utf-8"), "lazarillo", nlp, seen, records)

    payload = {
        "kind": "morphology-reference",
        "version": 1,
        "model": "es_core_news_sm",
        "spacy": spacy.__version__,
        "note": "Hipótesis: tallo/sufijo derivados del lema; las formas irregulares requieren revisión.",
        "counts": {
            "total": len(records),
            "quijote": sum(record["source"] == "quijote" for record in records),
            "lazarillo": sum(record["source"] == "lazarillo" for record in records),
            "alta": sum(record["confidence"] == "alta" for record in records),
            "media": sum(record["confidence"] == "media" for record in records),
            "baja": sum(record["confidence"] == "baja" for record in records),
            "pasado": sum(record["tense"] == "pasado" for record in records),
            "presente": sum(record["tense"] == "presente" for record in records),
            "futuro": sum(record["tense"] == "futuro" for record in records),
        },
        "records": records,
    }
    REFERENCE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rng = random.Random(17)
    review = rng.sample(records, min(100, len(records)))
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "form", "lower", "lemma", "stem", "suffix", "tense", "mood", "person",
                "verb_form", "confidence", "source", "sentence", "reviewed", "correction",
            ],
        )
        writer.writeheader()
        for record in review:
            writer.writerow({**record, "reviewed": "no", "correction": ""})
    print(json.dumps({**payload["counts"], "review": len(review)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
