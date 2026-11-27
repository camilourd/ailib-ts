# F10.0: referencia morfológica

## Qué genera

`python lab/benchmark/build_reference.py`:

- `data/morphology-reference.json`: 800 formas verbales finitas con `form`, `lemma`, `stem`,
  `suffix`, `tense`, `mood`, `person`, `confidence`, `source` y `sentence`.
- `data/morphology-review.csv`: 100 formas para revisión humana (`reviewed=no`).

## Dependencias

```bash
python -m pip install spacy
python -m spacy download es_core_news_sm
```

## Auditoría

```bash
python lab/benchmark/build_reference.py
python lab/benchmark/audit_review.py      # detecta lo que se saltó la revisión
python lab/benchmark/repair_review.py     # produce CSV/referencia corregidos
python lab/benchmark/audit_fixed.py       # debe salir sin issues
```

## Límites

- spaCy no es perfecto: `adrede` puede aparecer como verbo.
- Las formas irregulares (`fue`, `pidió`) suelen quedar con `stem` vacío y `confidence=baja`.
- La referencia es una **hipótesis**, no una verdad lingüística.
- La revisión de 100 formas es obligatoria antes de cerrar F10.0.
