# Variantes del transformer

## RMSNorm

LayerNorm centra (resta la media) y divide por la desviación. RMSNorm se queda con el RMS:

```
rms(x) = sqrt(mean(x²) + ε)
y = x / rms(x) · gamma
```

No tiene media ni `beta`. Es más barata y suele funcionar igual de bien en transformers modernos.
El gradiente de entrada es:

```
dx = inv · dNorm − x · (Σ dNorm·x) · inv³ / n
```

## SwiGLU

El FFN clásico es `Linear → ReLU → Linear`. SwiGLU usa una compuerta:

```
y = W2( SiLU(W1 x) ⊙ W3 x )
SiLU(x) = x · σ(x)
```

La compuerta `W3` decide cuánto pasa de la rama `W1`. Suele necesitar menos dimensión oculta que un
FFN ReLU para el mismo coste.

## NoPE

Sin posición explícita: la atención causal ya rompe la simetría por el orden de los tokens. En este
proyecto se activa con `rope: false`.

## Paralelo

En vez de atención y FFN en serie:

```
x = x + Attention(Norm(x))
y = x + FFN(Norm(x))
```

se suman las dos ramas desde la misma entrada. Ahorra una norma y permite más paralelismo; en las
pruebas no mejora al bloque clásico.

## Pre-norm y post-norm

- **Pre-norm:** `x + Attention(Norm(x))`; más estable al entrenar.
- **Post-norm:** `Norm(x + Attention(x))`; el transformer original.

En las pruebas a 5,8M/100 pasos son casi indistinguibles.

## Resultados

Con 5,8M, 100 pasos, contexto 128 y BPE:

| Variante | BPB test | BPB externo |
| --- | ---: | ---: |
| Clásico pre-norm | 2,859 | 3,058 |
| Post-norm | 2,855 | 3,054 |
| Modern (RMSNorm+SwiGLU) | **2,823** | **3,024** |
| NoPE | 2,858 | 3,058 |
| Paralelo | 2,855 | 3,055 |

Modern gana, pero tiene 7,36M de parámetros frente a 5,79M. Una sola semilla, 50k de entrenamiento.
Los detalles están en `docs/experimentos/familias-tokenizers.md`.
