# Destilación

La destilación entrena un modelo **student** para imitar a un modelo **teacher**. No es copiar
pesos: es copiar la **distribución de probabilidad** que produce el teacher.

## La idea

Sin destilación, el student aprende solo de las etiquetas duras:

```
token correcto = 1
resto = 0
```

Con destilación, aprende también de las probabilidades del teacher:

```
P_teacher(siguiente token | contexto)
```

Eso le da información sobre clases parecidas: si el teacher duda entre “casa” y “casita”, el student
aprende esa relación, no solo que “casa” era la respuesta.

## Pérdida

```
L = CE(student, y) + α · T² · KL( softmax(student/T) || softmax(teacher/T) )
```

| Símbolo | Qué es |
| --- | --- |
| `CE` | Cross-entropy normal contra el token correcto |
| `KL` | Divergencia entre las distribuciones del student y el teacher |
| `T` | Temperatura; `T > 1` suaviza las distribuciones |
| `α` | Peso de la destilación |
| `T²` | Compensa la escala del gradiente al suavizar |

Casos límite:

- `α = 0`: entrenamiento normal.
- `T = 1`: imita al teacher sin suavizar.
- `α` grande: el student prioriza imitar al teacher por encima del corpus.

## Por qué funciona

- Las probabilidades del teacher contienen **relaciones entre clases** que la etiqueta dura no tiene.
- El student recibe una señal más rica por token.
- Puede aprender con menos datos o más rápido.
- También sirve para comprimir: teacher grande → student pequeño.

## En este proyecto

En F14.3:

1. Teacher de 6 capas, `d_model=384`, `d_ff=1536`.
2. Student de 8 capas: copia las 6 capas y añade 2 con zero-init identity.
3. El student entrena con `CE + 0,5 · KL(T=1)` contra el teacher congelado.

Resultado:

| Modelo | BPB test | Código |
| --- | ---: | ---: |
| Teacher 6 capas | 2,876 | 2,746 |
| Student 8 destilado | 2,864 | 2,709 |
| 8 desde cero, 150 pasos | **2,802** | **2,490** |

La destilación mejora al teacher y al progresivo sin destilar, pero **no alcanza al modelo entrenado
desde cero con más pasos**.

## Limitaciones

- El teacher también es finito: no puede enseñar lo que no sabe.
- Si el teacher y el student tienen capacidades muy distintas, la imitación puede limitar al student.
- Cuesta cómputo extra: hay que entrenar al teacher y hacer forward de ambos.
- No sustituye datos ni pasos: es una ayuda, no magia.

## Cuándo usarla

- Comprimir un modelo grande en uno pequeño.
- Aprovechar un teacher ya entrenado.
- Estabilizar el crecimiento de profundidad.
- Transferir a un dominio con pocos datos.

## Código

- [train_distill.py](../lab/tokenizers/train_distill.py)
- [moe.ts](../../src/nn/moe.ts) no tiene relación directa; la destilación es una pérdida de
  entrenamiento, no una arquitectura.
- La pérdida se implementa en el script de entrenamiento con `torch.nn.functional.kl_div`.
