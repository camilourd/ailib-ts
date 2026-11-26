# Inferencia

## 1. Qué problema resuelve

Entrenar es una parte; **usar** el modelo es otra. En generación autorregresiva cada token nuevo vuelve
a pasar toda la secuencia por el transformer. Eso es $O(L^2)$ por token y se nota. Antes de optimizar
hay que **medir**: tokens/s, latencia y memoria.

## 2. Definición y notación

- **Tokens/s**: tokens generados por segundo, sin contar la carga del checkpoint.
- **Latencia**: milisegundos por los $N$ tokens generados.
- **Memoria**: heap usado antes y después; en JS es una foto, no una medida exacta.
- **Misma calidad**: mismo vocabulario, misma perplejidad o mismos logits dentro de tolerancia.

## 3. Línea base

- Modelo: `TransformerLM` de **1.401.792 parámetros**, checkpoint del paso 1000.
- Prompt: `"En un lugar de la Mancha"` (10 tokens BPE).
- 48 tokens generados, temperatura 0.8, semilla 7.
- Tres corridas después de un warmup de 8 tokens.
- Sin caché: cada paso recalcula Q, K y V de toda la secuencia.

| Corrida | Tiempo | Tokens/s |
| --- | ---: | ---: |
| 1 | 6037 ms | 7.95 |
| 2 | 5854 ms | 8.20 |
| 3 | 5882 ms | 8.16 |
| **Media** | **5924 ms** | **8.10** |

- Heap antes/después: 64,6 → 8,2 MB (el recolector interviene; no es una fuga).
- Muestra: fragmentos de bytes; el modelo sigue siendo el de F6.

## 4. Por qué el baseline es lento

En cada paso se procesan $L$ posiciones. La atención compara $L$ queries con $L$ claves; al crecer la
secuencia, el coste crece cuadráticamente. Para generar 48 tokens se hicieron 48 pasadas completas.

La optimización evidente es **KV cache**: guardar claves y valores ya calculados y, en cada paso,
procesar solo el token nuevo. La calidad debe quedar igual; la mejora se mide contra esta tabla.

## 5. Implementación

- [`lab/transformer/benchmark.ts`](../../lab/transformer/benchmark.ts): warmup, tres corridas,
  memoria y muestra.
- [`src/models/transformer-lm.ts`](../../src/models/transformer-lm.ts): `generate` sin caché.
- [`data/inference-baseline.json`](../../data/inference-baseline.json): resultados crudos.

## 6. Verificación

- La misma entrada, temperatura y semilla en las tres corridas.
- La varianza entre corridas es baja (7.95–8.20 tokens/s).
- El baseline queda guardado antes de tocar la inferencia.

## 7. Límites y errores comunes

- **Medir en Node/CPU**, no en GPU: el número no se compara con otros frameworks.
- **Heap de JS**: la memoria no baja necesariamente al liberar tensores.
- **Sin caché** el resultado es correcto, solo lento.
- **Optimizar sin medir** es cambiar una intuición por otra.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Pope et al. (2023), *Efficiently Scaling Transformer Inference*.
- Documentación de KV cache en Llama/GPT-NeoX.
