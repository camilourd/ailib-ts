# KV cache

## 1. Qué problema resuelve

En generación autorregresiva, para producir el token $t$ el transformer vuelve a calcular la atención de
todas las posiciones anteriores. Es trabajo repetido: las claves y los valores del pasado no cambian.
El **KV cache** los guarda y solo procesa el token nuevo.

## 2. Definición y notación

- Por cada capa y cada cabeza se guardan:
  - $K_{\le t}$: claves de las posiciones $0..t$;
  - $V_{\le t}$: valores de las posiciones $0..t$.
- En el paso $t+1$:
  1. se proyecta solo el token nuevo en $q, k, v$;
  2. se aplica RoPE a $q$ y $k$ con **offset** $t+1$;
  3. se añaden $k$ y $v$ a la caché;
  4. se atiende $q$ contra toda la caché.
- La máscara causal ya no hace falta en el paso: la query es una sola y todas las claves son pasado.

## 3. Por qué funciona

- Sin caché, el coste por token crece con la longitud: $O(L^2)$.
- Con caché, la atención del token nuevo es $O(L)$; el total de la generación baja de $O(L^3)$ a
  $O(L^2)$.
- La calidad no cambia: los mismos $K$ y $V$, en el mismo orden, producen los mismos logits.
- El precio es memoria: la caché crece $O(L)$ por capa y cabeza.

## 4. Resultado medido

Mismo modelo de **1.401.792 parámetros**, prompt de 10 tokens, 48 generados, temperatura 0.8 y semilla 7:

| Modo | Tiempo medio | Tokens/s |
| --- | ---: | ---: |
| Sin caché | 5736 ms | 8.34 |
| **Con KV cache** | **222 ms** | **210.06** |

- **Mejora: 25.18×**.
- La salida generada es **idéntica** token a token.
- La caché de 4 bloques × 8 cabezas × 20 dimensiones crece con la secuencia; para 58 posiciones es
  pequeña, pero no gratis.

## 5. Implementación

- [`src/nn/kv-cache.ts`](../../src/nn/kv-cache.ts): `KVCache`, `AttentionCache` y `appendRows`.
- [`src/nn/multihead.ts`](../../src/nn/multihead.ts): `forwardStep` con máscara ya resuelta y RoPE por
  posición.
- [`src/nn/attention.ts`](../../src/nn/attention.ts): atención con $Q$ de una posición y $K/V$ de
  varias; el backward se generalizó para longitudes distintas.
- [`src/nn/transformer-block.ts`](../../src/nn/transformer-block.ts): `forwardCached`.
- [`src/models/transformer-lm.ts`](../../src/models/transformer-lm.ts): `generate(..., { useCache: true })`.
- [`lab/transformer/benchmark-cache.ts`](../../lab/transformer/benchmark-cache.ts): comparación y
  verificación de igualdad.

## 6. Verificación

- `forwardStep` coincide con `forward` en cada posición (atención multi-head con RoPE).
- La generación con caché es **idéntica** a la generación sin caché.
- 171/171 pruebas, build y lint en verde.
- El benchmark corre tres veces cada modo y reporta la media.

## 7. Límites y errores comunes

- **Memoria**: la caché crece con la longitud y el número de capas.
- **Prefill**: procesar el prompt inicial sigue siendo $O(P^2)$; la mejora se nota al generar.
- **Una secuencia**: no hay batch ni beam search.
- **Posición**: el offset de RoPE debe ser el índice real; reiniciarlo rompe la equivalencia.
- **No cambia la calidad**: si cambia la salida, hay un bug en la caché o en la máscara.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Pope et al. (2023), *Efficiently Scaling Transformer Inference*.
- Documentación de `past_key_values` en Hugging Face Transformers.
