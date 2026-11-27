# Sampling

## 1. Qué problema resuelve

El modelo devuelve logits, no tokens. **Cómo** se elige el siguiente token cambia la diversidad, la
repetición y la calidad percibida. Sampling es el conjunto de reglas que convierten logits en una
decisión.

## 2. Definición y notación

- **Temperatura** $T$: divide los logits antes del softmax.
  - $T < 1$: distribución más afilada, menos diversidad.
  - $T > 1$: más plana, más diversidad y más ruido.
- **Top-k**: se queda con los $k$ logits más altos.
- **Top-p** (nucleus): se queda con el menor conjunto cuya probabilidad acumulada llega a $p$.

Orden aplicado: temperatura → top-k → top-p → muestreo.

## 3. Por qué funciona

- Top-k corta la cola de tokens improbables que producen basura.
- Top-p adapta el corte a la forma de la distribución: si el modelo está seguro, deja pocos; si duda,
  deja más.
- La temperatura no cambia el orden de los tokens, solo la forma de la distribución.

## 4. Ejemplo medido

Mismo prompt y modelo de 1,4M, 48 tokens:

| Configuración | Muestra |
| --- | --- |
| Greedy (top-k 1) | `aaaaaoadoooooooooooe oio io o o o o o ... que , que ...` |
| Temperatura 0.3 | `aa m, ad�timad�oocy adini orespiomma�o�aa la ú...` |
| Temperatura 0.8 | `ay �qcpvios Qla gorí cong... había y ó 7o, laal...` |
| Temperatura 1.2 | `n e étodresp/as, aqu��el ando en �inpr�o...` |
| Top-k 20 | `aao y , que n y , que opcas�o me 7las que...` |
| Top-p 0.9 | `ay ig&cames es liden )br�qui�cver...` |

- Greedy se queda en un bucle de `a` y `que`.
- Subir la temperatura aumenta la variedad, no la calidad.
- Top-k/top-p no arreglan un modelo poco entrenado; solo cambian el reparto.

## 5. Implementación

- [`src/nn/sampling.ts`](../../src/nn/sampling.ts): `sampleFromValues` con temperatura, top-k y top-p.
- [`src/models/transformer-lm.ts`](../../src/models/transformer-lm.ts): `generate` acepta `topK` y
  `topP`.
- [`src/models/char-lm.ts`](../../src/models/char-lm.ts): las mismas opciones para la RNN.
- [`lab/transformer/sampling.ts`](../../lab/transformer/sampling.ts): seis configuraciones.
- [`data/sampling-report.json`](../../data/sampling-report.json): textos completos.

## 6. Verificación

- Top-k 1 devuelve siempre el argmax.
- Top-p 0.5 con `[10, 0]` devuelve el índice 0.
- `topK=0`, `topP=0` y `topP>1` fallan.
- 172/172 pruebas y `verify.mjs` en verde.
- La generación con KV cache y la generación normal siguen coincidiendo.

## 7. Límites y errores comunes

- **Temperatura alta no es creatividad**: es más ruido.
- **Top-k fijo** puede cortar una distribución con varias opciones válidas; top-p se adapta mejor.
- **Sampling no cambia tokens/s**: el coste está en el modelo, no en ordenar 512 logits.
- **El modelo es el de F6**: sus muestras siguen siendo fragmentos; sampling no lo arregla.
- **Reproducibilidad**: misma semilla y mismas opciones dan el mismo texto.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Holtzman et al. (2020), *The Curious Case of Neural Text Degeneration* (nucleus sampling).
- Fan et al. (2018), *Hierarchical Neural Story Generation* (top-k).
