# Cuantización

## 1. Qué problema resuelve

Los pesos se guardan en `float32`: 4 bytes por valor. Para un modelo de 1,4M parámetros son ~5,6 MB,
y en modelos grandes la memoria es el cuello de botella. **INT8** guarda cada peso en 1 byte y una
escala por grupo.

## 2. Definición y notación

Cuantización simétrica:

$$s = \frac{\max |x|}{127}, \qquad q = \mathrm{round}(x/s), \qquad \hat{x} = q \cdot s.$$

- $q \in [-127, 127]$: entero de 8 bits.
- $s$: escala; una por **columna** (por canal) o una sola para todo el tensor.
- La de-cuantización reconstruye $\hat{x}$ para la inferencia.

## 3. Por qué funciona

- Cada peso pesa 1 byte en vez de 4: hasta **4× menos memoria**.
- Per-channel usa una escala por columna y reduce el error cuando las columnas tienen escalas
  distintas.
- Per-tensor usa una sola escala y es más barato, pero pierde precisión si hay valores extremos.

## 4. Resultado medido

Modelo de 1.401.792 parámetros, 8 ventanas de validación de 128 tokens:

| Formato | Bytes | Ratio | Perplejidad |
| --- | ---: | ---: | ---: |
| Float32 | 5,35 MB | 1,00× | **379.30** |
| INT8 por canal | **1,36 MB** | **3,93×** | 379.34 |
| INT8 por tensor | 1,34 MB | 4,00× | 510.16 |

- Per-channel reduce el almacenamiento **3,93×** con una perplejidad prácticamente igual
  (+0,04).
- Per-tensor ahorra un poco más pero **degrada** la perplejidad a 510.16.
- La memoria es de **pesos almacenados**; no se implementó una matmul INT8, así que no se reclama
  mejora de tokens/s.

## 5. Implementación

- [`src/nn/quantization.ts`](../../src/nn/quantization.ts): `quantizeInt8`, `dequantizeInt8`,
  `quantizedBytes` y `floatBytes`.
- [`lab/transformer/quantize.ts`](../../lab/transformer/quantize.ts): cuantiza todos los pesos,
  evalúa perplejidad y restaura.
- [`test/nn/quantization.test.ts`](../../test/nn/quantization.test.ts): ida y vuelta, error y escalas
  por canal.
- [`data/quantization-report.json`](../../data/quantization-report.json): resultados crudos.

## 6. Verificación

- La ida y vuelta INT8 tiene error menor que 0,01 en el caso de prueba.
- Per-channel usa una escala por columna; per-tensor, una sola.
- La perplejidad se mide con las mismas 8 ventanas antes y después.
- 174/174 pruebas, build y lint en verde.

## 7. Límites y errores comunes

- **Memoria de almacenamiento, no de ejecución**: el modelo se de-cuantiza a float32 para evaluar.
- **Sin matmul INT8**: no hay mejora de tokens/s; solo de tamaño.
- **Sin calibración**: la escala sale del propio tensor, no de un conjunto de calibración.
- **Per-tensor puede arruinar la calidad** si hay valores extremos.
- **Cuantizar normas y sesgos** no siempre es lo mejor; aquí se hizo para medir el caso completo.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Jacob et al. (2018), *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*.
- Dettmers et al. (2022), *LLM.int8()*.
