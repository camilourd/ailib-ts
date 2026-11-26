# Embeddings

## 1. Qué problema resuelve

Un token es un índice discreto: el 7 no está «cerca» del 8. El modelo necesita vectores continuos que
pueda sumar, multiplicar y ajustar con gradiente. Un **embedding** es una tabla que aprende un vector
por token.

## 2. Definición y notación

- Tabla `weights` de forma $[\text{vocabulario}, \text{dimensión}]$.
- `forward(indices)` devuelve una fila por índice: forma `[n, dimensión]`.
- `row(indice)` devuelve `[1, dimensión]`, útil para procesar una secuencia paso a paso.

## 3. Forward y backward

- **Forward**: *gather*; se copian las filas elegidas.
- **Backward**: *scatter*; el gradiente de cada fila de salida se **suma** en la fila del token
  correspondiente. Si un token aparece dos veces, recibe el doble.

Es un gradiente **disperso**: solo se tocan las filas usadas; el resto no cambia en ese paso.

## 4. Ejemplo a mano

Tabla de 3 tokens y dimensión 2:

| token | vector |
| --- | --- |
| 0 | [1, 2] |
| 1 | [3, 4] |
| 2 | [5, 6] |

`forward([0, 2])` → `[[1, 2], [5, 6]]`. Si la pérdida suma todas las salidas, el gradiente de la
tabla es `[[1, 1], [0, 0], [1, 1]]`: las filas 0 y 2 reciben uno, la 1 no se toca.

## 5. Implementación

- [`src/nn/embedding.ts`](../../src/nn/embedding.ts): `Embedding(vocabulario, dimensión, { seed })` con
  inicialización uniforme pequeña, `forward`, `row` y `parameters`.
- [`test/nn/embedding.test.ts`](../../test/nn/embedding.test.ts): filas, gradiente disperso,
  reproducibilidad y validaciones.
- [`lab/embedding/run.ts`](../../lab/embedding/run.ts): texto → bytes → vectores y filas con gradiente.

## 6. Verificación

- `forward([0, 2])` devuelve las filas 0 y 2 con forma `[2, 2]`.
- `sum(forward([0, 2, 0])).backward()` deja el gradiente `[2, 2, 0, 0, 1, 1]`.
- La misma semilla da la misma tabla.
- Vocabulario o dimensión no positivos e índices fuera de rango fallan.
- 114/114 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Sin información posicional**: el embedding no sabe dónde está el token; eso lo aportará la posición
  (F5).
- **Vocabulario fijo**: un token nuevo exige ampliar la tabla.
- **Sin ataduras de pesos** (*weight tying*) con la salida.
- **Solo índices válidos**: no hay `<unk>`; con byte-level todo texto tiene token.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Bengio et al. (2003), *A Neural Probabilistic Language Model* (embeddings aprendidos).
- Mikolov et al. (2013), *word2vec*.
- Documentación de `torch.nn.Embedding`.
