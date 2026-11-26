# Checkpoints

## 1. Qué problema resuelve

Entrenar cuesta tiempo y tokens. Si el proceso termina, el modelo se pierde. Un **checkpoint** guarda
los parámetros y los metadatos suficientes para reconstruir el modelo y seguir donde se quedó.

## 2. Definición y formato

Un checkpoint tiene dos partes:

- **Parámetros**: un tensor por cada peso o sesgo, con su nombre.
- **Metadatos**: lo que hace falta para interpretarlos (modelo, arquitectura, semilla, paso, pérdida,
  fecha).

El formato es JSON versionado:

```json
{
  "format": "ailib-ts.checkpoint",
  "version": 1,
  "metadata": { "modelo": "xor", "capas": [2, 4, 2], "seed": 1, "paso": 3000 },
  "parameters": [
    { "name": "hidden.weights", "shape": [2, 4], "data": [0.1, -0.2, ...] }
  ]
}
```

Los valores de `data` son números; al cargarlos vuelven a `Float32Array`.

## 3. Por qué así

- **JSON**: inspeccionable, portable y sin dependencias. Para modelos grandes se cambiará a binario
  (F6), pero ahora prima poder mirar el archivo.
- **Versión y `format`**: un lector futuro puede rechazar lo que no entiende en vez de leer mal.
- **Escritura atómica**: se escribe a `ruta.tmp` y se renombra, para no dejar un archivo a medias si
  el proceso falla.
- **Validación al cargar**: formato, versión, formas, longitudes, nombres duplicados y valores no
  finitos. Un checkpoint corrupto falla con un error claro.
- **Sin `eval`**: se lee JSON, no se ejecuta código. Aun así, no se carga un checkpoint de una fuente
  que no se confía.

## 4. Ejemplo a mano

Un checkpoint de una capa `2 → 2` guarda `weights` con forma `[2, 2]` (cuatro números) y `bias` con
forma `[2]` (dos). Al cargarlo, se crean tensores nuevos con esas formas y se copian en la capa; las
salidas deben coincidir exactamente con las de antes de guardar.

## 5. Implementación

- [`src/io/checkpoint.ts`](../../src/io/checkpoint.ts): `saveCheckpoint(path, parámetros, metadatos)`
  y `loadCheckpoint(path, { requiresGrad })`. Escribe a temporal y renombra; valida al leer.
- [`test/io/checkpoint.test.ts`](../../test/io/checkpoint.test.ts): ida y vuelta, `requiresGrad`,
  formato/versión/forma/datos inválidos, duplicados y valores no finitos.
- [`lab/checkpoint/run.ts`](../../lab/checkpoint/run.ts): entrena, guarda, carga y compara salidas.

## 6. Verificación

- Los parámetros vuelven con la misma forma, los mismos valores y los mismos metadatos.
- Con `requiresGrad: true`, los tensores cargados piden gradiente.
- Un archivo con formato o versión incorrectos, una forma que no cuadra, datos `null`/no finitos o un
  nombre duplicado fallan.
- Guardar un tensor con `NaN` o `Infinity` falla (JSON los convertiría en `null`).
- Las salidas del modelo cargado coinciden con las del original.

## 7. Límites y errores comunes

- **Sin estado del optimizador**: SGD no tiene estado; con momentum/Adam hará falta guardarlo.
- **Sin clase `Model`**: los parámetros se nombran a mano (`hidden.weights`); un `Module` con
  `parameters()` y nombre llegará cuando haga falta.
- **JSON no es compacto** para modelos grandes; binario más adelante.
- **No cifra ni firma**: no es para secretos ni para ejecutar código de terceros.
- **`f32`**: los valores se guardan como números y vuelven a `Float32Array`.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- PyTorch, `state_dict` y `torch.save` (checkpoints con parámetros y estado del optimizador).
- Safetensors (formato binario pensado para tensores).
