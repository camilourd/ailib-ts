# Inventario de conceptos del C#

Fuente: repo `ailib-cshrap` (C#, .NET Framework 4.5.2), commit `ccead45`, remoto
`https://github.com/camilourd/ailib-cshrap.git`. El repo se quitará del workspace; las rutas de abajo
son relativas a `ailib-cshrap/ailib/ailib/` y quedan como referencia histórica.

Este documento es el **pozo de conceptos**: lo que existe en C#, qué hacemos con ello y a qué fase
pertenece. Solo cuando un concepto se adopta pasa al índice (`teoria/README.md`) y empieza su nota.

## Cómo se lee

- **Fase**: F0–F8 de la hoja `ailib-ts` en la capa.
- **Destino**: archivo o carpeta de `ailib-ts` donde vivirá.
- **Decisión**: `portar` (reimplementar entendiendo), `rehacer` (la idea sirve, la implementación no),
  `descartar` o `más tarde`.
- **Estado**: `por revisar` (no he leído el archivo a fondo) o `decidido`.

## ann/

| Concepto | Archivo C# | Qué es | Fase | Destino | Decisión | Estado |
| --- | --- | --- | --- | --- | --- | --- |
| ANN | `ann/ANN.cs` | Contrato `propagate(input)` de una red | F2 | `src/models/` | portar | decidido |
| Layer | `ann/Layer.cs` | Capa con neuronas: `propagate`, `weighted_input`, `activate` | F2 | `src/nn/layer.ts` | rehacer | decidido |
| Perceptron | `ann/Perceptron.cs` | Neurona: pesos, sesgo `beta` y activación | F2 | `src/nn/neuron.ts` | rehacer | decidido |
| Perceptron1D/2D/3D | `ann/perceptron/*.cs` | Tres variantes según la dimensión del input | F2 | `src/nn/` | rehacer (un tensor, no tres clases) | decidido |
| Layer1D | `ann/layer/Layer1D.cs` | Capa densa | F2 | `src/nn/linear.ts` | portar | decidido |
| SoftmaxLayer | `ann/layer/SoftmaxLayer.cs` | Softmax por exponencial y normalización | F2 | `src/nn/softmax.ts` | rehacer (no resta el máximo: inestable) | decidido |
| ReceptiveField | `ann/convolutional/ReceptiveField.cs` | Pila de capas convolucionales (canales) | F5 | `src/models/cnn.ts` | rehacer | por revisar |
| ConvolutionalLayer | `ann/convolutional/ConvolutionalLayer.cs` | Rejilla de perceptrones por ventana | F5 | `src/nn/conv.ts` | rehacer (sin pesos compartidos) | decidido |
| PoolingField | `ann/convolutional/PoolingField.cs` | "Pooling" construido con `ConvolutionalLayer` | F5 | `src/nn/pooling.ts` | rehacer (no hay max/avg pooling) | decidido |
| IntegrationLayer | `ann/convolutional/IntegrationLayer.cs` | Aplana la salida a un vector | F5 | `src/nn/flatten.ts` | portar | por revisar |
| ConvolutionalNetwork | `ann/convolutional/ConvolutionalNetwork.cs` | Receptivo → pooling → integración → salida | F5 | `src/models/cnn.ts` | rehacer (20 unidades fijas) | decidido |
| ConvolutionalNetworkClassifier | `ann/convolutional/ConvolutionalNetworkClassifier.cs` | Variante clasificador | F5 | `src/models/cnn.ts` | rehacer | por revisar |
| RecurentUnit | `ann/recurrent/RecurentUnit.cs` | Base de celda recurrente | F4 | `src/models/rnn.ts` | rehacer | por revisar |
| GRU / GRUUnit | `ann/recurrent/GRU.cs`, `GRUUnit.cs` | Recurrencia con compuertas | F4 | `src/models/rnn.ts` | rehacer (una unidad por paso) | decidido |
| LSTM / LSTMCell | `ann/recurrent/LSTM.cs`, `LSTMCell.cs` | Recurrencia con memoria | F4 | `src/models/rnn.ts` | rehacer (una celda por paso) | decidido |
| Test | `Test.cs` | `Main` que lee una red serializada | F0/F2 | `test/` | descartar (lo reemplaza `node:test`) | decidido |

## math/

| Concepto | Archivo C# | Qué es | Fase | Destino | Decisión | Estado |
| --- | --- | --- | --- | --- | --- | --- |
| ActivationFunction | `math/ActivationFunction.cs` | Contrato `activate` / `derivate` | F2 | `src/nn/activation.ts` | portar | decidido |
| Activaciones | `math/activation_functions/*.cs` | BinaryStep, Exponential, Identity, LeakyReLU, ReLU, Sigmoid, Tanh | F2 | `src/nn/activations.ts` | portar (con test numérico) | decidido |
| Statistics | `math/statistic/Statistics.cs` | mínimo, máximo, media, varianza y desviación | F2 | `src/train/metrics.ts` | portar | por revisar |
| StatisticsWithMedian | `math/statistic/StatisticsWithMedian.cs` | Añade la mediana | F2 | `src/train/metrics.ts` | portar | por revisar |
| Rand | `math/integer/Rand.cs` | Generación entera por lotes | F3 | `src/evolution/selection.ts` | más tarde | por revisar |
| Roulette | `math/integer/Roulette.cs` | Selección proporcional por densidad | F3 | `src/evolution/selection.ts` | portar | decidido |
| Function / Predicate | `math/functions/`, `math/logic/` | Abstracciones funcionales | F2/F3 | `src/core/` | más tarde (en TS son funciones) | por revisar |
| Graph / Vertex / Edge | `math/graphs/` | Grafo genérico | F8 | `src/evolution/` | más tarde | por revisar |

## random/

| Concepto | Archivo C# | Qué es | Fase | Destino | Decisión | Estado |
| --- | --- | --- | --- | --- | --- | --- |
| Generator / RandomGenerator | `random/Generator.cs`, `random/RandomGenerator.cs` | Contrato `next()` | F1 | `src/core/rng.ts` | rehacer (con semilla) | decidido |
| StandarRandomGenerator | `random/StandarRandomGenerator.cs` | `new Random()` sin semilla | F1 | `src/core/rng.ts` | rehacer (no reproducible) | decidido |
| Shuffle | `random/Shuffle.cs` | Mezcla por intercambio aleatorio | F1 | `src/core/rng.ts` | rehacer (sesgada; falta Fisher-Yates) | decidido |
| InverseGenerator | `random/InverseGenerator.cs` | Generación por inversa de la CDF | F3 | `src/core/rng.ts` | portar | por revisar |
| StandardPowerLaw | `random/real/StandardPowerLaw.cs` | Distribución power-law | F8 | `src/evolution/rng.ts` | más tarde | por revisar |

## Soporte (lo que usan estas áreas)

| Concepto | Archivo C# | Qué es | Fase | Destino | Decisión | Estado |
| --- | --- | --- | --- | --- | --- | --- |
| ArrayTools / Dimension / Location | `utils/*.cs` | Utilidades de arrays, tamaños y coordenadas | F1/F5 | `src/core/` | portar según necesidad | por revisar |
| Scanner | `io/Scanner.cs` | Lectura de redes serializadas | — | — | más tarde | por revisar |
| Tagged / TaggedManager | `objects/*.cs` | Individuo con etiquetas (evolución) | F3 | `src/evolution/` | portar | por revisar |
| Sort / DisjointSetUnion | `sort/*.cs`, `structures/DisjointSetUnion.cs` | Orden y conjuntos disjuntos | F3/F8 | `src/evolution/` | más tarde | por revisar |

## Lectura de conjunto

- **No hay entrenamiento por gradientes.** Solo existen las derivadas de las activaciones
  (`math/ActivationFunction.cs:6`); `update`/`upload` serializan pesos
  (`ann/perceptron/Perceptron1D.cs:68-87`, `ann/convolutional/ConvolutionalNetwork.cs:65-94`). El
  entrenamiento venía del track evolutivo (`evolution/`, `search/`). Por eso F1–F2 añaden *backprop* y
  F3 recupera la evolución como laboratorio.
- **Las convolucionales no comparten pesos.** `ConvolutionalLayer` crea un `Perceptron2D` por
  posición (`ann/convolutional/ConvolutionalLayer.cs:23-33`): es una capa localmente conectada, no
  una convolución. Y `PoolingField` reutiliza esa misma capa (`PoolingField.cs:13`), así que el
  «pooling» también es aprendido. Se rehace en F5.
- **Los recurrentes no comparten pesos entre pasos.** `GRU` y `LSTM` crean una celda por paso
  (`ann/recurrent/GRU.cs:15-21`, `LSTM.cs:15-21`). Se rehacen en F4 sobre una sola celda reutilizada.
- **La aleatoriedad no es reproducible.** `new Random()` sin semilla
  (`random/StandarRandomGenerator.cs:7`, `ann/perceptron/Perceptron1D.cs:24`) y `Shuffle` no es
  Fisher-Yates (`random/Shuffle.cs:12-19`). F1 lo corrige.
- **No hay pruebas.** `Test.cs` es un `Main` que lee una red serializada (`Test.cs:12-16`).

## Huecos (lo que el C# no tiene y el camino al LLM sí necesita)

- Tensor n-D y operaciones vectorizadas.
- Autograd / retropropagación.
- Optimizadores (SGD, momentum, Adam) y schedules.
- Pérdidas numéricamente estables e inicialización (Xavier/He).
- Tokenizer y embeddings.
- Atención y transformer.
- Bucle de entrenamiento, batching, checkpoints y perplejidad.
- Pipeline de datos, evaluación y sampling.
- KV cache y cuantización (F7).

Estos huecos son la razón de ser de F1, F2, F4, F5, F6 y F7.
