# Glosario

Término · definición de una línea · nota canónica. Se añade el término la primera vez que aparece en
una nota o en el código.

| Término | Definición | Nota |
| --- | --- | --- |
| sigmoid | Función logística $\sigma(x)=1/(1+e^{-x})$; lleva los reales a (0,1) y su derivada es $\sigma(x)(1-\sigma(x))$. | [sigmoid](teoria/redes/sigmoid.md) |
| derivada | Tasa de cambio instantánea de una función; aquí, la pendiente de $\sigma$ en $x$. | [sigmoid](teoria/redes/sigmoid.md) |
| saturación | Zona donde la derivada se acerca a 0 y el gradiente deja de fluir. | [sigmoid](teoria/redes/sigmoid.md) |
| PRNG | Generador pseudoaleatorio: algoritmo determinista que, desde una semilla, produce una secuencia que parece aleatoria. | [rng](teoria/nucleo/rng.md) |
| semilla | Valor inicial del estado de un PRNG; fija toda la secuencia. | [rng](teoria/nucleo/rng.md) |
| uniforme | Distribución en la que cada valor del rango tiene la misma probabilidad. | [rng](teoria/nucleo/rng.md) |
| tensor | Arreglo n-dimensional de números: datos en un buffer plano más una forma. | [tensor](teoria/nucleo/tensor.md) |
| forma | Tamaño de cada dimensión de un tensor. | [tensor](teoria/nucleo/tensor.md) |
| rango | Número de dimensiones de un tensor. | [tensor](teoria/nucleo/tensor.md) |
| stride | Cuántos elementos saltar en memoria para avanzar uno en una dimensión. | [tensor](teoria/nucleo/tensor.md) |
| row-major | Layout en el que el último índice varía más rápido. | [tensor](teoria/nucleo/tensor.md) |
| broadcasting | Regla para combinar formas distintas alineando por la derecha; las dimensiones de tamaño 1 se expanden con stride 0. | [broadcasting](teoria/nucleo/broadcasting.md) |
| eje | Una dimensión de un tensor. | [tensor](teoria/nucleo/tensor.md) |
| matmul | Producto matricial: cada salida es el producto punto de una fila por una columna. | [matmul](teoria/nucleo/matmul.md) |
| reducción | Operación que colapsa uno o más ejes sumando o promediando. | [reducciones](teoria/nucleo/reductions.md) |
| gradiente | Tensor de derivadas parciales de una salida respecto de cada elemento de una entrada. | [autograd](teoria/nucleo/autograd.md) |
| retropropagación | Cálculo de gradientes recorriendo el grafo computacional en orden inverso. | [autograd](teoria/nucleo/autograd.md) |
| grafo computacional | Nodos (tensores) y operaciones que los conectan; el autograd lo recorre al revés. | [autograd](teoria/nucleo/autograd.md) |
| MLP | Perceptrón multicapa: red *feedforward* de capas densas con activaciones no lineales entre ellas. | [autograd](teoria/nucleo/autograd.md) |
| SGD | Descenso de gradiente estocástico: actualiza θ ← θ − η·∇L, con el gradiente estimado en un lote. | [SGD](teoria/entrenamiento/sgd.md) |
| tasa de aprendizaje | η: cuánto se mueve cada parámetro en cada paso. | [SGD](teoria/entrenamiento/sgd.md) |
| capa lineal | Capa densa y = xW + b; cada salida es una combinación lineal de la entrada más un sesgo. | [capa lineal](teoria/redes/capa-lineal.md) |
| ReLU | Activación max(0,x); derivada 1 si x > 0 y 0 si no. | [ReLU](teoria/redes/relu.md) |
| Xavier | Inicialización uniforme con límite sqrt(6/(entrada+salida)), pensada para mantener la escala del gradiente. | [capa lineal](teoria/redes/capa-lineal.md) |
| MSE | Error cuadrático medio: media de (predicción − objetivo)². | [pérdidas](teoria/entrenamiento/perdidas.md) |
| entropía cruzada | Pérdida de clasificación: −ln de la probabilidad que el modelo da a la clase correcta. | [pérdidas](teoria/entrenamiento/perdidas.md) |
| softmax | Convierte logits en probabilidades: exponencial normalizada. | [pérdidas](teoria/entrenamiento/perdidas.md) |
| logits | Salidas sin normalizar de la última capa. | [pérdidas](teoria/entrenamiento/perdidas.md) |
| paso | Una actualización de parámetros: forward, pérdida, backward y step. | [bucle](teoria/entrenamiento/bucle.md) |
| época | Una pasada completa por el dataset. | [bucle](teoria/entrenamiento/bucle.md) |
| lote | Subconjunto de datos que se procesa en un paso. | [bucle](teoria/entrenamiento/bucle.md) |
| checkpoint | Archivo con los parámetros de un modelo y los metadatos para reconstruirlo. | [checkpoints](teoria/entrenamiento/checkpoints.md) |
| set | Datos, split, pérdida, métrica y baseline de una tarea comparable. | [sets](teoria/entrenamiento/sets.md) |
| benchmark | Entrenar y evaluar varios modelos sobre el mismo set. | [sets](teoria/entrenamiento/sets.md) |
| MAE | Error absoluto medio: media del valor absoluto del error. | [sets](teoria/entrenamiento/sets.md) |
| RMSE | Raíz del error cuadrático medio. | [sets](teoria/entrenamiento/sets.md) |
| acierto direccional | Proporción de signos de la predicción que coinciden con el objetivo. | [sets](teoria/entrenamiento/sets.md) |
| matriz de confusión | Tabla de aciertos y errores por clase: TP, FP, TN y FN. | [dirección](teoria/entrenamiento/direccion.md) |
| precisión | De lo que el modelo marcó positivo, cuánto lo era. | [dirección](teoria/entrenamiento/direccion.md) |
| recall | De lo positivo real, cuánto detectó el modelo. | [dirección](teoria/entrenamiento/direccion.md) |
| especificidad | De lo negativo real, cuánto detectó el modelo. | [dirección](teoria/entrenamiento/direccion.md) |
| F1 | Media armónica de precisión y recall. | [dirección](teoria/entrenamiento/direccion.md) |
| MCC | Correlación entre predicción y realidad para binario; robusta al desbalance. | [dirección](teoria/entrenamiento/direccion.md) |
| balanced accuracy | Media de recall y especificidad. | [dirección](teoria/entrenamiento/direccion.md) |
| R² | Fracción de la varianza del objetivo explicada por el modelo. | [sets](teoria/entrenamiento/sets.md) |
| retorno | Cambio relativo del precio; aquí el del día siguiente en porcentaje. | [pronóstico](teoria/entrenamiento/pronostico.md) |
| baseline | Predicción de referencia (aquí, 0) contra la que se compara un modelo. | [pronóstico](teoria/entrenamiento/pronostico.md) |

El glosario no sustituye a la nota: solo fija el vocabulario.
