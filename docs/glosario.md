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
| genoma | Vector de números que codifica una solución en evolución. | [HAEA](teoria/evolucion/haea.md) |
| fitness | Valor que mide qué tan buena es una solución; aquí se minimiza. | [HAEA](teoria/evolucion/haea.md) |
| población | Conjunto de individuos que evolucionan juntos. | [HAEA](teoria/evolucion/haea.md) |
| torneo | Selección que toma k individuos al azar y gana el de mejor fitness. | [HAEA](teoria/evolucion/haea.md) |
| mutación | Variación de un solo padre que altera el genoma. | [HAEA](teoria/evolucion/haea.md) |
| cruce | Variación de dos padres que mezcla sus genes. | [HAEA](teoria/evolucion/haea.md) |
| elitismo | Conservar los mejores individuos entre generaciones. | [HAEA](teoria/evolucion/haea.md) |
| evaluación | Una llamada al fitness; es la unidad de coste de la evolución. | [evolución vs gradiente](teoria/evolucion/evolucion-vs-gradiente.md) |
| token | Unidad que recibe el modelo; aquí un byte. | [tokenizer](teoria/nucleo/tokenizer.md) |
| vocabulario | Conjunto de tokens posibles. | [tokenizer](teoria/nucleo/tokenizer.md) |
| byte | Valor 0..255; unidad de UTF-8. | [tokenizer](teoria/nucleo/tokenizer.md) |
| UTF-8 | Codificación de Unicode en 1–4 bytes; un carácter puede ocupar varios tokens. | [tokenizer](teoria/nucleo/tokenizer.md) |
| BPE | Byte Pair Encoding: tokenizer que fusiona pares frecuentes para acortar secuencias. | [tokenizer](teoria/nucleo/tokenizer.md) |
| embedding | Vector aprendido para cada token en una tabla. | [embeddings](teoria/redes/embeddings.md) |
| RNN | Red recurrente: actualiza un estado oculto en cada paso de la secuencia. | [RNN](teoria/redes/rnn.md) |
| estado oculto | Vector que resume el pasado en una RNN. | [RNN](teoria/redes/rnn.md) |
| BPTT | Retropropagación a través del tiempo: backward por toda la secuencia. | [RNN](teoria/redes/rnn.md) |
| tanh | Activación que lleva los reales a (−1, 1); su derivada es 1 − tanh². | [RNN](teoria/redes/rnn.md) |
| LM | Modelo de lenguaje: aprende la probabilidad del siguiente token dado el contexto. | [LM de caracteres](teoria/secuencias/lm-caracteres.md) |
| perplejidad | e^L; cuántas opciones efectivas maneja el LM. | [LM de caracteres](teoria/secuencias/lm-caracteres.md) |
| temperatura | Divisor de los logits antes del softmax; baja = conservador, alta = aleatorio. | [LM de caracteres](teoria/secuencias/lm-caracteres.md) |
| muestreo | Elegir un token según la distribución de probabilidad. | [LM de caracteres](teoria/secuencias/lm-caracteres.md) |
| desvanecimiento | El gradiente se encoge al retroceder en el tiempo y los pasos lejanos dejan de aprender. | [desvanecimiento](teoria/secuencias/desvanecimiento.md) |
| jacobiano | Matriz de derivadas de una transformación; en BPTT se multiplica paso a paso. | [desvanecimiento](teoria/secuencias/desvanecimiento.md) |
| norma | Longitud de un vector; aquí, la del gradiente de un estado. | [desvanecimiento](teoria/secuencias/desvanecimiento.md) |
| GRU | Celda recurrente con compuertas de actualización y reinicio. | [GRU y LSTM](teoria/secuencias/gru-lstm.md) |
| LSTM | Celda recurrente con compuertas de entrada, olvido y salida, y estado de celda. | [GRU y LSTM](teoria/secuencias/gru-lstm.md) |
| compuerta | Valor en (0,1) que regula cuánto pasa de una señal. | [GRU y LSTM](teoria/secuencias/gru-lstm.md) |
| celda | Estado interno adicional del LSTM, actualizado de forma aditiva. | [GRU y LSTM](teoria/secuencias/gru-lstm.md) |
| merge | Par de tokens que BPE fusiona en uno nuevo. | [BPE](teoria/nucleo/bpe.md) |
| BPC | Bits por carácter; métrica comparable entre tokenizers. | [BPE](teoria/nucleo/bpe.md) |
| atención | Mezcla ponderada de valores según la afinidad entre consultas y claves. | [atención](teoria/secuencias/atencion.md) |
| QKV | Consultas, claves y valores: las tres proyecciones de la atención. | [atención](teoria/secuencias/atencion.md) |
| máscara causal | Impide que una posición mire al futuro. | [atención](teoria/secuencias/atencion.md) |
| cabeza | Subespacio de atención con su propia proyección Q/K/V. | [multi-head](teoria/secuencias/multi-head.md) |
| multi-head | Varias cabezas de atención en paralelo, concatenadas y proyectadas. | [multi-head](teoria/secuencias/multi-head.md) |
| LayerNorm | Normaliza cada fila y aprende gamma/beta. | [bloque transformer](teoria/secuencias/bloque-transformer.md) |
| FFN | Red densa por posición: Linear → ReLU → Linear. | [bloque transformer](teoria/secuencias/bloque-transformer.md) |
| residual | Sumar la entrada a la salida de un sub-bloque. | [bloque transformer](teoria/secuencias/bloque-transformer.md) |
| pre-norm | Normalizar antes del sub-bloque. | [bloque transformer](teoria/secuencias/bloque-transformer.md) |
| RoPE | Rotación de Q/K por posición; el producto depende de la distancia relativa. | [RoPE](teoria/secuencias/rope.md) |
| posición relativa | Distancia entre dos posiciones, no su índice absoluto. | [RoPE](teoria/secuencias/rope.md) |
| transformer | Modelo que apila bloques de atención, norm y FFN; aquí con RoPE y máscara causal. | [LM transformer](teoria/secuencias/lm-transformer.md) |
| ablación | Quitar una pieza del modelo y comparar con el resto igual. | [ablación](teoria/secuencias/ablacion-transformer.md) |
| perplejidad | Exponencial de la cross-entropy media; tokens efectivos entre los que duda el modelo. | [preentreno](teoria/entrenamiento/preentreno.md) |
| warmup | Subir la tasa de aprendizaje al principio del entrenamiento. | [preentreno](teoria/entrenamiento/preentreno.md) |
| gradient clipping | Escalar los gradientes si su norma supera un límite. | [preentreno](teoria/entrenamiento/preentreno.md) |
| SFT | Ajuste supervisado con pares instrucción→respuesta. | [SFT](teoria/entrenamiento/sft.md) |
| instrucción | Texto de entrada que el modelo debe responder. | [SFT](teoria/entrenamiento/sft.md) |
| tokens/s | Tokens generados por segundo; mide velocidad de inferencia. | [inferencia](teoria/entrenamiento/inferencia.md) |
| KV cache | Guardar K y V ya calculados para no repetir la atención del pasado. | [KV cache](teoria/entrenamiento/kv-cache.md) |
| temperatura | Divide los logits antes del softmax; más alta, más diversidad. | [sampling](teoria/entrenamiento/sampling.md) |
| top-k | Se queda con los k tokens más probables. | [sampling](teoria/entrenamiento/sampling.md) |
| top-p | Se queda con el núcleo de probabilidad acumulada p. | [sampling](teoria/entrenamiento/sampling.md) |
| cuantización | Guardar pesos con menos bits; INT8 usa 1 byte por peso. | [cuantización](teoria/entrenamiento/cuantizacion.md) |
| escala | Factor que convierte el entero cuantizado de vuelta a punto flotante. | [cuantización](teoria/entrenamiento/cuantizacion.md) |
| cuantización selectiva | Cuantizar solo algunos módulos y comparar perplejidad y memoria. | [F8](teoria/entrenamiento/f8-cuantizacion-selectiva.md) |
| union-find | Estructura para mantener conjuntos disjuntos y unirlos. | [DSU](teoria/nucleo/disjoint-set-union.md) |
| DSU | *Disjoint Set Union*; ver union-find. | [DSU](teoria/nucleo/disjoint-set-union.md) |

El glosario no sustituye a la nota: solo fija el vocabulario.
