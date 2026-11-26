# HAEA en el LM

## 1. Qué problema resuelve

En F3, HAEA resolvió XOR con 22 parámetros. ¿Puede entrenar los pesos de un LM de caracteres, donde
hay miles? Este experimento compara HAEA con SGD en RNN, GRU y LSTM.

## 2. Protocolo

- LM pequeño: embedding 8, oculto 16, vocabulario byte (256).
- Parámetros por celda: **6.800** (RNN), **7.600** (GRU) y **8.000** (LSTM).
- Train: 636 bytes; test: 64 bytes.
- **HAEA**: población 20 × 50 generaciones = 1000 evaluaciones; cada una promedia 4 ventanas de 32.
- **SGD**: 4000 pasos de una ventana de 32; mismas ~128.000 predicciones de byte.
- Métrica comparable: **BPC** en test; pérdida por token no es comparable entre vocabularios.

## 3. Resultado

| Celda | HAEA pérdida / BPC | SGD pérdida / BPC |
| --- | --- | --- |
| RNN | 5.0736 / 7.205 | 2.8567 / 4.057 |
| GRU | 5.0559 / 7.180 | 2.7032 / 3.839 |
| LSTM | 5.1400 / 7.300 | 2.7389 / 3.890 |

HAEA se queda cerca del azar (un byte uniforme son ~8 bits); SGD baja a 3.8–4.1 BPC. Las muestras de
HAEA son bytes sueltos; las de SGD, fragmentos con palabras.

### Híbrido: 20 pasos de SGD como operador

Se añadió un operador `paso-sgd`: cuando HAEA lo elige, da **un** paso de SGD sobre el genoma del
individuo. El presupuesto total de gradiente es **20 pasos**, no 1000; el resto sigue siendo
mutación/cruce.

| Celda | HAEA | HAEA + 20 pasos SGD | SGD |
| --- | --- | --- | --- |
| RNN | 7.205 | 7.250 | 4.057 |
| GRU | 7.180 | 7.211 | 3.839 |
| LSTM | 7.300 | 7.293 | 3.890 |

**No mueve la aguja.** Veinte pasos repartidos entre 1000 evaluaciones son demasiado pocos: el
operador se elige pocas veces, a menudo sobre individuos malos, y un solo paso sobre una ventana no
basta para salir de la meseta. Un memético serio aplicaría búsqueda local al mejor/élite con muchos
más pasos; pero entonces se parece cada vez más a SGD.

## 4. Por qué

- **Dimensión**: 6.800–8.000 pesos. Con 1000 evaluaciones, HAEA no puede explorar ni de lejos.
- **Información local**: SGD usa el gradiente para decidir la dirección de cada peso; HAEA solo ve el
  fitness final de un genoma completo.
- **Coste por evaluación**: cada individuo de HAEA cuesta un forward completo; el gradiente reutiliza
  ese forward para calcular todos los gradientes.
- **XOR no era representativo**: 22 parámetros y una tarea simple; a escala de LM, el gradiente
  domina. La evolución sigue teniendo sentido en dimensiones bajas, hiperparámetros o problemas sin
  gradiente.

## 5. Implementación

- [`lab/lm/haea.ts`](../../lab/lm/haea.ts): aplana los parámetros del `CharLM`, corre HAEA y SGD, y
  evalúa BPC y muestras.
- [`lab/lm/haea-gradient.ts`](../../lab/lm/haea-gradient.ts): el híbrido con 20 pasos de SGD como
  operador.
- [`src/evolution/`](../../src/evolution/): el HAEA.
- [`src/models/char-lm.ts`](../../src/models/char-lm.ts): el LM con `cell` configurable.

## 6. Verificación

- Ambos métodos usan ~128.000 predicciones de byte.
- La tabla sale de la misma semilla, texto, ventanas y modelo.
- El híbrido gasta exactamente 20 pasos de SGD como operador.
- 134/134 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Una semilla y un modelo pequeño**: no es una ley universal.
- **Mismo presupuesto, no mismo tiempo**: HAEA podría mejorar con más generaciones, pero a un coste de
  muestras enorme.
- **Sin ajuste de HAEA**: población, sigma y operadores son fijos.
- **BPC sobre 64 bytes de test**: el número es ruidoso; la diferencia es demasiado grande para ser
  solo ruido.
- **20 pasos de SGD no son “SGD como operador”**: son un empujón mínimo; el resultado no descarta un
  memético con búsqueda local más fuerte.
- **No prueba que la evolución sea inútil**: prueba que no compite en 8.000 dimensiones con 1000
  evaluaciones.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Salimans et al. (2017), *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*.
- Gómez, J. (2009), *A Hybrid Adaptive Evolutionary Algorithm*.
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 8.
