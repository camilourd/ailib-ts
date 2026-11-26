# HAEA

## 1. Qué problema resuelve

El gradiente necesita una función derivable. HAEA (*Hybrid Adaptive Evolutionary Algorithm*) optimiza
una **caja negra**: solo se evalúa el fitness de un genoma, sin derivadas. Sirve para problemas no
diferenciables, con ruido o con restricciones.

## 2. Definición y notación

- **Genoma**: un vector de números reales (`Float32Array`).
- **Fitness**: $f(\text{genoma})$, cuanto menor mejor.
- **Población**: un conjunto de individuos `{ genoma, fitness }`.
- **Selección**: torneo; se eligen $k$ al azar y gana el de menor fitness.
- **Variación**: mutación gaussiana (1 padre) o cruce (2 padres).
- **HAEA**: cada individuo tiene **probabilidades por operador**; se elige uno por ruleta y, tras el
  descendiente, se **premia** si mejora o se **castiga** si no.

## 3. Cómo funciona

En cada generación, para cada individuo:

1. Se elige un operador según sus probabilidades.
2. Se completa la aridad del operador con padres elegidos por torneo.
3. Se genera y evalúa el descendiente.
4. Si mejora al individuo, lo reemplaza y el operador se premia:

$$r_i \leftarrow r_i(1+u), \qquad u \sim U(0,1).$$

5. Si no mejora, el operador se castiga:

$$r_i \leftarrow r_i(1-u).$$

6. Las probabilidades se normalizan para que sumen 1. El elitismo conserva los mejores.

La adaptación es local a cada individuo: un operador que funciona para uno no arrastra a los demás.

## 4. Ejemplo a mano

Esfera 2D, población de 30, 100 generaciones, mutación gaussiana ($\sigma = 0.5$), cruce uniforme y
cruce aritmético. El mejor fitness empieza en 0.0815 y llega a 0.0000 tras 3030 evaluaciones; el
historial nunca empeora porque el reemplazo es steady-state con elitismo.

## 5. Implementación

- [`src/evolution/types.ts`](../../src/evolution/types.ts): `Genome`, `Fitness`, `Individual`,
  `Variation`.
- [`src/evolution/operators.ts`](../../src/evolution/operators.ts): mutación gaussiana, cruce
  uniforme y cruce aritmético; normal estándar por Box-Muller.
- [`src/evolution/selection.ts`](../../src/evolution/selection.ts): torneo.
- [`src/evolution/haea.ts`](../../src/evolution/haea.ts): población, probabilidades por individuo,
  reward/punish, elitismo, historial y contador de evaluaciones.
- [`test/evolution/haea.test.ts`](../../test/evolution/haea.test.ts): esfera, determinismo,
  operadores, torneo, `onGeneration` y validaciones.
- [`lab/evolution/haea.ts`](../../lab/evolution/haea.ts): la curva de la esfera.

## 6. Verificación

- La esfera 2D baja por debajo de 0.1 y el historial no empeora.
- La misma semilla da el mismo genoma y el mismo historial.
- La mutación cambia el genoma; los cruces toman genes de los padres y respetan el rango.
- El torneo elige de la población y rechaza tamaños inválidos.
- 6 pruebas nuevas (107/107) y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Versión simplificada**: sin CAHAEA, sin extinción, sin individuos etiquetados ni descriptores.
  Esas variantes se migrarán si el experimento las necesita.
- **Solo vectores reales**: sin genomas discretos, árboles ni permutaciones.
- **Sin restricciones ni nichos**: no hay penalizaciones, factibilidad ni especiación.
- **Sin garantía de convergencia**: la población puede estancarse; el elitismo evita perder lo bueno,
  no garantiza llegar al óptimo.
- **Caro en evaluaciones**: cada descendiente es una llamada al fitness; en redes grandes se nota.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Gómez, J. (2009), *A Hybrid Adaptive Evolutionary Algorithm* (HAEA).
- Eiben y Smith, *Introduction to Evolutionary Computing*.
- Referencia C#: `ailib-cshrap/ailib/ailib/evolution/haea/` (`HaeaOperators`, `HaeaVariation`,
  `HaeaReplacement`).
