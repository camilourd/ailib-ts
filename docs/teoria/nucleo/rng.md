# rng

## 1. Qué problema resuelve

Casi todo en una librería de aprendizaje automático necesita azar: inicializar los pesos, barajar
los datos en cada época, muestrear texto, aplicar dropout. Y necesita **reproducibilidad**: con la
misma semilla, la misma secuencia, para poder comparar experimentos y depurar un resultado.

`Math.random()` no acepta semilla ni se puede reproducir. Un **PRNG** (generador pseudoaleatorio) es
un algoritmo determinista que, partiendo de un estado inicial (la semilla), produce una secuencia
que se comporta como aleatoria. Reproducible no significa predecible para quien no conoce el estado.

## 2. Definición y notación

Un PRNG tiene un estado interno $s$. En cada paso:

$$s \leftarrow f(s), \qquad \text{salida} = g(s).$$

La **semilla** es el valor inicial de $s$. La salida es **uniforme** si cada valor de $[0,1)$ tiene
la misma probabilidad.

La interfaz que usaremos:

- `next(): number` — uniforme en $[0,1)$.
- `int(maxExclusive): number` — entero uniforme en $[0, n)$.
- `shuffle(values): T[]` — copia mezclada, sin mutar la entrada.

La implementación es **mulberry32**: estado de 32 bits, periodo $2^{32}$. Determinista, rápido y
**no criptográfico**.

## 3. Cómo funciona (el porqué de cada operación)

No hay una derivación como la de sigmoid: es un diseño. Pero cada pieza tiene una razón:

1. **Avanzar el estado.** $s \leftarrow (s + \texttt{0x6D2B79F5}) \bmod 2^{32}$. La constante es
   impar, así que el estado recorre los $2^{32}$ valores antes de repetirse (periodo completo).
2. **Mezclar.** $t \leftarrow \mathrm{imul}(s \oplus (s \gg 15),\, s \mid 1)$. El XOR con el
   desplazamiento lleva bits altos a posiciones bajas; la multiplicación de 32 bits
   (`Math.imul`) dispersa la información. El `| 1` fuerza un multiplicador impar.
3. **Segunda ronda.** $t \leftarrow t \oplus \big(t + \mathrm{imul}(t \oplus (t \gg 7),\, t \mid 61)\big)$.
4. **Salida.** $(t \oplus (t \gg 14)) \bmod 2^{32}$, dividida entre $2^{32}$ para quedar en $[0,1)$.

Un generador lineal puro ($s \leftarrow as+c$) tiene bits bajos muy pobres; mezclar con XOR y
multiplicación reparte la entropía por todo el estado.

### Otros generadores de la familia

| Generador | Estado | Periodo | Fuerza | Uso típico |
| --- | --- | --- | --- | --- |
| LCG (Numerical Recipes) | 32 bits | $2^{32}$ | baja: bits bajos pobres | didáctico |
| xorshift32 / xorshift64 | 32 / 64 bits | $2^{32}-1$ / $2^{64}-1$ | media | rápido, contraste |
| xoshiro256** | 256 bits | $2^{256}-1$ | alta | simulación seria |
| PCG32 / PCG64 | 64 / 128 bits | $2^{64}$ / $2^{128}$ | alta, con *streams* | buen valor por defecto |
| SplitMix64 | 64 bits | $2^{64}$ | alta | sembrar otros generadores |
| Mersenne Twister (MT19937) | ~2,5 KB | $2^{19937}-1$ | buena, lenta | legado (Python, NumPy) |
| ChaCha20 / `crypto` | variable | — | criptográfica | secretos y seguridad |

mulberry32 es el punto de partida por su tamaño; la interfaz `RandomNumberGenerator` permite
sustituirlo por otro algoritmo sin tocar el código que lo usa. Los cinco primeros son **no
criptográficos**: reproducen una secuencia, pero no resisten un ataque.

## 4. Ejemplo a mano

Semilla $1$, primer paso:

| Paso | Valor |
| --- | --- |
| estado | $1 + \texttt{0x6D2B79F5} = \texttt{0x6D2B79F6}$ |
| $t$ tras la primera mezcla | $\texttt{0x91517F60}$ |
| $t$ tras la segunda mezcla | $\texttt{0xA08568E6}$ |
| salida | $\texttt{0xA087EAF3}$ |
| valor | $\texttt{0xA087EAF3} / 2^{32} = 0.6270739405881613$ |

Los cinco primeros valores con semilla $1$ son:

```
0.62707394058816135  0.0027357211802154779  0.52744703995995224
0.98105096747167408  0.96837789821438491
```

Con semilla $2$ la secuencia es distinta desde el primer valor. Eso es lo que permite tener
experimentos reproducibles sin que dos ejecuciones se parezcan.

## 5. Implementación

- [`src/core/random/rng.ts`](../../src/core/random/rng.ts): el contrato `RandomNumberGenerator`
  (`next`, `int`, `shuffle`) y la base `Uint32RandomGenerator` (Template Method). La base implementa
  `next`, `int` (con **rechazo** para no sesgar cuando $n$ no divide a $2^{32}$) y `shuffle`
  (Fisher-Yates) sobre un único `nextUint32()` que aporta cada algoritmo.
- [`src/core/random/mulberry32.ts`](../../src/core/random/mulberry32.ts): `Mulberry32`, el
  algoritmo.
- [`src/core/random/index.ts`](../../src/core/random/index.ts): `createRng(seed)`, el generador por
  defecto, y los re-exporta.
- Un generador nuevo (xorshift32, PCG…) solo tiene que extender la base y aportar su paso crudo; su
  teoría irá en su propia nota.
- [`test/core/random/rng.test.ts`](../../test/core/random/rng.test.ts): la base se prueba con un
  generador de secuencia fija; el algoritmo, con reproducibilidad, rango, uniformidad, permutación y
  un vector de regresión.
- [`lab/rng/run.ts`](../../lab/rng/run.ts): histograma de 100 000 tiradas y comprobación de
  reproducibilidad.

## 6. Verificación

- La misma semilla produce la misma secuencia (100 valores).
- Semillas distintas producen secuencias distintas.
- `next()` cae en $[0,1)$ y `int(n)` en $[0,n)$.
- Reparto por cubos: 60 000 tiradas en 6 cubos, cada uno a menos de 500 del esperado (10 000).
- `shuffle` es una permutación y no muta la entrada.
- Vector de regresión con semilla 1: fija el algoritmo. Si alguien cambia la constante o el orden
  de las mezclas, el test falla **a propósito**: cambiar el PRNG cambia todas las secuencias.

## 7. Límites y errores comunes

- **No es criptográfico.** Conociendo el estado se predice la salida; para seguridad hay que usar
  `crypto`.
- **Periodo $2^{32}$.** Suficiente para entrenar y para experimentos pequeños; no para simular
  billones de eventos.
- **Estado de 32 bits.** La calidad es buena, pero PCG o xoshiro son mejores; se pueden cambiar sin
  tocar la interfaz.
- **Solo $2^{32}$ valores posibles** en `next()`: no genera todos los números de coma flotante.
- **Sembrar con el reloj** rompe la reproducibilidad; la semilla la elige quien llama, no la
  librería.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Tommy Ettinger, *mulberry32* (2017), el algoritmo que se implementa aquí.
- Melissa O'Neill, *PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for
  Random Number Generation* (2014).
- Knuth, *The Art of Computer Programming*, vol. 2, sección 3.4.2 (Fisher-Yates y generadores
  lineales).
