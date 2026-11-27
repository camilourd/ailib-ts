# Disjoint Set Union

## 1. Qué problema resuelve

Mantener **conjuntos disjuntos** mientras se unen: saber si dos elementos están conectados, agrupar
componentes y construir clusters. Es la estructura clásica de Kruskal, componentes conexas y
agrupamiento gravitacional.

## 2. Definición y notación

Cada elemento tiene un **padre**; la raíz del árbol representa su conjunto.

- `find(v)`: devuelve la raíz de `v`.
- `join(a, b)`: une los conjuntos de `a` y `b`.
- `restart(n)`: vuelve a `n` elementos aislados.

Sin optimizaciones, un árbol puede degenerar en una lista y `find` cuesta $O(n)$.

## 3. Por qué funciona

- **Compresión de caminos**: al buscar, los nodos del camino apuntan directo a la raíz.
- **Unión por tamaño**: el árbol más pequeño se cuelga del más grande.
- Con las dos, una secuencia de operaciones cuesta casi $O(1)$ amortizado (inversa de Ackermann).

## 4. Ejemplo

```
0-1, 1-2, 3-4, 5-6, 4-2
```

queda:

```
0-1-2-3-4 · 5-6 · 7
```

El orden de los `join` no cambia las componentes; sí puede cambiar la forma interna del árbol.

## 5. Implementación

- [`src/core/disjoint-set-union.ts`](../../src/core/disjoint-set-union.ts): `DisjointSetUnion` con
  `find` iterativo, compresión de caminos y unión por tamaño.
- [`test/core/disjoint-set-union.test.ts`](../../test/core/disjoint-set-union.test.ts): unión,
  transitividad, reinicio y validaciones.
- [`lab/structures/disjoint-set-union.ts`](../../lab/structures/disjoint-set-union.ts): grupos.

Viene del `DisjointSetUnion` de `ailib-cshrap` (`structures/DisjointSetUnion.cs`): se conserva
`restart`, `find` y `join`; se añaden compresión de caminos y unión por tamaño, que no cambian las
componentes y mejoran el coste.

## 6. Verificación

- Cada elemento empieza en su propio conjunto.
- `join` es transitivo e idempotente.
- `restart` deja todo aislado.
- Índices y tamaños inválidos fallan.
- 178/178 pruebas, build y lint en verde.

## 7. Límites y errores comunes

- **No hay rollback**: las uniones no se deshacen.
- **No hay pesos**: solo importa la conectividad.
- **No hay persistencia**: se reconstruye desde cero.
- **Compresión de caminos no cambia el resultado**, solo la forma del árbol.
- **No confundir `find` con `join`**: `find` no modifica las componentes.

## 8. Mi explicación

Opcional: la persona puede escribirla; no es puerta de la fase.

## 9. Para profundizar (opcional)

- Tarjan (1975), *Efficiency of a Good But Not Linear Set Union Algorithm*.
- CLRS, capítulo de estructuras de datos para conjuntos disjuntos.
