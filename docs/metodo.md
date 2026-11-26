# Método de estudio

Este documento fija cómo se estudia en `ailib-ts`: qué se escribe, dónde vive, cómo se verifica y
cuándo un concepto se da por entendido. Vale para todo el repo.

## 1. Principios

- **Aprender construyendo.** Cada concepto se entiende, se implementa y se verifica; no se copia de
  otra librería.
- **Autocontenido por módulo.** La nota explica todo lo que ese código usa. Si el código lo usa y la
  nota no lo explica, la nota está incompleta.
- **Una nota por concepto.** Canónica y única; los demás módulos la enlazan, no la repiten.
- **La nota va antes que el código** (como plan) y se corrige después (con lo aprendido al
  implementarlo).
- **Sin nota no hay módulo.** La teoría es parte de la definición de «hecho», no un extra.
- **Las referencias son para profundizar, nunca para entender.** Van al final de la nota.

## 2. Un módulo

Un módulo es un concepto con cuatro piezas y un puntero:

| Pieza | Dónde | Qué es |
| --- | --- | --- |
| Teoría | `docs/teoria/<tema>/<concepto>.md` | la nota |
| Código | `src/<área>/...` | la implementación |
| Test | `test/...` | la verificación, con test numérico cuando aplique |
| Experimento | `lab/<concepto>/` | script reproducible que lo pone a prueba |
| Puntero | primera línea del archivo fuente (salvo el punto de entrada `src/index.ts`) | `// Teoría: docs/teoria/<tema>/<concepto>.md` |

Los temas de `docs/teoria/` son las **capas del proyecto**, no las carpetas del código: `nucleo`,
`redes`, `evolucion`, `secuencias`, `transformer`, `entrenamiento`, `eficiencia`, `ideas`. El índice
hace el mapa entre ambos.

## 3. La nota

Formato fijo; las ocho primeras secciones son obligatorias y la novena opcional:

```markdown
# <Concepto>

## 1. Qué problema resuelve
Intuición, sin fórmulas.

## 2. Definición y notación
Fórmula y qué significa cada símbolo.

## 3. Derivación
Paso a paso: el porqué, no solo el resultado.

## 4. Ejemplo a mano
Números pequeños y el resultado esperado.

## 5. Implementación
Qué archivos de `src/` y qué tests la implementan (puede ser más de uno).

## 6. Verificación
Qué test numérico o gradcheck lo comprueba, con qué tolerancia y qué fallaría si estuviera mal.

## 7. Límites y errores comunes
Cuándo no aplica y qué se rompe.

## 8. Mi explicación
La persona, en sus palabras y sin mirar arriba.

## 9. Para profundizar (opcional)
Libros o papers; nunca necesarios para entender la nota.
```

Reglas de la nota:

- 1–3 páginas. Si necesita más, son dos conceptos.
- Intuición antes que formalismo; cada fórmula con su lectura en palabras.
- El ejemplo a mano se resuelve con números, no solo se enuncia.
- La sección 6 dice **qué fallaría**: un test que no puede fallar no verifica.
- La sección 8 la escribe la persona; no se rellena por ella.

## 4. El índice

`docs/teoria/README.md` tiene una fila por concepto:

| Concepto | Fase | Nota | Código | Test | Estado |
| --- | --- | --- | --- | --- | --- |

- **Fase** es la de la hoja de ruta (`F0`…`F8`), para saber cuándo toca.
- **Estado** es `borrador` o `estable`.
- Se actualiza en el mismo cambio que la nota o el código.
- Es la respuesta rápida a «¿dónde está la teoría de esto?».

## 5. Fundamentos y glosario

- `docs/fundamentos/`: mini-primers (álgebra lineal, derivadas y regla de la cadena, probabilidad,
  entropía) para que ninguna nota asuma un libro abierto. Se enlazan desde la sección 2 o 3.
- `docs/glosario.md`: término · definición de una línea · nota canónica. Se añade el término la
  primera vez que aparece.

## 6. Nombres y formato

- Notas en español; código e identificadores en inglés.
- El archivo de un concepto se nombra con el término técnico que usa el código (`tensor.md`,
  `attention.md`, `layer-norm.md`). Los documentos de método van en español (`metodo.md`,
  `glosario.md`).
- Markdown con LaTeX (`$...$`) y Mermaid o ASCII para los diagramas.
- Enlaces relativos y **a la nota, no a la línea**: las líneas cambian.
- Nada de copiar texto de libros o papers: la explicación es propia y al final se cita de dónde
  salió la idea.

## 7. Cuándo se da por entendido

Un concepto está entendido —y el módulo «hecho»— cuando se cumplen las seis:

- [ ] La nota tiene las secciones 1–8.
- [ ] El código existe y empieza con su puntero a la nota.
- [ ] Los tests pasan, incluido el numérico o gradcheck cuando aplique.
- [ ] El experimento de `lab/` corre y su resultado coincide con lo esperado.
- [ ] El ejemplo a mano de la nota se reproduce con el código.
- [ ] La persona escribe la sección 8 sin mirar la nota.

## 8. Cómo trabajamos

- La IA redacta la nota, el andamiaje y los tests; la persona los lee, pregunta y corrige.
- La sección 8 la escribe la persona: es la puerta real de «entendido».
- La hoja de ruta (en la capa) manda en el **orden y las puertas** de cada fase; este documento manda
  en la **forma**.
- Cada cambio de nota o de código actualiza el índice en el mismo paso.

## 9. Verificación automática (prevista)

Cuando haya volumen, un `npm run docs:check` comprobará: que todo archivo de `src/` tenga puntero a
una nota existente, que el índice no tenga filas rotas y que los enlaces relativos resuelvan. No se
implementa en F0.
