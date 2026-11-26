# Reglas de trabajo

> **Punto de entrada:** al abrir este workspace, lee **solo las reglas críticas** de abajo. El
> detalle completo (24 reglas, permisos, auditoría, flujo) está en `ai/project.md`; para el día a
> día bastan §4–§5 y §10 (niveles, reglas y arranque). Los hechos del entorno, en `ai/platform.md`;
> los repos, en `ai/repos.md`; el estado, en `ai/state.md`. Los prompts y sus frases, en
> `ai/prompts/README.md`.

## Reglas críticas (se siguen siempre)

1. **Secretos.** No se leen, no se imprimen, no se copian, no se commitean. Nunca.

2. **Producción, dinero y datos.** La IA no decide sobre ellos: desplegar, migrar datos o tocar
   producción o dinero exige permiso explícito y solo con los scripts del proyecto (`ops/`), si
   existen. Un rollback se hace con la imagen que ya existía (su digest), nunca volviendo el árbol atrás.

3. **Borrados e irreversibles.** Todo `delete` —pods, filas, ramas, archivos, imágenes— exige
   permiso explícito sobre ese objetivo concreto y caduca en una hora. Nunca `push --force`,
   `reset --hard` ni reescribir historial.

4. **Verificar.** Ningún cambio se da por bueno sin verificación: build/test/lint si existen; si no
   hay verificador, revisión humana explícita y se deja constancia de ello.

5. **Código de IA = no confiable hasta verificar.** Todo cambio asistido por IA se marca con
   `Assisted-by: <humano>`; la IA no se firma a sí misma. Lethal Trifecta: no dar a un asistente, a
   la vez, datos privados + contenido no confiable + capacidad de actuar.

6. **Evidencia.** Un hallazgo sin `archivo:línea` no cuenta; un fallo no se silencia ni se salta.

7. **Primero el panorama, después el código.** Un «sí» a una idea o a una lista de opciones no es
   permiso para escribir código. En todo lo que no sea un arreglo local, antes de tocar archivos se
   explica en palabras simples qué piezas hay y para qué sirve cada una, se acuerda ese diseño y solo
   entonces se escribe, un paso por vez. Lo que no esté claro se pregunta; no se rellena por cuenta
   propia.

8. **El estado vive en su fuente.** Lo que es estado —tareas, hojas de ruta, bitácora— vive en el
   servicio o en la capa, no en un archivo local. Se escribe por su camino: **el MCP** si lo hay, la
   **herramienta del proyecto** con sesión si no, y si no hay conexión, **la cola local del proyecto**,
   que se envía al reconectar y se borra (`tareas.mjs enviar`, `hojas.mjs enviar`). Nunca se sustituye
   la fuente por un documento local: una copia que no se vacía es una segunda verdad. Si no hay MCP ni
   sesión, se dice y se encola; no se escribe el estado en un `.md`.

## Escrutinio proporcional al riesgo

Lo de arriba es estricto **porque es crítico**. Para todo lo demás, **libertad creativa**: el nivel
de ceremonia y revisión debe ser proporcional al riesgo del cambio, no fijo. Un cambio local y
aislado se aprueba rápido; un cambio de contrato, datos, seguridad o producción exige la ceremonia
completa.

## Referencia (consultar cuando aplica)

- **Configuración y rutas:** no se editan YAML, manifiestos, secretos ni config sin permiso del
  dueño; antes de crear o cambiar una ruta (path + método) se anuncia.
- **Merges:** prohibido `-X theirs` / `-X ours`; en un conflicto se resuelve a mano o se pregunta.
- **Diagnóstico antes que código:** primero el pedido y la respuesta reales (URL, método, código
  HTTP y cuerpo), después el código.
- **Estilo del código que escribe la IA:** el de `ai/docs/estilo.md` (por defecto; el equipo lo
  cambia en `decisions.md`).
- **Flujo:** propuesta → aprobación → verificación → registro (`project.md` §4–§10). Mapa en
  `ai/index.md`; búsquedas con `links.mjs`/`query.mjs`.
