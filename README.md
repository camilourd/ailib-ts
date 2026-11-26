# ailib-ts

Librería de estudio de IA en TypeScript.

## Desarrollo

```bash
npm ci
npm run build
```

- `npm run build` — compila `src/` a `dist/` con tipos.
- `npm run typecheck` — comprueba tipos sin emitir.
- `npm test` — pruebas con `node:test`.
- `npm run lint` — ESLint con las reglas de estilo del estándar.

## Estructura

- `src/` — código (punto de entrada: `src/index.ts`).
- `test/` — pruebas (`*.test.ts`).
- `dist/` — salida compilada; no se commitea.

## Uso

```ts
import { VERSION } from "ailib-ts";
```
