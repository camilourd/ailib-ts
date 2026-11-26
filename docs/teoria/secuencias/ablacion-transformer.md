# Ablación del transformer

## 1. Qué problema resuelve

Un transformer tiene varias piezas: atención, RoPE, FFN, residuales y LayerNorm. La pregunta no es
«¿están?», sino **¿cuánto aporta cada una?** Una ablación quita una pieza, deja todo lo demás igual y
compara.

## 2. Protocolo

- Mismo texto (624 bytes de train, 64 de test), mismos 3000 pasos, misma semilla.
- Mismo LM: `d_model=32`, 4 cabezas, un bloque, `d_ff=64`.
- Variantes:
  - **completo**;
  - **sin RoPE**;
  - **sin atención** (solo FFN por posición);
  - **sin FFN** (solo atención).
- Métrica: BPC en test; parámetros totales y pérdida de train como contexto.

## 3. Resultado

| Variante | Parámetros | Pérdida train | BPC test |
| --- | --- | --- | --- |
| Completo | 25.248 | 1.2402 | 5.092 |
| Sin RoPE | 25.248 | 1.6560 | 4.853 |
| Sin atención | 20.960 | 2.2426 | **4.128** |
| Sin FFN | 20.992 | 1.4621 | 5.091 |

Muestras con el mismo arranque:

```
completo:     En un acÛpadenr. pa ellires el lsate io o a esa, esex e
sin RoPE:     En un a un. ladel s co lay tela pa un po un demeco latre
sin atención: En un Cay la dequere e metes ca mparendo un cie addegund
sin FFN:      En un Hay papapadapade pay to elo bomery y agunacualeso
```

## 4. Lectura

- **Sin atención es la mejor en test** (4.128 BPC) y tiene menos parámetros. Con 624 bytes, la atención
  parece sobreajustar: el FFN por posición aprende frecuencias locales y generaliza mejor.
- **Sin RoPE mejora el BPC** (4.853 vs 5.092) aunque empeora la pérdida de train. En secuencias de 33
  bytes y con este presupuesto, la posición relativa no está pagando.
- **Sin FFN no cambia el BPC** (5.091) y ahorra parámetros; el FFN tampoco aporta aquí.
- **El completo tiene la mejor pérdida de train y el peor BPC**: la señal clásica de sobreajuste.

Esto **no** significa que atención, RoPE o FFN sean inútiles: significa que a esta escala (un bloque,
32 dimensiones, 624 bytes, 3000 pasos) no compensan. La ablación mide el presupuesto, no el techo de la
arquitectura.

## 5. Implementación

- [`lab/transformer/ablation.ts`](../../lab/transformer/ablation.ts): entrena las cuatro variantes y
  calcula BPC y muestras.
- [`src/nn/transformer-block.ts`](../../src/nn/transformer-block.ts): opciones `attention` y `ffn` para
  apagar piezas; `parameters()` excluye lo apagado.
- [`src/models/transformer-lm.ts`](../../src/models/transformer-lm.ts): pasa `rope`, `attention` y
  `ffn` a los bloques.

## 6. Verificación

- Las cuatro variantes usan los mismos datos, pasos, semilla y contexto.
- El número de parámetros baja al apagar atención o FFN.
- 161/161 pruebas, incluidas las variantes apagadas del bloque, y `verify.mjs` en verde.
- El resultado negativo se conserva; no se elige la variante que «suena mejor».

## 7. Límites y errores comunes

- **Una semilla y 64 bytes de test**: el BPC es ruidoso; la diferencia de atención es grande, pero no
  es una ley.
- **Un bloque y 3000 pasos**: el transformer no está en su régimen.
- **Sin más datos**: la ablación dice qué pasa aquí, no qué pasa con 100 MB de texto.
- **BPC no es calidad de texto**: las muestras son fragmentos en todas las variantes.
- **Quitar atención no es un transformer**: es la ablación, no una alternativa recomendada.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Vaswani et al. (2017), *Attention Is All You Need* (ablaciones del transformer original).
- Hooker, *The Hardware Lottery* y discusiones sobre ablación y coste.
