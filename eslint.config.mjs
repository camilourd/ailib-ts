// Reglas de estilo del estándar (copia de config/eslint de la raíz) más el parser de TypeScript.
import tseslint from "typescript-eslint";

export default tseslint.config(
    { ignores: ["dist/**", "coverage/**", "node_modules/**"] },
    ...tseslint.configs.recommended,
    {
        files: ["**/*.ts"],
        rules: {
            // Regla 20: nada de continue ni break.
            "no-continue": "error",
            "no-restricted-syntax": [
                "error",
                { selector: "BreakStatement", message: "No uses break: un switch devuelve o lanza en cada caso." },
                { selector: "ContinueStatement", message: "No uses continue: filtra antes o saca el cuerpo a un método." },
            ],
            // Regla 23: la clase no lee el entorno; solo el punto de composición.
            "no-process-env": "error",
        },
    },
    {
        // El punto de composición es el único que lee process.env.
        files: ["**/index.ts", "**/bootstrap/**", "**/composition/**"],
        rules: { "no-process-env": "off" },
    },
);
