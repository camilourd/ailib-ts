// Teoría: docs/teoria/entrenamiento/checkpoints.md

import { readFileSync, renameSync, writeFileSync } from "node:fs";

import { Tensor } from "../core/tensor.ts";

export const CHECKPOINT_FORMAT = "ailib-ts.checkpoint";
export const CHECKPOINT_VERSION = 1;

export interface CheckpointMetadata {
    [key: string]: unknown;
}

export interface Checkpoint {
    metadata: CheckpointMetadata;
    parameters: Record<string, Tensor>;
}

interface ParameterRecord {
    name: string;
    shape: number[];
    data: number[];
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
    typeof value === "object" && value !== null && !Array.isArray(value);

const shapeSize = (shape: readonly number[]): number =>
    shape.reduce((total, dimension) => total * dimension, 1);

/** Guarda parámetros con nombre y metadatos; escribe a un temporal y renombra. */
export function saveCheckpoint(
    path: string,
    parameters: Record<string, Tensor>,
    metadata: CheckpointMetadata = {},
): void {
    const records: ParameterRecord[] = Object.entries(parameters).map(([name, tensor]) => {
        if (name.length === 0) throw new Error("un parámetro no puede llamarse vacío");
        for (const value of tensor.data) {
            if (!Number.isFinite(value)) {
                throw new Error(`el parámetro «${name}» tiene un valor no finito`);
            }
        }
        return { name, shape: [...tensor.shape], data: [...tensor.data] };
    });
    const payload = {
        format: CHECKPOINT_FORMAT,
        version: CHECKPOINT_VERSION,
        metadata,
        parameters: records,
    };
    const temporary = `${path}.tmp`;
    writeFileSync(temporary, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
    renameSync(temporary, path);
}

/** Carga un checkpoint validando formato, formas y datos. */
export function loadCheckpoint(
    path: string,
    options: { requiresGrad?: boolean } = {},
): Checkpoint {
    let raw: string;
    try {
        raw = readFileSync(path, "utf8");
    } catch {
        throw new Error(`no se puede leer el checkpoint ${path}`);
    }
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        throw new Error(`${path} no es JSON válido`);
    }
    if (!isRecord(parsed)) throw new Error(`${path} no es un checkpoint válido`);
    if (parsed.format !== CHECKPOINT_FORMAT || parsed.version !== CHECKPOINT_VERSION) {
        throw new Error(`${path} no es un checkpoint ${CHECKPOINT_FORMAT} v${CHECKPOINT_VERSION}`);
    }
    if (!isRecord(parsed.metadata)) throw new Error(`${path}: metadatos inválidos`);
    if (!Array.isArray(parsed.parameters)) throw new Error(`${path}: falta la lista de parámetros`);

    const parameters: Record<string, Tensor> = {};
    for (const entry of parsed.parameters) {
        if (!isRecord(entry)) throw new Error(`${path}: parámetro inválido`);
        const name = entry.name;
        const shape = entry.shape;
        const data = entry.data;
        if (typeof name !== "string" || name.length === 0) {
            throw new Error(`${path}: nombre de parámetro inválido`);
        }
        if (name in parameters) throw new Error(`${path}: parámetro duplicado «${name}»`);
        if (
            !Array.isArray(shape) ||
            !shape.every((dimension) => Number.isInteger(dimension) && (dimension as number) >= 0)
        ) {
            throw new Error(`${path}: forma inválida en «${name}»`);
        }
        if (
            !Array.isArray(data) ||
            !data.every((value) => typeof value === "number" && Number.isFinite(value))
        ) {
            throw new Error(`${path}: datos inválidos en «${name}»`);
        }
        const dimensions = shape as number[];
        if (data.length !== shapeSize(dimensions)) {
            throw new Error(
                `${path}: «${name}» tiene ${data.length} valores y la forma pide ${shapeSize(dimensions)}`,
            );
        }
        parameters[name] = new Tensor(Float32Array.from(data as number[]), dimensions, {
            requiresGrad: options.requiresGrad ?? false,
        });
    }
    return { metadata: parsed.metadata, parameters };
}
