"""Extract bounded diagnostic prefixes; never register or replace a model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operator", required=True, type=int, action="append")
    args = parser.parse_args()
    source_hash = digest(args.model)
    if source_hash != "6f03564f13abfae18185581a809cb5c0ee5c0a425dc5f85102194b2b321eba3d":
        parser.error("Diagnostic prefixes require the reviewed original Large export")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.interpreter import Interpreter
    from convert_depth import verify_environment
    verify_environment()
    original = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(args.model.read_bytes(), 0))
    graph = original.subgraphs[0]
    pixels = np.fromfile(args.input, dtype="<f4").reshape(1, 3, 518, 686)
    codes = {value: name for name, value in vars(schema.BuiltinOperator).items() if isinstance(value, int)}
    results = []
    for index in args.operator:
        if not 0 <= index < len(graph.operators):
            raise ValueError("Operator index outside the original graph")
        selected = graph.operators[index]
        if len(selected.outputs) != 1:
            raise ValueError("Probe requires a single-output operator")
        output = int(selected.outputs[0])
        tensor = graph.tensors[output]
        shape = [int(value) for value in tensor.shape]
        if tensor.type != schema.TensorType.FLOAT32 or not 0 < int(np.prod(shape)) <= 16 * 1024 * 1024:
            raise ValueError("Probe output must be bounded, static FP32")
        live = {output}
        kept = []
        for operator in reversed(graph.operators[:index + 1]):
            if any(int(value) in live for value in operator.outputs):
                kept.append(copy.deepcopy(operator))
                live.update(int(value) for value in operator.inputs if value >= 0)
                live.update(int(value) for value in operator.outputs if value >= 0)
        live.update(int(value) for value in graph.inputs)
        tensors = sorted(live)
        mapping = {old: new for new, old in enumerate(tensors)}
        buffers = sorted({graph.tensors[value].buffer for value in tensors} | {0})
        buffer_mapping = {old: new for new, old in enumerate(buffers)}
        prefix = schema.SubGraphT()
        prefix.name = b"diagnostic_prefix"
        prefix.tensors = [copy.copy(graph.tensors[value]) for value in tensors]
        for item in prefix.tensors:
            item.buffer = buffer_mapping[item.buffer]
        prefix.inputs = [mapping[int(value)] for value in graph.inputs]
        prefix.outputs = [mapping[output]]
        prefix.operators = list(reversed(kept))
        for operator in prefix.operators:
            operator.inputs = [mapping[int(value)] if value >= 0 else -1 for value in operator.inputs]
            operator.outputs = [mapping[int(value)] for value in operator.outputs]
        model = schema.ModelT()
        model.version = original.version
        model.operatorCodes = original.operatorCodes
        model.subgraphs = [prefix]
        model.buffers = [original.buffers[value] for value in buffers]
        builder = flatbuffers.Builder(1024)
        builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
        path = args.output_dir / f"op-{index}.tflite"
        path.write_bytes(builder.Output())
        runtime = Interpreter(model_path=str(path), num_threads=4)
        runtime.allocate_tensors()
        runtime.set_tensor(runtime.get_input_details()[0]["index"], pixels)
        runtime.invoke()
        reference = runtime.get_tensor(runtime.get_output_details()[0]["index"])
        if list(reference.shape) != shape or not np.isfinite(reference).all():
            raise ValueError("Invalid CPU diagnostic reference")
        reference.astype("<f4").tofile(args.output_dir / f"op-{index}.reference.f32")
        pixels.tofile(args.output_dir / f"op-{index}.input.f32")
        record = {"operator": index, "kind": codes[original.operatorCodes[selected.opcodeIndex].builtinCode],
                  "tensor": output, "name": tensor.name.decode(), "shape": shape,
                  "inputs": [{"tensor": int(value), "shape": [int(dimension) for dimension in graph.tensors[value].shape]}
                             for value in selected.inputs if value >= 0],
                  "model_sha256": digest(path), "elements": reference.size,
                  "reference_sha256": digest(args.output_dir / f"op-{index}.reference.f32")}
        results.append(record)
        print(json.dumps({key: record[key] for key in ("operator", "kind", "shape", "model_sha256")}), flush=True)
    (args.output_dir / "probes.json").write_text(json.dumps({"source_sha256": source_hash,
        "input_sha256": digest(args.input), "probes": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
