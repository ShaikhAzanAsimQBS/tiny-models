"""
Export the fine-tuned YOLOX-Tiny checkpoint to ONNX.

Uses the same Exp / class-count / input-size as training so the graph
matches what you just trained. Follows the official YOLOX export recipe
(SiLU swap, optional decode-in-graph, opset 11, optional onnx-simplifier).

Every setting lives in config.py - no command line arguments. Run with:

    python -m training.export_yolox_tiny_onnx
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config

sys.path.insert(0, config.YOLOX_REPO_DIR)

import torch
from torch import nn

from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

from training.train_yolox_tiny import build_exp


def _resolve_checkpoint() -> str:
    ckpt_path = config.YOLOX_ONNX_CHECKPOINT
    if os.path.isfile(ckpt_path):
        return ckpt_path
    latest = os.path.join(
        config.YOLOX_OUTPUT_DIR, config.YOLOX_EXPERIMENT_NAME, "latest_ckpt.pth"
    )
    if os.path.isfile(latest):
        print(f"[export_yolox_tiny_onnx] {ckpt_path} not found; falling back to {latest}")
        return latest
    raise FileNotFoundError(
        f"No YOLOX checkpoint at {ckpt_path} (or {latest}). "
        f"Train first, or set config.YOLOX_ONNX_CHECKPOINT."
    )


def _export_onnx(model: nn.Module, dummy_input: torch.Tensor, output_path: str) -> None:
    input_name = "images"
    output_name = "output"
    dynamic_axes = None
    if config.YOLOX_ONNX_DYNAMIC_BATCH:
        dynamic_axes = {input_name: {0: "batch"}, output_name: {0: "batch"}}

    # dynamo=False: this venv's torch defaults dynamo=True, which needs
    # onnxscript and is not what official YOLOX export uses. The TorchScript
    # exporter is the path Megvii's tools/export_onnx.py relies on.
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        opset_version=int(config.YOLOX_ONNX_OPSET),
        do_constant_folding=True,
        dynamo=False,
    )


def _maybe_simplify(output_path: str) -> None:
    if not config.YOLOX_ONNX_SIMPLIFY:
        return
    try:
        import onnx
        from onnxsim import simplify
    except ImportError as exc:
        print(f"[export_yolox_tiny_onnx] onnx-simplifier not installed ({exc}); "
              f"keeping the unsimplified graph. pip install onnx onnxsim")
        return

    print("[export_yolox_tiny_onnx] simplifying with onnxsim...")
    onnx_model = onnx.load(output_path)
    try:
        model_simp, check = simplify(onnx_model)
    except Exception as exc:
        print(f"[export_yolox_tiny_onnx] onnxsim failed ({type(exc).__name__}: {exc}); "
              f"keeping the unsimplified graph.")
        return
    if not check:
        print("[export_yolox_tiny_onnx] onnxsim produced a graph that failed validation; "
              "keeping the unsimplified graph.")
        return
    onnx.save(model_simp, output_path)
    print("[export_yolox_tiny_onnx] simplified ONNX saved")


def _verify(output_path: str, dummy_input: torch.Tensor, torch_out: torch.Tensor) -> None:
    try:
        import onnx
    except ImportError:
        print("[export_yolox_tiny_onnx] onnx package not installed; skip checker. pip install onnx")
        return

    onnx.checker.check_model(onnx.load(output_path))
    print("[export_yolox_tiny_onnx] onnx.checker: OK")

    try:
        import onnxruntime as ort
    except ImportError:
        print("[export_yolox_tiny_onnx] onnxruntime not installed; skip numeric check. "
              "pip install onnxruntime")
        return

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"images": dummy_input.numpy()})[0]
    torch_np = torch_out.detach().cpu().numpy()
    if ort_out.shape != torch_np.shape:
        raise RuntimeError(
            f"ONNX output shape {ort_out.shape} != torch output shape {torch_np.shape}"
        )
    max_abs = float(abs(ort_out - torch_np).max())
    print(f"[export_yolox_tiny_onnx] onnxruntime vs torch max |diff| = {max_abs:.6g} "
          f"(shape {tuple(ort_out.shape)})")


def main():
    ckpt_path = _resolve_checkpoint()
    output_path = config.YOLOX_ONNX_OUTPUT_PATH
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    exp = build_exp()
    print(f"[export_yolox_tiny_onnx] classes={exp.num_classes} "
          f"test_size={exp.test_size} opset={config.YOLOX_ONNX_OPSET}")
    print(f"[export_yolox_tiny_onnx] checkpoint: {ckpt_path}")

    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = bool(config.YOLOX_ONNX_DECODE_IN_INFERENCE)

    dummy_input = torch.randn(
        int(config.YOLOX_ONNX_BATCH_SIZE), 3, exp.test_size[0], exp.test_size[1]
    )
    with torch.no_grad():
        torch_out = model(dummy_input)
        if isinstance(torch_out, (tuple, list)):
            torch_out = torch_out[0]

    print(f"[export_yolox_tiny_onnx] exporting -> {output_path}")
    _export_onnx(model, dummy_input, output_path)
    print(f"[export_yolox_tiny_onnx] wrote {output_path} "
          f"({os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")

    _maybe_simplify(output_path)
    _verify(output_path, dummy_input, torch_out)
    print("[export_yolox_tiny_onnx] done.")


if __name__ == "__main__":
    main()
