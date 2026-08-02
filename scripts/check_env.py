
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import UNetPlusPlusResNeXt50


def main() -> None:
    print("=" * 60)
    print("  Urbaneuron — GPU Environment Check")
    print("=" * 60)

    print(f"\n[1] CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  ERROR: CUDA not available. Aborting.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"     Device count: {device_count}")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"     GPU {i}: {props.name}")
        print(f"       VRAM: {props.total_mem / (1024**3):.1f} GB")
        print(f"       Compute capability: {props.major}.{props.minor}")
        print(f"       Multi-processor count: {props.multi_processor_count}")

    major = torch.cuda.get_device_properties(0).major
    print(f"\n[2] TF32 support: {'YES' if major >= 8 else 'NO'} (compute {major}.x)")
    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        print("     TF32 enabled for matmul")

    torch.backends.cudnn.benchmark = True
    print("\n[3] cuDNN benchmark: enabled")

    print(f"\n[4] PyTorch: {torch.__version__}")
    print(f"     CUDA toolkit: {torch.version.cuda}")
    print(f"     cuDNN: {torch.backends.cudnn.version()}")

    print(f"\n[5] VRAM before model load:")
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"     allocated: {allocated:.2f} GB")
    print(f"     reserved:  {reserved:.2f} GB")

    print("\n[6] Building U-Net++ (ResNeXt-50) ...")
    model = UNetPlusPlusResNeXt50(num_classes=8, pretrained=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     Total params:     {total_params:,}")
    print(f"     Trainable params: {trainable_params:,}")

    model = model.cuda()

    print(f"\n     VRAM after model load:")
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"     allocated: {allocated:.2f} GB")
    print(f"     reserved:  {reserved:.2f} GB")

    batch_size = 32
    print(f"\n[7] Forward pass (batch={batch_size}, 512×512, mixed precision) ...")
    dummy = torch.randn(batch_size, 3, 512, 512, device="cuda")

    with torch.amp.autocast("cuda"):
        out = model(dummy)
    print(f"     Output shape: {out.shape}  (expected: [{batch_size}, 8, 512, 512])")
    assert out.shape == (batch_size, 8, 512, 512), f"Unexpected output shape: {out.shape}"

    print(f"\n     VRAM after forward pass:")
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"     allocated: {allocated:.2f} GB")
    print(f"     reserved:  {reserved:.2f} GB")
    peak_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
    print(f"     peak allocated: {peak_gb:.2f} GB")

    print(f"\n[8] Backward pass ...")
    dummy_target = torch.randint(0, 8, (batch_size, 512, 512), device="cuda")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss = loss_fn(out, dummy_target)
    loss.backward()
    print(f"     Loss: {loss.item():.4f}")

    peak_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
    print(f"     peak allocated (after backward): {peak_gb:.2f} GB")

    del model, dummy, dummy_target, out, loss
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    if peak_gb < 72:
        print("  RESULT: PASS — VRAM headroom is safe for training.")
    elif peak_gb < 78:
        print("  RESULT: TIGHT — consider reducing batch_size to 28.")
    else:
        print("  RESULT: FAIL — reduce batch_size and retry.")
    print(f"  Peak VRAM used: {peak_gb:.2f} GB / 80 GB")
    print("=" * 60)


if __name__ == "__main__":
    main()