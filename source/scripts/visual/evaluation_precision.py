"""Explicit inference arithmetic policy shared by all learned comparison arms."""
import torch

def configure_evaluation_precision():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    return {'matmul_precision':torch.get_float32_matmul_precision(),
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'scope':'float32 inference without TF32; no claim of cross-hardware bitwise determinism'}
