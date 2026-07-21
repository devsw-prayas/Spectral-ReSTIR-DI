import sys
import torch
import scipy
import matplotlib


def checkEnv():
    print("=" * 63)
    print(" Spectral ReSTIR DI — Environment Check")
    print("=" * 63)

    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Platform       : {sys.platform}")

    print("\n[Dependencies]")
    print(f"PyTorch        : {torch.__version__}")
    print(f"SciPy          : {scipy.__version__}")
    print(f"Matplotlib     : {matplotlib.__version__}")

    print("\n[Dtype]")
    print(f"Default dtype  : {torch.get_default_dtype()}")
    assert torch.get_default_dtype() == torch.float64, "float64 not set — call torch.set_default_dtype(torch.float64) at entry point"
    print("float64        : OK")

    print("\n[Hardware]")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available : {'YES' if cuda_available else 'NO'}")
    if cuda_available:
        print(f"GPU Device     : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version   : {torch.version.cuda}")

    print("\n" + "=" * 63)
    if cuda_available:
        print(" Environment is ready for GPU spectral R&D.")
    else:
        print(" Environment is ready (CPU-only).")
    print("=" * 63)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    checkEnv()
