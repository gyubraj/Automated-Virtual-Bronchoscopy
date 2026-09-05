"""Check whether the virtual bronchoscopy runtime is ready to use."""

from pathlib import Path
import argparse
import importlib
import os
import platform
import shutil
import sys


REQUIRED_MODULES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "SimpleITK": "SimpleITK",
    "scikit-image": "skimage",
    "PyVista": "pyvista",
    "VTK": "vtk",
    "PyTorch": "torch",
}


def module_version(module):
    return getattr(module, "__version__", "installed")


def main():
    default_data_root = os.environ.get(
        "AVB_DATA_ROOT",
        str(Path.home() / "AMS_Project" / "datasets_new"),
    )
    parser = argparse.ArgumentParser(description="Verify dependencies and required AVB project paths.")
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--checkpoint", default="saved_model_topology/wingsnet_best.pth")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    print("Automated Virtual Bronchoscopy setup check")
    print("  Python:", sys.version.split()[0])
    print("  platform:", platform.platform())
    failures = []

    if sys.version_info < (3, 10):
        failures.append("Python 3.10 or newer is required.")

    loaded = {}
    for label, import_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(import_name)
            loaded[import_name] = module
            print(f"  [OK] {label}: {module_version(module)}")
        except Exception as exc:
            failures.append(f"Missing or unusable dependency {label}: {exc}")
            print(f"  [FAIL] {label}: {exc}")

    torch = loaded.get("torch")
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        print("  CUDA available:", cuda_available)
        print("  PyTorch CUDA build:", torch.version.cuda)
        if cuda_available:
            print("  GPU:", torch.cuda.get_device_name(0))
        if args.require_cuda and not cuda_available:
            failures.append("CUDA was required but torch.cuda.is_available() is False.")

    idc_path = shutil.which("idc")
    if idc_path:
        print("  [OK] IDC CLI:", idc_path)
    else:
        failures.append("IDC CLI not found. Install requirements.txt or run: pip install --upgrade idc-index")
        print("  [FAIL] IDC CLI: command not found")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if checkpoint.is_file():
        print("  [OK] checkpoint:", checkpoint)
    else:
        failures.append(f"Checkpoint not found: {checkpoint}")
        print("  [FAIL] checkpoint:", checkpoint)

    data_root = Path(args.data_root).expanduser().resolve()
    if data_root.is_dir():
        print("  [OK] data root:", data_root)
    else:
        print("  [WARN] data root does not exist yet:", data_root)
        print("         Create it before preprocessing or inference.")

    if failures:
        print("\nSetup is not ready:")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)

    print("\nSetup is ready.")


if __name__ == "__main__":
    main()
