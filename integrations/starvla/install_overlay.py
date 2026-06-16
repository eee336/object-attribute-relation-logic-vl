from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_file(src: Path, dst: Path, *, executable: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if executable:
        dst.chmod(dst.stat().st_mode | 0o111)
    print(f"copied {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install OARL-VLA StarVLA overlay files.")
    parser.add_argument("--starvla-root", type=Path, required=True, help="Path to a StarVLA checkout.")
    parser.add_argument(
        "--oarlvla-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to this OARL-VLA repository.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing overlay files.")
    args = parser.parse_args()

    starvla_root = args.starvla_root.resolve()
    oarlvla_root = args.oarlvla_root.resolve()
    overlay_dir = Path(__file__).resolve().parent
    if not (starvla_root / "starVLA" / "model" / "framework" / "VLM4A").exists():
        raise FileNotFoundError(f"Not a StarVLA checkout: {starvla_root}")
    if not (oarlvla_root / "src" / "oarlvla").exists():
        raise FileNotFoundError(f"Not an OARL-VLA checkout: {oarlvla_root}")

    targets = [
        (
            overlay_dir / "OARLVLAQwenPI.py",
            starvla_root / "starVLA" / "model" / "framework" / "VLM4A" / "OARLVLAQwenPI.py",
            False,
        ),
        (
            overlay_dir / "oarlvla_qwenpi_libero.yaml",
            starvla_root / "examples" / "LIBERO" / "train_files" / "oarlvla_qwenpi_libero.yaml",
            False,
        ),
        (
            overlay_dir / "run_oarlvla_libero_train.sh",
            starvla_root / "examples" / "LIBERO" / "train_files" / "run_oarlvla_libero_train.sh",
            True,
        ),
    ]
    for _, dst, _ in targets:
        if dst.exists() and not args.force:
            raise FileExistsError(f"{dst} exists; pass --force to overwrite.")
    for src, dst, executable in targets:
        copy_file(src, dst, executable=executable)

    print("\nNext steps:")
    print(f"  cd {starvla_root}")
    print(f"  export OARLVLA_REPO={oarlvla_root}")
    print("  export PYTHONPATH=${OARLVLA_REPO}/src:${PYTHONPATH}")
    print("  python starVLA/model/framework/VLM4A/OARLVLAQwenPI.py --config_yaml examples/LIBERO/train_files/oarlvla_qwenpi_libero.yaml")
    print("  bash examples/LIBERO/train_files/run_oarlvla_libero_train.sh")


if __name__ == "__main__":
    main()
