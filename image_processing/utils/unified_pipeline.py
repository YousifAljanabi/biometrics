import os
import cv2
import numpy as np

# Optional PIL for saving PNGs with DPI; fallback to cv2 if unavailable
try:
    from PIL import Image as _PILImage  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _save_image(path: str, arr: np.ndarray, dpi=(500, 500)) -> None:
    """Save image; use PIL for PNG with DPI when available, else cv2.imwrite."""
    ext = os.path.splitext(path)[1].lower()
    if _HAS_PIL and ext in {".png", ".tif", ".tiff"}:
        _PILImage.fromarray(arr).save(path, dpi=dpi)
    else:
        cv2.imwrite(path, arr)


def run_pipeline(img: np.ndarray, mode: str = "normal", distortion_type: str | None = None,
                 block_size: int = 16, do_augments: bool = False):
    """
    Unified fingerprint processing pipeline.

    - mode="normal": uses geometric/noise routing + IITK enhancement (normal.py)
      - uses `distortion_type` to route pre-restoration.
    - mode="phone":  uses phone-capture preprocessing + IITK enhancement (normal_phone.py)

    Returns a dict of intermediate outputs. Keys depend on selected mode.
    """

    mode_lc = (mode or "").strip().lower()
    if mode_lc not in {"normal", "phone"}:
        raise ValueError("mode must be 'normal' or 'phone'")

    if mode_lc == "normal":
        # Defer import to avoid pulling phone-only deps
        from normal import route_and_restore, iitk_pipeline  # type: ignore

        restored = route_and_restore(img, distortion_type or "unknown")
        enhanced = iitk_pipeline(restored, block_size=block_size)
        outputs = {"restored": restored, **enhanced}
        return outputs

    else:  # mode == "phone"
        # Import phone pipeline only when required (brings in PIL/skimage)
        from normal_phone import phone_pipeline, generate_rotations_and_flips  # type: ignore

        outputs = phone_pipeline(img, block_size=block_size)
        if do_augments and "skeleton" in outputs:
            outputs.update(generate_rotations_and_flips(outputs["skeleton"]))
        return outputs


def cli():
    import argparse

    ap = argparse.ArgumentParser(description="Unified pipeline for normal or phone captures")
    ap.add_argument("--input", required=True, help="Path to fingerprint image")
    ap.add_argument("--mode", choices=["normal", "phone"], required=True,
                    help="Select processing mode")
    ap.add_argument("--type", default=None,
                    help="Distortion type label (used only in normal mode)")
    ap.add_argument("--outdir", default="out_results", help="Output directory")
    ap.add_argument("--block-size", type=int, default=16, help="Block size for IITK stages")
    ap.add_argument("--augments", action="store_true",
                    help="In phone mode, also save rotations and flips of skeleton")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {args.input}")

    outputs = run_pipeline(img, mode=args.mode, distortion_type=args.type,
                           block_size=args.block_size, do_augments=args.augments)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    for key, arr in outputs.items():
        if not isinstance(arr, np.ndarray):
            continue
        out_path = os.path.join(args.outdir, f"{stem}_{args.mode}_{key}.png")
        _save_image(out_path, arr)

    print(f"[OK] {args.mode} results saved to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    cli()

