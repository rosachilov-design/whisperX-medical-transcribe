"""Shared PaddleOCR construction for Zoom name-bar OCR (Latin + Cyrillic)."""

import os

# PaddlePaddle 3.3+ on CPU: oneDNN + PIR crashes with ConvertPirAttribute2RuntimeAttribute.
# Must pass enable_mkldnn=False into PaddleOCR (env FLAGS alone is not enough).

_OCR_DEVICE = None


def _paddle_has_gpu():
    try:
        import paddle

        return bool(
            paddle.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:
        return False


def resolve_ocr_device():
    """Return ``cpu`` or ``gpu:0`` (override with LOCAL_OCR_DEVICE=cpu|gpu|auto)."""
    override = (os.getenv("LOCAL_OCR_DEVICE") or "auto").strip().lower()
    if override in ("gpu", "gpu:0"):
        if not _paddle_has_gpu():
            raise RuntimeError(
                "LOCAL_OCR_DEVICE=gpu was requested, but PaddlePaddle cannot see a CUDA GPU. "
                "RunPod OCR must use a GPU worker with paddlepaddle-gpu installed."
            )
        return "gpu:0"
    if override == "cpu":
        return "cpu"
    if override not in ("", "auto"):
        raise ValueError(
            f"Invalid LOCAL_OCR_DEVICE={override!r}; use auto, cpu, or gpu."
        )
    return "gpu:0" if _paddle_has_gpu() else "cpu"


def get_ocr_device():
    """Cached device string used for the current process."""
    global _OCR_DEVICE
    if _OCR_DEVICE is None:
        _OCR_DEVICE = resolve_ocr_device()
    return _OCR_DEVICE


def create_zoom_paddle_ocr():
    """Return a PaddleOCR instance tuned for Zoom speaker name overlays.

    PaddleOCR 3.x does not accept ``lang="cyrillic"`` (that names a recognition
    model, not a locale). We load ``cyrillic_PP-OCRv5_mobile_rec`` explicitly.

    Uses GPU when ``paddlepaddle-gpu`` is installed and CUDA is available, unless
    ``LOCAL_OCR_DEVICE=cpu`` is set. If ``LOCAL_OCR_DEVICE=gpu`` is set, missing
    CUDA is a hard error.
    """
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install paddleocr and paddlepaddle to use local OCR."
        ) from exc

    device = get_ocr_device()
    common = {"device": device}
    if device == "cpu":
        common["enable_mkldnn"] = False

    explicit = {
        **common,
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_recognition_model_name": "cyrillic_PP-OCRv5_mobile_rec",
        "use_textline_orientation": False,
    }
    modern = {
        **common,
        "lang": "ru",
        "ocr_version": "PP-OCRv5",
        "use_textline_orientation": False,
    }
    legacy = {**common, "lang": "ru", "use_angle_cls": False}

    last_error = None
    for kwargs in (explicit, modern, legacy):
        try:
            ocr = PaddleOCR(**kwargs)
            print(f"PaddleOCR ready on {device}")
            return ocr
        except (TypeError, ValueError) as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    return PaddleOCR(**explicit)
