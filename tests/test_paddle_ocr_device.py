import importlib
import os
import sys
import types
import unittest
from unittest import mock


def load_factory_with_fake_paddle(compiled_with_cuda, device_count):
    sys.modules.pop("paddle_ocr_factory", None)

    fake_paddle = types.ModuleType("paddle")
    fake_paddle.is_compiled_with_cuda = lambda: compiled_with_cuda
    fake_paddle.device = types.SimpleNamespace(
        cuda=types.SimpleNamespace(device_count=lambda: device_count)
    )

    sys.modules["paddle"] = fake_paddle
    return importlib.import_module("paddle_ocr_factory")


class PaddleOcrDeviceTests(unittest.TestCase):
    def test_runpod_gpu_override_fails_if_paddle_cannot_see_cuda(self):
        factory = load_factory_with_fake_paddle(compiled_with_cuda=False, device_count=0)

        with mock.patch.dict(os.environ, {"LOCAL_OCR_DEVICE": "gpu"}):
            with self.assertRaisesRegex(RuntimeError, "LOCAL_OCR_DEVICE=gpu"):
                factory.resolve_ocr_device()

    def test_runpod_gpu_override_uses_gpu_when_available(self):
        factory = load_factory_with_fake_paddle(compiled_with_cuda=True, device_count=1)

        with mock.patch.dict(os.environ, {"LOCAL_OCR_DEVICE": "gpu"}):
            self.assertEqual(factory.resolve_ocr_device(), "gpu:0")

    def test_auto_mode_can_fall_back_to_cpu_for_non_runpod_tests(self):
        factory = load_factory_with_fake_paddle(compiled_with_cuda=False, device_count=0)

        with mock.patch.dict(os.environ, {"LOCAL_OCR_DEVICE": "auto"}):
            self.assertEqual(factory.resolve_ocr_device(), "cpu")


if __name__ == "__main__":
    unittest.main()
