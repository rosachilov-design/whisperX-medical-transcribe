import contextlib
import importlib
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class FakeApp:
    def add_middleware(self, *args, **kwargs):
        return None

    def mount(self, *args, **kwargs):
        return None

    def get(self, *args, **kwargs):
        return self._decorator

    def post(self, *args, **kwargs):
        return self._decorator

    def _decorator(self, fn):
        return fn


class FakeTransferConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeS3Client:
    def list_objects_v2(self, *args, **kwargs):
        return {}

    def download_file(self, *args, **kwargs):
        return None

    def get_object(self, *args, **kwargs):
        raise FileNotFoundError


def load_server_module():
    sys.modules.pop("server", None)

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.FastAPI = lambda *args, **kwargs: FakeApp()
    fake_fastapi.UploadFile = object
    fake_fastapi.File = lambda *args, **kwargs: None
    fake_fastapi.Body = lambda *args, **kwargs: None

    fake_cors = types.ModuleType("fastapi.middleware.cors")
    fake_cors.CORSMiddleware = object

    fake_staticfiles = types.ModuleType("fastapi.staticfiles")
    fake_staticfiles.StaticFiles = lambda *args, **kwargs: object()

    fake_responses = types.ModuleType("fastapi.responses")
    fake_responses.FileResponse = object
    fake_responses.JSONResponse = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

    fake_pydantic = types.ModuleType("pydantic")
    fake_pydantic.BaseModel = object
    fake_pydantic.Field = lambda *args, **kwargs: None

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.__path__ = []
    fake_boto3.client = lambda *args, **kwargs: FakeS3Client()

    fake_boto3_s3 = types.ModuleType("boto3.s3")
    fake_boto3_s3.__path__ = []
    fake_boto3_transfer = types.ModuleType("boto3.s3.transfer")
    fake_boto3_transfer.TransferConfig = FakeTransferConfig
    fake_boto3.s3 = fake_boto3_s3
    fake_boto3_s3.transfer = fake_boto3_transfer

    fake_botocore = types.ModuleType("botocore")
    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None

    fake_paramiko = types.ModuleType("paramiko")
    fake_scp = types.ModuleType("scp")
    fake_scp.SCPClient = object
    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda *args, **kwargs: None
    fake_requests.post = lambda *args, **kwargs: None

    modules = {
        "fastapi": fake_fastapi,
        "fastapi.middleware.cors": fake_cors,
        "fastapi.staticfiles": fake_staticfiles,
        "fastapi.responses": fake_responses,
        "pydantic": fake_pydantic,
        "boto3": fake_boto3,
        "boto3.s3": fake_boto3_s3,
        "boto3.s3.transfer": fake_boto3_transfer,
        "botocore": fake_botocore,
        "botocore.config": fake_botocore_config,
        "dotenv": fake_dotenv,
        "paramiko": fake_paramiko,
        "scp": fake_scp,
        "requests": fake_requests,
    }

    env = {
        "S3_MULTIPART_THRESHOLD": str(64 * 1024 * 1024),
        "S3_MULTIPART_CHUNKSIZE": str(64 * 1024 * 1024),
        "S3_MAX_CONCURRENCY": "4",
    }

    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch.dict(os.environ, env),
        mock.patch("sys.platform", "linux"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        return importlib.import_module("server")


class S3UploadConfigTests(unittest.TestCase):
    def test_large_uploads_use_multipart_below_700mb(self):
        server = load_server_module()

        config = server.build_s3_transfer_config()

        self.assertLess(config.kwargs["multipart_threshold"], 700 * 1024 * 1024)
        self.assertEqual(config.kwargs["multipart_chunksize"], 64 * 1024 * 1024)
        self.assertEqual(config.kwargs["max_concurrency"], 4)
        self.assertTrue(config.kwargs["use_threads"])

    def test_cloud_processing_requires_completed_s3_upload(self):
        server = load_server_module()

        s3_key, response = server.require_uploaded_s3_key({"status": "uploading"})

        self.assertIsNone(s3_key)
        self.assertEqual(response["kwargs"]["status_code"], 409)
        self.assertIn("not finished uploading", response["kwargs"]["content"]["error"])

    def test_cloud_processing_uses_uploaded_s3_key(self):
        server = load_server_module()

        s3_key, response = server.require_uploaded_s3_key(
            {"status": "uploaded", "s3_key": "abc123.m4a"}
        )

        self.assertIsNone(response)
        self.assertEqual(s3_key, "transcriber/uploads/abc123.m4a")

    def test_cloud_processing_keeps_prefixed_s3_key(self):
        server = load_server_module()

        s3_key, response = server.require_uploaded_s3_key(
            {"status": "uploaded", "s3_key": "transcriber/uploads/abc123.m4a"}
        )

        self.assertIsNone(response)
        self.assertEqual(s3_key, "transcriber/uploads/abc123.m4a")

    def test_runpod_completed_output_error_is_detected(self):
        server = load_server_module()

        self.assertEqual(
            server.get_runpod_output_error({"error": "torchcodec failed"}),
            "torchcodec failed",
        )
        self.assertIsNone(server.get_runpod_output_error({"result": []}))

    def test_progress_s3_key_uses_uploaded_audio_key(self):
        server = load_server_module()

        self.assertEqual(
            server.get_progress_s3_key({"s3_key": "transcriber/uploads/abc123.m4a"}),
            "transcriber/progress/abc123.progress.json",
        )

    def test_dockerfile_pins_torchcodec_for_torch_2_8(self):
        dockerfile = Path("Dockerfile.base").read_text(encoding="utf-8")

        self.assertIn('"torch==2.8.0+cu128"', dockerfile)
        self.assertIn('"torchcodec==0.7.0"', dockerfile)


if __name__ == "__main__":
    unittest.main()
