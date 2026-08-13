import asyncio
import contextlib
import importlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def upload_file(self, *args, **kwargs):
        return None

    def put_object(self, *args, **kwargs):
        return None


def load_server_module():
    sys.modules.pop("server", None)

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.FastAPI = lambda *args, **kwargs: FakeApp()
    fake_fastapi.UploadFile = object
    fake_fastapi.File = lambda *args, **kwargs: None
    fake_fastapi.Form = lambda *args, **kwargs: None
    fake_fastapi.Body = lambda *args, **kwargs: None
    fake_fastapi.Request = object

    fake_cors = types.ModuleType("fastapi.middleware.cors")
    fake_cors.CORSMiddleware = object

    fake_staticfiles = types.ModuleType("fastapi.staticfiles")
    fake_staticfiles.StaticFiles = lambda *args, **kwargs: object()

    fake_responses = types.ModuleType("fastapi.responses")
    fake_responses.FileResponse = object
    fake_responses.JSONResponse = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
    fake_responses.StreamingResponse = object

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
    def test_delete_all_s3_files_preserves_local_uploads_and_transcriptions(self):
        server = load_server_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            uploads_dir = Path(temp_dir)
            local_audio = uploads_dir / "focus-group.m4a"
            local_transcript = uploads_dir / "focus-group.json"
            local_audio.write_bytes(b"audio")
            local_transcript.write_text('{"segments": []}', encoding="utf-8")
            server.transcriptions["focus-group.m4a"] = {"status": "completed"}

            with (
                mock.patch.object(server, "UPLOAD_DIR", uploads_dir),
                mock.patch.object(
                    server,
                    "list_s3_bucket_files",
                    return_value=[{"key": "remote-audio.m4a"}, {"key": "remote-transcript.json"}],
                ),
                mock.patch.object(server, "delete_s3_keys", return_value=2) as delete_remote,
            ):
                result = asyncio.run(
                    server.delete_all_s3_files(SimpleNamespace(confirm="DELETE ALL"))
                )

            delete_remote.assert_called_once_with(["remote-audio.m4a", "remote-transcript.json"])
            self.assertEqual(result["deleted_remote"], 2)
            self.assertNotIn("deleted_local", result)
            self.assertTrue(local_audio.exists())
            self.assertTrue(local_transcript.exists())
            self.assertIn("focus-group.m4a", server.transcriptions)

    def test_s3_library_record_uses_original_task_name_and_transcript(self):
        server = load_server_module()
        server.transcriptions["Original interview.m4a"] = {
            "filename": "Original interview.m4a",
            "status": "completed",
            "progress": 100,
            "result": [{"text": "done"}],
            "s3_key": "technical-key.m4a",
        }

        record = server.describe_s3_file({
            "key": "transcriber/uploads/technical-key.m4a",
            "name": "technical-key.m4a",
            "size": 123,
            "last_modified": None,
        })

        self.assertEqual(record["name"], "Original interview.m4a")
        self.assertEqual(record["storage_name"], "technical-key.m4a")
        self.assertEqual(record["task_id"], "Original interview.m4a")
        self.assertEqual(record["transcript_name"], "Original interview.json")
        self.assertEqual(
            record["transcript_s3_key"],
            "transcriber/results/technical-key.json",
        )
        self.assertTrue(record["has_transcript"])
        self.assertEqual(record["transcription"]["state"], "completed")
        self.assertEqual(record["transcription"]["progress"], 100)
        self.assertEqual(record["normalization"]["state"], "not_started")

    def test_library_transcription_summary_exposes_live_runpod_progress(self):
        server = load_server_module()

        summary = server.summarize_transcription_state({
            "status": "processing",
            "progress": 46,
            "runpod_progress_message": "Cloud transcription: 3:12 elapsed",
        })

        self.assertEqual(summary["state"], "running")
        self.assertEqual(summary["label"], "Диаризация и транскрибация")
        self.assertEqual(summary["progress"], 46)
        self.assertEqual(summary["message"], "Cloud transcription: 3:12 elapsed")
        self.assertTrue(summary["active"])

    def test_library_transcription_summary_reports_error_and_stops_polling(self):
        server = load_server_module()

        summary = server.summarize_transcription_state({
            "status": "error",
            "progress": 71,
            "error": "RunPod job failed",
        })

        self.assertEqual(summary["state"], "failed")
        self.assertEqual(summary["message"], "RunPod job failed")
        self.assertFalse(summary["active"])

    def test_completed_transcription_does_not_show_stale_runpod_waiting_message(self):
        server = load_server_module()

        summary = server.summarize_transcription_state({
            "status": "completed",
            "progress": 100,
            "runpod_progress_message": "Cloud transcription: waiting for RunPod result",
        })

        self.assertEqual(summary["state"], "completed")
        self.assertEqual(summary["message"], "Транскрипт доступен для просмотра и нормализации.")

    def test_library_normalization_summary_names_the_blocked_step(self):
        server = load_server_module()
        workflow = {
            "overall_progress": 30,
            "steps": [
                {"id": "source", "title": "Источник", "index": 0, "status": "completed"},
                {"id": "structure", "title": "Роли и реплики", "index": 1, "status": "completed"},
                {"id": "chunks", "title": "Чанки", "index": 2, "status": "completed"},
                {"id": "terms", "title": "Термины", "index": 3, "status": "needs_review", "gate": {"summary": "Нужно разобрать 33 решения."}},
                {"id": "language", "title": "Язык", "index": 4, "status": "locked"},
            ],
        }

        with mock.patch.object(server.normalization_manager, "get", return_value=workflow):
            summary = server.summarize_normalization_state("focus-group.m4a", True)

        self.assertEqual(summary["state"], "blocked")
        self.assertEqual(summary["current_step"]["id"], "terms")
        self.assertEqual(summary["blocked_step"]["number"], 4)
        self.assertEqual(summary["blocked_step"]["title"], "Термины")
        self.assertEqual(summary["message"], "Нужно разобрать 33 решения.")
        self.assertEqual(summary["overall_progress"], 30)

    def test_library_normalization_is_disabled_without_transcript(self):
        server = load_server_module()

        summary = server.summarize_normalization_state("audio-only.m4a", False)

        self.assertEqual(summary["state"], "unavailable")
        self.assertFalse(summary["can_normalize"])

    def test_successful_s3_upload_persists_filename_mapping_immediately(self):
        server = load_server_module()
        task_id = "Original interview.m4a"
        server.transcriptions[task_id] = {
            "filename": task_id,
            "status": "uploading",
            "progress": 0,
            "result": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / task_id
            audio_path.write_bytes(b"audio")
            with mock.patch.object(server, "persist_task_json") as persist:
                server.upload_to_s3(audio_path, task_id)

        self.assertEqual(server.transcriptions[task_id]["status"], "uploaded")
        self.assertTrue(server.transcriptions[task_id]["s3_key"].endswith(".m4a"))
        persist.assert_called_once_with(task_id)

    def test_existing_s3_audio_is_registered_without_reupload(self):
        server = load_server_module()

        task_id, task = server.register_existing_s3_audio(
            "transcriber/uploads/existing-recording.m4a"
        )

        self.assertEqual(task_id, "existing-recording.m4a")
        self.assertEqual(task["filename"], "existing-recording.m4a")
        self.assertEqual(task["status"], "uploaded")
        self.assertEqual(task["progress"], 100)
        self.assertEqual(
            task["s3_key"],
            "transcriber/uploads/existing-recording.m4a",
        )
        self.assertIs(server.transcriptions[task_id], task)

    def test_existing_s3_registration_rejects_non_audio_objects(self):
        server = load_server_module()

        with self.assertRaises(ValueError):
            server.register_existing_s3_audio("transcriber/results/result.json")

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
