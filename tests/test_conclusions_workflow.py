import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

from conclusions_workflow import ConclusionsWorkflowManager, extract_document_text


class ParallelFakeCodexRunner:
    def __init__(self, expected=2):
        self.expected = expected
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.all_started = threading.Event()

    def run(self, run_dir, prompt, schema):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= self.expected:
                self.all_started.set()
        self.all_started.wait(timeout=2)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        filename = "one.txt" if "one.txt" in prompt else "two.txt"
        return {"title": "Выводы", "content": f"# Выводы\n\nИсходный файл: {filename}\n\nТестовый результат."}


class ConclusionsWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runner = ParallelFakeCodexRunner()
        self.manager = ConclusionsWorkflowManager(self.root, codex_runner=self.runner)

    def tearDown(self):
        self.temporary.cleanup()

    def wait_for(self, task_ids):
        deadline = time.time() + 4
        while time.time() < deadline:
            tasks = [self.manager.get(task_id) for task_id in task_ids]
            if all(task["status"] not in {"queued", "running"} for task in tasks):
                return tasks
            time.sleep(0.01)
        self.fail("Conclusion jobs did not finish")

    def test_files_run_in_parallel_and_source_is_not_modified(self):
        source_one = "Операторская версия один."
        source_two = "Операторская версия два."
        first = self.manager.create("one.txt", source_one.encode(), "Сделай выводы")
        second = self.manager.create("two.txt", source_two.encode(), "Сделай выводы")
        tasks = self.wait_for([first["id"], second["id"]])

        self.assertEqual([task["status"] for task in tasks], ["completed", "completed"])
        self.assertEqual(self.runner.max_active, 2)
        self.assertEqual((self.root / first["id"] / "source.txt").read_text(), source_one)
        self.assertEqual((self.root / second["id"] / "source.txt").read_text(), source_two)

    def test_txt_and_docx_have_the_same_content_and_survive_reload(self):
        self.runner.expected = 1
        task = self.manager.create("focus-group.txt", "Текст группы".encode(), "Сделай выводы")
        self.wait_for([task["id"]])
        txt_path, _ = self.manager.result_path(task["id"], "txt")
        docx_path, _ = self.manager.result_path(task["id"], "docx")

        self.assertTrue(txt_path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertIn("Тестовый результат", txt_path.read_text(encoding="utf-8-sig"))
        self.assertTrue(zipfile.is_zipfile(docx_path))
        self.assertIn("Тестовый результат", extract_document_text(docx_path))

        restored = ConclusionsWorkflowManager(self.root, codex_runner=self.runner)
        self.assertEqual(restored.get(task["id"])["status"], "completed")
        self.assertEqual(restored.result_path(task["id"], "docx")[0], docx_path)


if __name__ == "__main__":
    unittest.main()
