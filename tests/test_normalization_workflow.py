import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from normalization_workflow import CodexRunner, NormalizationWorkflowManager, _is_lexically_faithful


class FakeCodexRunner:
    def available(self):
        return True, "codex-cli test"

    def run(self, run_dir, prompt, schema):
        parts = Path(run_dir).parts
        if "review" in parts:
            result = {
                "verdict": "pass",
                "summary": "Артефакт готов к следующему этапу.",
                "findings": [],
                "assumptions": [],
                "term_decisions": [],
            }
            if "source" in parts:
                result["transcript_context"] = (
                    "ФГ о медицинской практике: обсуждают опыт и критерии выбора; "
                    "модератор выясняет привычки, барьеры и сценарии решений."
                )
            return result
        if "structure" in parts:
            return {
                "speakers": [
                    {"source_id": "SPEAKER_00", "role": "Интервьюер", "name": "Анна", "confidence": "safe"},
                    {"source_id": "SPEAKER_01", "role": "Респондент", "name": "Артём", "confidence": "safe"},
                ],
                "notes": [],
            }
        if "terms" in parts:
            return {"terms": []}
        if "language" in parts:
            return {
                "changes": [
                    {
                        "turn_id": "t00002",
                        "text": "Я думаю... Это хорошо. (неразборчиво, 00:04)",
                        "reason": "Нормализована пунктуация.",
                        "confidence": "safe",
                    }
                ]
            }
        if "fidelity" in parts:
            return {"issues": []}
        raise AssertionError(f"Unexpected run directory: {run_dir}")


class AmbiguousCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.diarization_calls = 0

    def run(self, run_dir, prompt, schema):
        if Path(run_dir).name.startswith("diarization-"):
            self.diarization_calls += 1
            return {"assignments": [{
                "turn_id": "t00002",
                "speaker_source_id": "SPEAKER_00",
                "confidence": "safe",
                "reason": "Короткий ответ продолжает вопрос предыдущего участника.",
            }]}
        if Path(run_dir).name == "structure":
            return {
                "speakers": [
                    {"source_id": "SPEAKER_00", "role": "Интервьюер", "name": "Анна", "confidence": "safe"},
                    {"source_id": "Unknown", "role": "Респондент", "name": "", "confidence": "low"},
                ],
                "notes": [],
            }
        return super().run(run_dir, prompt, schema)


class LowConfidenceAmbiguousCodexRunner(AmbiguousCodexRunner):
    def run(self, run_dir, prompt, schema):
        if Path(run_dir).name.startswith("diarization-"):
            return {"assignments": [{
                "turn_id": "t00002",
                "speaker_source_id": "SPEAKER_00",
                "confidence": "low",
                "reason": "Контекста мало; выбран наиболее вероятный существующий участник.",
            }]}
        return super().run(run_dir, prompt, schema)


class PendingTermCodexRunner(FakeCodexRunner):
    def run(self, run_dir, prompt, schema):
        parts = Path(run_dir).parts
        if "review" not in parts and "terms" in parts:
            return {"terms": [{
                "turn_id": "t00002",
                "original": "Конкорр",
                "proposed": "Конкор",
                "safety": "mid",
                "reason": "Нужно решение оператора.",
            }]}
        return super().run(run_dir, prompt, schema)


class PromptCaptureCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.prompts = []

    def run(self, run_dir, prompt, schema):
        self.prompts.append((str(run_dir), prompt))
        return super().run(run_dir, prompt, schema)


class ContextualRewriteCodexRunner(FakeCodexRunner):
    def run(self, run_dir, prompt, schema):
        parts = Path(run_dir).parts
        if "language" in parts and "review" not in parts:
            return {"changes": [{
                "turn_id": "t00002",
                "text": "Только кардиолог — таких нет?",
                "reason": "Контекстно восстановлен уточняющий вопрос.",
                "confidence": "safe",
            }]}
        return super().run(run_dir, prompt, schema)


class ResidualAsrCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.adjudication_calls = 0

    def run(self, run_dir, prompt, schema):
        if "language-adjudication" in Path(run_dir).parts:
            self.adjudication_calls += 1
            return {"decisions": [{
                "turn_id": "t00002", "action": "revert", "replacement": "",
                "reason": "Правка medium недостаточно подтверждена текстом.", "confidence": "safe",
            }]}
        return super().run(run_dir, prompt, schema)


class ParallelLanguageCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.lock = threading.Lock()
        self.active_language = 0
        self.max_active_language = 0
        self.active_adjudication = 0
        self.max_active_adjudication = 0

    def _tracked(self, active_name, maximum_name, result):
        with self.lock:
            active = getattr(self, active_name) + 1
            setattr(self, active_name, active)
            setattr(self, maximum_name, max(getattr(self, maximum_name), active))
        time.sleep(0.06)
        with self.lock:
            setattr(self, active_name, getattr(self, active_name) - 1)
        return result

    def run(self, run_dir, prompt, schema):
        path = Path(run_dir)
        if "language-adjudication" in path.parts:
            return self._tracked("active_adjudication", "max_active_adjudication", {"decisions": []})
        if "language" in path.parts and "review" not in path.parts:
            return self._tracked("active_language", "max_active_language", {"changes": []})
        return super().run(run_dir, prompt, schema)


class TargetedLanguageRetryCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.lock = threading.Lock()
        self.language_chunks = []
        self.adjudication_chunks = []

    def run(self, run_dir, prompt, schema):
        path = Path(run_dir)
        if "language-adjudication" in path.parts:
            chunk_id = path.name
            with self.lock:
                self.adjudication_chunks.append(chunk_id)
            return {"decisions": ([{
                "turn_id": "t00002", "action": "revert", "replacement": "",
                "reason": "Смысловая правка не подтверждена.", "confidence": "safe",
            }] if chunk_id == "c01" else [])}
        if "language" in path.parts and "review" not in path.parts:
            with self.lock:
                self.language_chunks.append(path.name)
            payload_text = prompt.rsplit("Реплики: ", 1)[1].split("\n\nЗамечания предыдущего", 1)[0]
            payload = json.loads(payload_text)
            turn = next(item for item in payload if item.get("core"))
            return {"changes": [{
                "turn_id": turn["id"],
                "text": f"{turn['text']}!",
                "reason": "Добавлен знак завершения.",
                "confidence": "safe",
            }]}
        return super().run(run_dir, prompt, schema)


class CompleteTermsReviewCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.lock = threading.Lock()
        self.reviewed_ids = []
        self.coverage_chunks = []
        self.active_reviews = 0
        self.max_active_reviews = 0
        self.active_coverage = 0
        self.max_active_coverage = 0

    def run(self, run_dir, prompt, schema):
        path = Path(run_dir)
        if "terms-coverage" in path.parts:
            with self.lock:
                self.active_coverage += 1
                self.max_active_coverage = max(self.max_active_coverage, self.active_coverage)
                self.coverage_chunks.append(path.name)
            time.sleep(0.03)
            with self.lock:
                self.active_coverage -= 1
            return {"terms": ([{
                "turn_id": "t00384", "original": "Непертен", "proposed": "Нипертен",
                "confidence": "safe", "decision": "accepted", "reason": "Канонический бренд.",
            }] if path.name == "c08" else [])}
        if "terms-decisions" in path.parts:
            candidates = json.loads(prompt.split("Кандидаты: ", 1)[1])
            with self.lock:
                self.active_reviews += 1
                self.max_active_reviews = max(self.max_active_reviews, self.active_reviews)
            time.sleep(0.03)
            with self.lock:
                self.active_reviews -= 1
                self.reviewed_ids.extend(item["id"] for item in candidates)
            return {"term_decisions": [{
                "term_id": item["id"], "decision": "accepted", "reason": "Подтверждено.", "confidence": "safe",
            } for item in candidates]}
        return super().run(run_dir, prompt, schema)


class BatchedLanguageRepairCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.lock = threading.Lock()
        self.targets = []
        self.calls = 0
        self.active = 0
        self.maximum = 0

    def run(self, run_dir, prompt, schema):
        if Path(run_dir).name.startswith("retry-batch-"):
            cases = json.loads(prompt.split("Случаи: ", 1)[1].split("\n\nРеплики:", 1)[0])
            with self.lock:
                self.calls += 1
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                self.targets.extend(case["turn_id"] for case in cases)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return {"changes": [{
                "turn_id": case["turn_id"],
                "text": next(item["text"] for item in case["context"] if item["core"]) + ".",
                "reason": "Адресное исправление.", "confidence": "safe",
            } for case in cases]}
        return super().run(run_dir, prompt, schema)


class LanguageAdjudicationCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.prompts = []

    def run(self, run_dir, prompt, schema):
        if "language-adjudication" in Path(run_dir).parts:
            self.prompts.append(prompt)
            return {"decisions": [
                {
                    "turn_id": "t00001", "action": "replace",
                    "replacement": "Метформин принимаю.",
                    "reason": "Очевидная близкая ASR-ошибка в названии препарата.", "confidence": "safe",
                },
                {
                    "turn_id": "t00002", "action": "revert", "replacement": "",
                    "reason": "Medium добавил отрицание.", "confidence": "safe",
                },
            ]}
        return super().run(run_dir, prompt, schema)


class FidelityRevertCodexRunner(FakeCodexRunner):
    def run(self, run_dir, prompt, schema):
        if "review" in Path(run_dir).parts and "fidelity" in Path(run_dir).parts:
            return {"issues": [{
                "change_id": "change-00001", "severity": "high", "message": "Добавлено отрицание.",
            }]}
        return super().run(run_dir, prompt, schema)


class ContextualRediarizationCodexRunner(FakeCodexRunner):
    def __init__(self, confidence="safe"):
        self.confidence = confidence
        self.rediarization_calls = 0

    def run(self, run_dir, prompt, schema):
        name = Path(run_dir).name
        if name.startswith("rediarization-"):
            self.rediarization_calls += 1
            return {"assignments": [
                {
                    "segment_id": "s00002",
                    "speaker_source_id": "SPEAKER_01",
                    "confidence": self.confidence,
                    "reason": "Приветствие синтаксически продолжает представление респондентки.",
                },
                {
                    "segment_id": "s00004",
                    "speaker_source_id": "SPEAKER_00",
                    "confidence": "mid",
                    "reason": "Благодарность и следующий вопрос принадлежат интервьюеру.",
                },
            ]}
        return super().run(run_dir, prompt, schema)


class ParallelRediarizationCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0
        self.prompts = []

    def run(self, run_dir, prompt, schema):
        if Path(run_dir).name.startswith("rediarization-"):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                self.prompts.append(prompt)
            time.sleep(0.06)
            with self.lock:
                self.active -= 1
            return {"assignments": []}
        return super().run(run_dir, prompt, schema)


class InteriorTurnSplitCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.interior_split_calls = 0

    def run(self, run_dir, prompt, schema):
        name = Path(run_dir).name
        if name.startswith("rediarization-"):
            return {"assignments": [{
                "segment_id": "s00003",
                "speaker_source_id": "SPEAKER_00",
                "confidence": "mid",
                "reason": "Вопрос принадлежит интервьюеру.",
            }]}
        if name.startswith("interior-splits-"):
            self.interior_split_calls += 1
            return {"splits": [{
                "turn_id": "t00003",
                "parts": [
                    {"segment_ids": ["s00003"], "speaker_source_id": "SPEAKER_00"},
                    {"segment_ids": ["s00004", "s00005"], "speaker_source_id": "SPEAKER_01"},
                    {"segment_ids": ["s00006"], "speaker_source_id": "SPEAKER_00"},
                ],
                "confidence": "safe",
                "reason": "Короткий ответ расположен между вопросом и реакцией интервьюера.",
            }]}
        return super().run(run_dir, prompt, schema)


class StructureRepairCodexRunner(FakeCodexRunner):
    def __init__(self):
        self.structure_reviews = 0
        self.remediation_calls = 0

    def run(self, run_dir, prompt, schema):
        path = Path(run_dir)
        if "review" in path.parts and "structure" in path.parts:
            self.structure_reviews += 1
            if self.structure_reviews == 1:
                return {
                    "verdict": "fail",
                    "summary": "Назначение противоречит собственному обоснованию.",
                    "findings": [{
                        "severity": "high",
                        "code": "speaker_assignment_contradicts_reason",
                        "message": "s00002 продолжает реплику интервьюера.",
                        "item_id": "s00002",
                    }],
                    "assumptions": [],
                    "term_decisions": [],
                }
            return {
                "verdict": "pass",
                "summary": "Противоречие исправлено.",
                "findings": [],
                "assumptions": [],
                "term_decisions": [],
            }
        if path.name.startswith("rediarization-"):
            return {"assignments": []}
        if path.name.startswith("remediation-"):
            self.remediation_calls += 1
            return {"assignments": [{
                "turn_id": "t00002",
                "speaker_source_id": "SPEAKER_00",
                "confidence": "safe",
                "reason": "Фрагмент синтаксически продолжает реплику интервьюера.",
            }]}
        return super().run(run_dir, prompt, schema)


class NormalizationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.uploads = []

        def upload(task_id, path, checksum):
            self.uploads.append((task_id, path.read_text(encoding="utf-8"), checksum))
            return {"status": "uploaded", "key": "transcriber/final/interview_normalized.md", "filename": "interview_normalized.md", "sha256": checksum}

        self.manager = NormalizationWorkflowManager(
            Path(self.temporary.name),
            upload_callback=upload,
            codex_runner=FakeCodexRunner(),
            auto_advance=False,
        )
        self.task_id = "interview.m4a"
        self.task = {
            "filename": self.task_id,
            "status": "completed",
            "result": [
                {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Добрый день."},
                {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01", "text": "Я думаю... Это хорошо. [пауза]"},
            ],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def run_step(self, step_id):
        self.manager.start(self.task_id, self.task, step_id)
        deadline = time.time() + 5
        while time.time() < deadline:
            state = self.manager.get(self.task_id)
            step = next(item for item in state["steps"] if item["id"] == step_id)
            if step["status"] not in {"queued", "running", "reviewing"}:
                self.assertEqual(step["status"], "completed", step.get("error"))
                return state
            time.sleep(0.01)
        self.fail(f"Step {step_id} did not finish")

    def test_complete_workflow_renders_gold_format_and_uploads_exact_file(self):
        initial = self.manager.ensure(self.task_id, self.task)
        self.assertEqual(initial["codex"]["editor"], {
            "available": True,
            "model": "gpt-5.6-sol",
            "effort": "medium",
        })
        self.assertEqual(initial["codex"]["diarization"], {
            "available": True,
            "model": "gpt-5.6-sol",
            "effort": "medium",
        })
        self.assertEqual(initial["codex"]["reviewer"], {
            "available": True,
            "model": "gpt-5.6-sol",
            "effort": "xhigh",
        })
        for step in ("source", "structure", "chunks", "terms", "language", "fidelity", "assemble"):
            self.run_step(step)
        self.run_step("approve")
        self.run_step("render")
        self.run_step("upload")

        markdown = self.manager.final_path(self.task_id).read_text(encoding="utf-8")
        self.assertIn("# Transcription: interview.m4a", markdown)
        self.assertIn("**[00:00] Интервьюер (Анна): Добрый день.**", markdown)
        self.assertIn("**[00:03] Респондент 1 (Артём):** Я думаю… Это хорошо. (неразборчиво)", markdown)
        self.assertNotIn("...", markdown)
        self.assertNotIn("[пауза]", markdown)
        self.assertEqual(self.uploads[0][1], markdown)
        receipt = self.manager.artifact(self.task_id, "upload")
        self.assertEqual(receipt["handoff_owner"], "Sol xhigh")
        self.assertEqual(receipt["assumption_count"], 0)

    def test_default_runners_use_sol_medium_but_keep_reviewer_xhigh(self):
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "runner-efforts",
            auto_advance=False,
        )

        self.assertEqual(manager.codex_worker.model, "gpt-5.6-sol")
        self.assertEqual(manager.codex_worker.reasoning_effort, "medium")
        self.assertEqual(manager.codex_worker.web_search_mode, "disabled")
        self.assertEqual(manager.codex_diarization.model, "gpt-5.6-sol")
        self.assertEqual(manager.codex_diarization.reasoning_effort, "medium")
        self.assertEqual(manager.codex_diarization.web_search_mode, "disabled")
        self.assertEqual(manager.codex_reviewer.model, "gpt-5.6-sol")
        self.assertEqual(manager.codex_reviewer.reasoning_effort, "xhigh")
        self.assertEqual(manager.codex_reviewer.web_search_mode, "cached")

    def test_codex_runner_retries_timeout_and_discards_partial_output(self):
        run_dir = Path(self.temporary.name) / "timeout-retry"
        output_path = run_dir / "codex-output.json"
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            self.assertFalse(output_path.exists())
            if len(calls) == 1:
                output_path.write_text('{"partial": true}', encoding="utf-8")
                raise subprocess.TimeoutExpired(command, 1)
            output_path.write_text('{"terms": []}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        runner = CodexRunner(command="/mock/codex", timeout_seconds=1, timeout_attempts=2)
        with patch("normalization_workflow.subprocess.run", side_effect=fake_run):
            result = runner.run(run_dir, "prompt", {"type": "object"})

        self.assertEqual(result, {"terms": []})
        self.assertEqual(len(calls), 2)
        self.assertTrue((run_dir / "codex-timeout-attempt-1.log").exists())

    def test_codex_runner_reports_exhausted_timeout_without_raw_command(self):
        run_dir = Path(self.temporary.name) / "timeout-failure"
        runner = CodexRunner(command="/mock/codex", timeout_seconds=1, timeout_attempts=2)

        with patch(
            "normalization_workflow.subprocess.run",
            side_effect=lambda command, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(command, 1)),
        ):
            with self.assertRaisesRegex(RuntimeError, "Codex не ответил за 1 с; выполнено попыток: 2") as raised:
                runner.run(run_dir, "prompt", {"type": "object"})

        self.assertNotIn("Command '[", str(raised.exception))

    def test_lexical_guard_rejects_contextual_reconstruction(self):
        self.assertFalse(_is_lexically_faithful(
            "Ну да, где-то процентов 80 сам, процентов 20 кардиолог с кардиологом.",
            "Ну да, где-то процентов 80 сам, процентов 20 — совместно с кардиологом.",
            [],
        ))
        self.assertFalse(_is_lexically_faithful(
            "Только кардиолог таких не слышал.",
            "Только кардиолог — таких нет?",
            [],
        ))

    def test_lexical_guard_allows_punctuation_spelling_and_approved_terms(self):
        self.assertTrue(_is_lexically_faithful("Это харошая мысль", "Это хорошая мысль!", []))
        self.assertTrue(_is_lexically_faithful(
            "Назначили Конкорр вчера.",
            "Назначили Конкор вчера.",
            [{"original": "Конкорр", "proposed": "Конкор"}],
        ))

    def test_language_stage_does_not_apply_contextual_word_rewrite(self):
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "literal-language",
            codex_runner=ContextualRewriteCodexRunner(),
            auto_advance=False,
        )
        task_id = "literal.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "Уточните."},
                {"start": 2, "end": 5, "speaker": "SPEAKER_01", "text": "Только кардиолог таких не слышал."},
            ],
        }
        manager.ensure(task_id, task)
        for step_id in ("source", "structure", "chunks", "terms", "language"):
            manager.start(task_id, task, step_id)
            self._wait_for_manager_step(manager, task_id, step_id)
        artifact = json.loads(manager._artifact_path(task_id, "language-changes.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact["changes"], [])
        self.assertEqual(artifact["rejected_lexical_rewrites"][0]["turn_id"], "t00002")

    def test_rerunning_earlier_step_marks_downstream_as_stale(self):
        self.manager.ensure(self.task_id, self.task)
        self.run_step("source")
        self.run_step("structure")
        self.run_step("chunks")
        self.run_step("terms")
        self.run_step("language")
        self.run_step("fidelity")
        self.run_step("assemble")
        self.manager.approve(self.task_id)
        self.run_step("render")

        self.run_step("chunks")
        state = self.manager.get(self.task_id)
        statuses = {step["id"]: step["status"] for step in state["steps"]}
        self.assertEqual(statuses["chunks"], "completed")
        self.assertEqual(statuses["terms"], "ready")
        self.assertEqual(statuses["language"], "stale")
        self.assertEqual(statuses["render"], "stale")
        chunks_step = next(step for step in state["steps"] if step["id"] == "chunks")
        self.assertEqual(chunks_step["history"][0]["files"], ["chunks.json"])

    def test_operator_can_override_speaker_for_one_turn(self):
        self.manager.ensure(self.task_id, self.task)
        self.run_step("source")
        self.run_step("structure")
        speakers = [
            {"source_id": "SPEAKER_00", "role": "Интервьюер", "name": "Анна"},
            {"source_id": "SPEAKER_01", "role": "Респондент", "name": "Артём"},
        ]
        self.manager.update_registry(
            self.task_id,
            speakers,
            [{"turn_id": "t00002", "source_id": "SPEAKER_00"}],
        )
        deadline = time.time() + 5
        while time.time() < deadline:
            structure = next(step for step in self.manager.get(self.task_id)["steps"] if step["id"] == "structure")
            if structure["status"] not in {"reviewing", "queued", "running"}:
                self.assertEqual(structure["status"], "completed", structure.get("error"))
                break
            time.sleep(0.01)
        turns = self.manager.artifact(self.task_id, "structure")["turns"]
        self.assertEqual(turns[1]["speaker"]["role"], "Интервьюер")
        self.assertEqual(turns[1]["source_speaker"], "SPEAKER_01")

    def test_codex_automatically_repairs_safe_unknown_turn(self):
        registry_runner = AmbiguousCodexRunner()
        diarization_runner = AmbiguousCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "ambiguous",
            codex_runner=registry_runner,
            diarization_runner=diarization_runner,
            auto_advance=False,
        )
        task_id = "ambiguous.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "Вы согласны?"},
                {"start": 2.1, "end": 2.6, "speaker": "Unknown", "text": "Да."},
            ],
        }
        manager.ensure(task_id, task)

        def run(step_id):
            manager.start(task_id, task, step_id)
            deadline = time.time() + 5
            while time.time() < deadline:
                step = next(item for item in manager.get(task_id)["steps"] if item["id"] == step_id)
                if step["status"] not in {"queued", "running", "reviewing"}:
                    self.assertEqual(step["status"], "completed", step.get("error"))
                    return step
                time.sleep(0.01)
            self.fail(f"Step {step_id} did not finish")

        run("source")
        structure = run("structure")
        self.assertEqual(structure["details"]["detected_defect_count"], 1)
        self.assertEqual(structure["details"]["auto_fixed_count"], 1)
        self.assertEqual(structure["details"]["review_turn_count"], 0)
        turns = manager.artifact(task_id, "structure")["turns"]
        self.assertEqual(turns[1]["speaker"]["name"], "Анна")
        self.assertEqual(turns[1]["source_speaker"], "Unknown")
        self.assertEqual(registry_runner.diarization_calls, 0)
        self.assertEqual(diarization_runner.diarization_calls, 1)
        self.assertNotIn("Unknown", {item["source_id"] for item in structure["details"]["speakers"]})
        registry = manager._artifact_path(task_id, "speaker-registry.json")
        registry_data = __import__("json").loads(registry.read_text(encoding="utf-8"))
        self.assertNotIn("Unknown", {item["source_id"] for item in registry_data["speakers"]})

    def test_low_confidence_unknown_is_assigned_and_logged_without_pausing(self):
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "low-confidence",
            codex_runner=LowConfidenceAmbiguousCodexRunner(),
            auto_advance=True,
        )
        task_id = "low-confidence.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "Продолжайте."},
                {"start": 2.1, "end": 2.6, "speaker": "Unknown", "text": "Да."},
            ],
        }
        manager.ensure(task_id, task)
        manager.start(task_id, task, "source")
        deadline = time.time() + 8
        while time.time() < deadline:
            state = manager.get(task_id)
            structure = next(item for item in state["steps"] if item["id"] == "structure")
            upload = next(item for item in state["steps"] if item["id"] == "upload")
            if upload["status"] == "ready":
                break
            failed = [step for step in state["steps"] if step["status"] == "failed"]
            self.assertFalse(failed, failed)
            time.sleep(0.02)
        else:
            self.fail("Low-confidence assignment did not complete automatically")

        self.assertEqual(structure["details"]["auto_fixed_count"], 1)
        self.assertEqual(structure["details"]["review_turn_count"], 1)
        self.assertEqual(structure["details"]["review_turns"][0]["selected_source_id"], "SPEAKER_00")
        self.assertEqual(structure["details"]["unknown_speaker_count"], 0)
        self.assertNotIn("Unknown", {item["source_id"] for item in structure["details"]["speakers"]})
        turns = manager.artifact(task_id, "structure")["turns"]
        self.assertEqual(turns[1]["speaker"]["source_id"], "SPEAKER_00")
        assumptions = manager.get(task_id)["assumptions"]
        decision = next(item for item in assumptions if item["item_id"] == "t00002")
        self.assertEqual(decision["owner"], "Sol xhigh")
        self.assertEqual(decision["confidence"], "low")

    def test_result_receipt_carries_xhigh_assumptions_for_operator(self):
        uploads = []

        def upload(_task_id, _path, checksum):
            uploads.append(checksum)
            return {
                "status": "uploaded",
                "key": "transcriber/final/low-confidence_normalized.md",
                "filename": "low-confidence_normalized.md",
                "sha256": checksum,
            }

        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "assumption-handoff",
            upload_callback=upload,
            codex_runner=LowConfidenceAmbiguousCodexRunner(),
            auto_advance=True,
        )
        task_id = "handoff.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "Продолжайте."},
                {"start": 2.1, "end": 2.6, "speaker": "Unknown", "text": "Да."},
            ],
        }
        manager.ensure(task_id, task)
        manager.start(task_id, task, "source")
        deadline = time.time() + 8
        while time.time() < deadline:
            state = manager.get(task_id)
            upload_step = next(item for item in state["steps"] if item["id"] == "upload")
            if upload_step["status"] == "completed":
                break
            failed = [step for step in state["steps"] if step["status"] == "failed"]
            self.assertFalse(failed, failed)
            time.sleep(0.02)
        else:
            self.fail("Automatic flow did not upload the final result")

        receipt = manager.artifact(task_id, "upload")
        self.assertTrue(uploads)
        self.assertEqual(receipt["handoff_owner"], "Sol xhigh")
        self.assertGreaterEqual(receipt["assumption_count"], 1)
        self.assertTrue(any(item["category"] == "speaker_assignment" for item in receipt["assumptions"]))

    def test_schema_three_unknown_registry_is_migrated_without_new_participant(self):
        import json

        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "migration",
            codex_runner=AmbiguousCodexRunner(),
            auto_advance=False,
        )
        task_id = "migration.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "Вы согласны?"},
                {"start": 2.1, "end": 2.6, "speaker": "Unknown", "text": "Да."},
            ],
        }
        manager.ensure(task_id, task)
        for step_id in ("source", "structure"):
            manager.start(task_id, task, step_id)
            deadline = time.time() + 5
            while time.time() < deadline:
                step = next(item for item in manager.get(task_id)["steps"] if item["id"] == step_id)
                if step["status"] not in {"queued", "running", "reviewing"}:
                    self.assertEqual(step["status"], "completed", step.get("error"))
                    break
                time.sleep(0.01)

        registry_path = manager._artifact_path(task_id, "speaker-registry.json")
        turns_path = manager._artifact_path(task_id, "turns.json")
        state_path = manager._state_path(task_id)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry["speakers"].append({"source_id": "Unknown", "role": "Респондент", "name": "", "number": 1, "confidence": "low"})
        turns = json.loads(turns_path.read_text(encoding="utf-8"))
        turns["turns"][1]["speaker"] = registry["speakers"][-1]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["schema_version"] = 3
        registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
        turns_path.write_text(json.dumps(turns, ensure_ascii=False), encoding="utf-8")
        state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

        migrated = manager.ensure(task_id, task)
        self.assertEqual(migrated["schema_version"], 16)
        self.assertEqual(migrated["codex"]["diarization"]["effort"], "medium")
        structure = next(item for item in migrated["steps"] if item["id"] == "structure")
        self.assertEqual(structure["status"], "ready")
        migrated_registry = json.loads(registry_path.read_text(encoding="utf-8"))
        migrated_turns = json.loads(turns_path.read_text(encoding="utf-8"))
        self.assertNotIn("Unknown", {item["source_id"] for item in migrated_registry["speakers"]})
        self.assertEqual(migrated_turns["turns"][1]["speaker"]["source_id"], "SPEAKER_00")

    def test_term_review_details_count_pending_and_xhigh_flagged_items(self):
        details = {
            "items": [
                {"id": "term-0001", "decision": "accepted"},
                {"id": "term-0002", "decision": "pending"},
                {"id": "term-0003", "decision": "accepted"},
            ],
            "pending": 1,
        }
        gate = {
            "reviewed_at": 123,
            "findings": [
                {"item_id": "term-0001", "code": "SAFETY_OVERSTATED"},
                {"item_id": "terms", "code": "terms_blocker"},
            ],
        }

        self.manager._refresh_term_review_details(details, gate)
        self.assertEqual(details["reviewer_flagged_ids"], ["term-0001"])
        self.assertEqual(details["reviewer_pending"], 1)
        self.assertEqual(details["action_required"], 2)

        details["items"][0]["operator_reviewed_gate_at"] = 123
        details["items"][1]["decision"] = "rejected"
        self.manager._refresh_term_review_details(details, gate)
        self.assertEqual(details["reviewer_pending"], 0)
        self.assertEqual(details["action_required"], 0)

    def test_passed_xhigh_gate_automatically_reaches_result(self):
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "automatic",
            codex_runner=FakeCodexRunner(),
            auto_advance=True,
        )
        manager.ensure(self.task_id, self.task)
        manager.start(self.task_id, self.task, "source")
        deadline = time.time() + 8
        while time.time() < deadline:
            state = manager.get(self.task_id)
            upload = next(step for step in state["steps"] if step["id"] == "upload")
            if upload["status"] == "ready":
                break
            failed = [step for step in state["steps"] if step["status"] == "failed"]
            self.assertFalse(failed, failed)
            time.sleep(0.02)
        else:
            self.fail("Automatic chain did not reach the result step")

        for step in state["steps"][:9]:
            self.assertEqual(step["status"], "completed")
            self.assertEqual(step["gate"]["verdict"], "pass")
            self.assertEqual(step["gate"]["effort"], "xhigh")
        self.assertEqual(upload["status"], "ready")

    def test_xhigh_resolves_pending_term_and_resumes_chain(self):
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "term-gate",
            codex_runner=PendingTermCodexRunner(),
            auto_advance=True,
        )
        manager.ensure(self.task_id, self.task)
        manager.start(self.task_id, self.task, "source")
        deadline = time.time() + 8
        while time.time() < deadline:
            state = manager.get(self.task_id)
            upload = next(step for step in state["steps"] if step["id"] == "upload")
            if upload["status"] == "ready":
                break
            failed = [step for step in state["steps"] if step["status"] == "failed"]
            self.assertFalse(failed, failed)
            time.sleep(0.02)
        else:
            self.fail("Sol xhigh did not resolve the term automatically")
        terms = next(step for step in state["steps"] if step["id"] == "terms")
        self.assertEqual(terms["gate"]["verdict"], "pass")
        self.assertEqual(terms["details"]["action_required"], 0)
        self.assertEqual(terms["details"]["items"][0]["decision"], "rejected")
        assumption = next(item for item in state["assumptions"] if item["category"] == "term_decision")
        self.assertEqual(assumption["owner"], "Sol xhigh")
        self.assertIn("Сохранено исходное", assumption["decision"])

    def test_operator_can_correct_model_term_before_accepting(self):
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "custom-term",
            codex_runner=PendingTermCodexRunner(),
            auto_advance=False,
        )
        manager.ensure(self.task_id, self.task)
        for step_id in ("source", "structure", "chunks", "terms"):
            manager.start(self.task_id, self.task, step_id)
            self._wait_for_manager_step(manager, self.task_id, step_id)

        terms = next(step for step in manager.get(self.task_id)["steps"] if step["id"] == "terms")
        term = terms["details"]["items"][0]
        manager.decide_term(self.task_id, term["id"], "accepted", "уточнённый термин")
        artifact = json.loads((manager._artifact_path(self.task_id, "terms.json")).read_text(encoding="utf-8"))
        saved = artifact["terms"][0]
        self.assertEqual(saved["proposed"], "уточнённый термин")
        self.assertEqual(saved["model_proposed"], term["proposed"])
        self.assertTrue(saved["operator_edited"])
        deadline = time.time() + 8
        while time.time() < deadline:
            state = manager.get(self.task_id)
            terms = next(step for step in state["steps"] if step["id"] == "terms")
            if terms["status"] not in {"queued", "running", "reviewing"}:
                break
            time.sleep(0.02)

    def test_operator_can_delete_false_recognition_fragment(self):
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "delete-term",
            codex_runner=PendingTermCodexRunner(),
            auto_advance=False,
        )
        manager.ensure(self.task_id, self.task)
        for step_id in ("source", "structure", "chunks", "terms"):
            manager.start(self.task_id, self.task, step_id)
            self._wait_for_manager_step(manager, self.task_id, step_id)
        terms = next(step for step in manager.get(self.task_id)["steps"] if step["id"] == "terms")
        term = terms["details"]["items"][0]
        manager.decide_term(self.task_id, term["id"], "accepted", "")
        artifact = json.loads(manager._artifact_path(self.task_id, "terms.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact["terms"][0]["proposed"], "")
        deadline = time.time() + 8
        while time.time() < deadline:
            state = manager.get(self.task_id)
            terms = next(step for step in state["steps"] if step["id"] == "terms")
            if terms["status"] not in {"queued", "running", "reviewing"}:
                break
            time.sleep(0.02)

    def test_transcriber_vocabulary_adds_safe_russian_drug_and_company_candidates(self):
        vocabulary = Path(self.temporary.name) / "vocabulary"
        vocabulary.mkdir()
        (vocabulary / "eaeu_drugs.jsonl").write_text(
            json.dumps({
                "canonical": "Метформин",
                "aliases": ["метформин"],
                "manufacturer": ["Мерк / Merck"],
                "kind": "active_ingredient",
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (vocabulary / "drug_aliases.jsonl").write_text(
            json.dumps({"canonical": "Метформин", "aliases": ["митформин"]}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        runner = PromptCaptureCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "with-vocabulary",
            codex_runner=runner,
            vocabulary_dir=vocabulary,
            auto_advance=False,
        )
        task_id = "drug-talk.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "Что назначали?"},
                {"start": 2, "end": 6, "speaker": "SPEAKER_01", "text": "Митформин от Merck."},
            ],
        }
        manager.ensure(task_id, task)

        def run(step_id):
            manager.start(task_id, task, step_id)
            deadline = time.time() + 5
            while time.time() < deadline:
                step = next(item for item in manager.get(task_id)["steps"] if item["id"] == step_id)
                if step["status"] not in {"queued", "running", "reviewing"}:
                    self.assertEqual(step["status"], "completed", step.get("error"))
                    return step
                time.sleep(0.01)
            self.fail(f"Step {step_id} did not finish")

        for step_id in ("source", "structure", "chunks"):
            run(step_id)
        terms = run("terms")
        artifact = json.loads(manager._artifact_path(task_id, "terms.json").read_text(encoding="utf-8"))
        replacements = {(item["original"], item["proposed"]): item for item in artifact["terms"]}
        self.assertEqual(replacements[("Митформин", "Метформин")]["decision"], "accepted")
        self.assertEqual(replacements[("Merck", "Мерк")]["decision"], "accepted")
        self.assertEqual(replacements[("Митформин", "Метформин")]["source"], "transcriber_dictionary")
        self.assertTrue(terms["details"]["vocabulary"]["available"])
        self.assertEqual(terms["details"]["vocabulary"]["exact_candidates"], 2)
        term_prompt = next(prompt for path, prompt in runner.prompts if "/terms/" in path and "/review/" not in path)
        self.assertIn("Метформин", term_prompt)
        self.assertIn("препараты, термины и компании", term_prompt)

        language = run("language")
        language_prompt = next(prompt for path, prompt in runner.prompts if "/language/" in path and "/review/" not in path)
        self.assertIn("повторный ASR-аудит", language_prompt)
        self.assertIn("Merck", language_prompt)
        self.assertEqual(language["details"]["vocabulary_hints"], 2)

    def test_xhigh_adjudicates_language_without_returning_to_medium(self):
        runner = ResidualAsrCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "coverage-retry",
            codex_runner=runner,
            auto_advance=True,
        )
        manager.ensure(self.task_id, self.task)
        manager.start(self.task_id, self.task, "source")
        deadline = time.time() + 8
        while time.time() < deadline:
            state = manager.get(self.task_id)
            upload = next(step for step in state["steps"] if step["id"] == "upload")
            if upload["status"] == "ready":
                break
            failed = [step for step in state["steps"] if step["status"] == "failed"]
            self.assertFalse(failed, failed)
            time.sleep(0.02)
        else:
            self.fail("Language adjudication did not finish")
        language = next(step for step in state["steps"] if step["id"] == "language")
        self.assertEqual(language["attempt"], 1)
        self.assertEqual(language["status"], "completed")
        self.assertGreaterEqual(runner.adjudication_calls, 1)
        self.assertEqual(language["details"]["retry_batches"], 0)
        self.assertEqual(language["details"]["adjudication_actions"]["revert"], 1)

    def test_language_worker_and_xhigh_adjudication_run_three_chunks_in_parallel(self):
        runner = ParallelLanguageCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "parallel-language",
            codex_runner=runner,
            auto_advance=False,
        )
        task_id = "parallel-language.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {
                    "start": index * 10,
                    "end": index * 10 + 5,
                    "speaker": f"SPEAKER_0{index % 2}",
                    "text": " ".join([f"слово{index}"] * 1100),
                }
                for index in range(6)
            ],
        }
        manager.ensure(task_id, task)
        for step_id in ("source", "structure", "chunks", "terms", "language"):
            manager.start(task_id, task, step_id)
            self._wait_for_manager_step(manager, task_id, step_id)

        chunks = manager.artifact(task_id, "chunks")["chunks"]
        self.assertGreaterEqual(len(chunks), 3)
        self.assertGreaterEqual(runner.max_active_language, 2)
        self.assertLessEqual(runner.max_active_language, 12)
        self.assertGreaterEqual(runner.max_active_adjudication, 2)
        self.assertLessEqual(runner.max_active_adjudication, 12)

    def test_language_adjudication_replaces_retry_without_parent_chunk_rerun(self):
        runner = TargetedLanguageRetryCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "targeted-language-retry",
            codex_runner=runner,
            auto_advance=False,
        )
        task_id = "targeted-language-retry.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {
                    "start": index * 10,
                    "end": index * 10 + 5,
                    "speaker": f"SPEAKER_0{index % 2}",
                    "text": " ".join([f"слово{index}"] * 1100),
                }
                for index in range(4)
            ],
        }
        manager.ensure(task_id, task)
        for step_id in ("source", "structure", "chunks", "terms"):
            manager.start(task_id, task, step_id)
            self._wait_for_manager_step(manager, task_id, step_id)
        manager.start(task_id, task, "language")
        language = self._wait_for_manager_step(manager, task_id, "language")

        self.assertEqual(language["attempt"], 1)
        self.assertEqual(runner.language_chunks.count("c01"), 1)
        self.assertNotIn("retry-batch-01", runner.language_chunks)
        self.assertEqual(runner.adjudication_chunks.count("c01"), 1)
        for chunk_id in set(runner.language_chunks) - {"c01"}:
            self.assertEqual(runner.language_chunks.count(chunk_id), 1)
        for chunk_id in set(runner.adjudication_chunks) - {"c01"}:
            self.assertEqual(runner.adjudication_chunks.count(chunk_id), 1)
        artifact = manager.artifact(task_id, "language")
        self.assertIsNone(language["details"]["retry_mode"])
        self.assertEqual(language["details"]["retry_target_turns"], 0)
        self.assertEqual(language["details"]["retry_batches"], 0)
        self.assertFalse(any(item["turn_id"] == "t00002" for item in artifact["changes"]))

    def test_complete_terms_review_covers_all_candidates_in_parallel_batches(self):
        runner = CompleteTermsReviewCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "complete-terms-review", codex_runner=runner, auto_advance=False,
        )
        task_id = "complete-terms-review.m4a"
        task = dict(self.task, filename=task_id)
        manager.ensure(task_id, task)
        manager._artifact_path(task_id, "turns.json").parent.mkdir(parents=True, exist_ok=True)
        turns = [{"id": f"t{index:05d}", "text": f"термин {index}", "speaker": {}} for index in range(385)]
        chunks = []
        for index in range(8):
            start = index * 48
            end = 385 if index == 7 else (index + 1) * 48
            chunks.append({"id": f"c{index + 1:02d}", "core_ids": [item["id"] for item in turns[start:end]], "context_before": [], "context_after": []})
        manager._artifact_path(task_id, "turns.json").write_text(json.dumps({"turns": turns}), encoding="utf-8")
        manager._artifact_path(task_id, "chunks.json").write_text(json.dumps({"chunks": chunks}), encoding="utf-8")
        candidates = [{
            "id": f"term-{index + 1:04d}", "turn_id": turns[index]["id"], "original": f"ошибка{index}",
            "proposed": f"термин{index}", "safety": ("mid" if index < 58 else "low" if index < 61 else "safe"),
            "reason": "Кандидат producer.", "source": "sol_medium", "decision": "pending", "chunk_id": chunks[min(index // 48, 7)]["id"],
        } for index in range(385)]
        manager._artifact_path(task_id, "terms.json").write_text(json.dumps({"terms": candidates}), encoding="utf-8")

        details = {"items": candidates[:200], "artifact": "terms.json"}
        gate = manager._review_terms_complete(task_id, 1, details)
        saved = json.loads(manager._artifact_path(task_id, "terms.json").read_text(encoding="utf-8"))["terms"]
        self.assertEqual(gate["verdict"], "pass")
        self.assertEqual(len(runner.reviewed_ids), 385)
        self.assertEqual(len(set(runner.reviewed_ids)), 385)
        self.assertEqual(details["review_batches"], 12)
        self.assertLessEqual(runner.max_active_reviews, 12)
        self.assertGreaterEqual(runner.max_active_reviews, 2)
        self.assertEqual(set(runner.coverage_chunks), {f"c{index:02d}" for index in range(1, 9)})
        self.assertLessEqual(runner.max_active_coverage, 12)
        self.assertEqual(details["coverage_added"], 1)
        self.assertEqual(sum(item["original"] == "Непертен" for item in saved), 1)

    def test_language_retry_batches_101_unique_turns_and_preserves_other_changes(self):
        runner = BatchedLanguageRepairCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "batched-language-retry", codex_runner=runner, auto_advance=False,
        )
        task_id = "batched-language-retry.m4a"
        task = dict(self.task, filename=task_id)
        manager.ensure(task_id, task)
        manager._artifact_path(task_id, "turns.json").parent.mkdir(parents=True, exist_ok=True)
        turns = [{"id": f"t{index:05d}", "text": f"реплика {index}", "speaker": {}} for index in range(112)]
        chunks = [{
            "id": f"c{index + 1:02d}", "core_ids": [item["id"] for item in turns[index * 14:(index + 1) * 14]],
            "context_before": [], "context_after": [],
        } for index in range(8)]
        manager._artifact_path(task_id, "turns.json").write_text(json.dumps({"turns": turns}), encoding="utf-8")
        manager._artifact_path(task_id, "chunks.json").write_text(json.dumps({"chunks": chunks}), encoding="utf-8")
        manager._artifact_path(task_id, "terms.json").write_text(json.dumps({"terms": []}), encoding="utf-8")
        previous = [{
            "id": f"change-{index + 1:05d}", "turn_id": turns[index]["id"], "text": f"правка {index}",
            "original": turns[index]["text"], "reason": "Первая попытка.", "confidence": "safe", "guardrail": "ok",
            "chunk_id": chunks[index // 14]["id"],
        } for index in range(112)]
        findings = [
            {"item_id": "change-00001", "code": "protected_gender_change", "message": "Вернуть исходник."},
            {"item_id": "change-00002", "code": "protected_person_change", "message": "Вернуть исходник."},
        ] + [{"item_id": turns[index]["id"], "code": "residual_asr", "message": "Исправить."} for index in range(2, 103)]
        schema = {"type": "object"}

        details = manager._run_language_retry(task_id, chunks, [], [], previous, findings, schema)
        artifact = json.loads(manager._artifact_path(task_id, "language-changes.json").read_text(encoding="utf-8"))["changes"]
        self.assertEqual(details["retry_target_turns"], 101)
        self.assertEqual(details["retry_batches"], 4)
        self.assertEqual(details["deterministic_reverts"], 2)
        self.assertEqual(runner.calls, 4)
        self.assertEqual(len(runner.targets), 101)
        self.assertEqual(len(set(runner.targets)), 101)
        self.assertLessEqual(runner.maximum, 12)
        self.assertEqual(details["preserved_changes"], 9)
        self.assertEqual([item["turn_id"] for item in artifact], sorted((item["turn_id"] for item in artifact)))

    def test_language_xhigh_accepts_reverts_and_replaces_without_medium_retry(self):
        runner = LanguageAdjudicationCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "language-adjudication", codex_runner=runner, auto_advance=False,
        )
        task_id = "language-adjudication.m4a"
        manager.ensure(task_id, dict(self.task, filename=task_id))
        turns = [
            {"id": "t00001", "text": "Митформин принимаю.", "speaker": {}},
            {"id": "t00002", "text": "Я принимаю препарат.", "speaker": {}},
        ]
        manager._artifact_path(task_id, "turns.json").parent.mkdir(parents=True, exist_ok=True)
        manager._artifact_path(task_id, "turns.json").write_text(json.dumps({"turns": turns}), encoding="utf-8")
        manager._artifact_path(task_id, "chunks.json").write_text(json.dumps({"chunks": [{
            "id": "c01", "core_ids": ["t00001", "t00002"], "context_before": [], "context_after": [],
        }]}), encoding="utf-8")
        manager._artifact_path(task_id, "terms.json").write_text(json.dumps({"terms": []}), encoding="utf-8")
        manager._artifact_path(task_id, "language-changes.json").write_text(json.dumps({"changes": [{
            "id": "change-00001", "turn_id": "t00002", "text": "Я не принимаю препарат.",
            "original": "Я принимаю препарат.", "reason": "Ошибочная реконструкция.",
            "confidence": "mid", "chunk_id": "c01", "guardrail": "review",
        }]}), encoding="utf-8")

        details = {}
        gate = manager._review_language_adjudication(task_id, 1, details)
        changes = json.loads(manager._artifact_path(task_id, "language-changes.json").read_text(encoding="utf-8"))["changes"]

        self.assertEqual(gate["verdict"], "pass")
        self.assertEqual({item["turn_id"] for item in changes}, {"t00001"})
        self.assertEqual(changes[0]["text"], "Метформин принимаю.")
        self.assertEqual(details["adjudication_actions"]["replace"], 1)
        self.assertEqual(details["adjudication_actions"]["revert"], 1)
        self.assertEqual(details["retry_batches"], 0)
        self.assertIn("аудио недоступно", runner.prompts[0])

    def test_fidelity_xhigh_reverts_risky_change_without_language_retry(self):
        runner = FidelityRevertCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "fidelity-revert", codex_runner=runner, auto_advance=False,
        )
        task_id = "fidelity-revert.m4a"
        manager.ensure(task_id, dict(self.task, filename=task_id))
        manager._artifact_path(task_id, "speaker-registry.json").parent.mkdir(parents=True, exist_ok=True)
        manager._artifact_path(task_id, "speaker-registry.json").write_text(json.dumps({"speakers": []}), encoding="utf-8")
        manager._artifact_path(task_id, "terms.json").write_text(json.dumps({"terms": [{
            "id": "term-0001", "turn_id": "t00001", "original": "Митформин", "proposed": "Метформин",
            "decision": "accepted", "safety": "safe",
        }]}), encoding="utf-8")
        manager._artifact_path(task_id, "language-changes.json").write_text(json.dumps({"changes": [{
            "id": "change-00001", "turn_id": "t00001", "original": "Митформин принимаю.",
            "text": "Метформин не принимаю.", "reason": "Ошибочная правка.", "confidence": "mid", "guardrail": "review",
        }]}), encoding="utf-8")

        details = manager._run_fidelity(task_id, {})
        gate = manager._review_stage(task_id, "fidelity", details)
        changes = json.loads(manager._artifact_path(task_id, "language-changes.json").read_text(encoding="utf-8"))["changes"]

        self.assertEqual(gate["verdict"], "pass")
        self.assertEqual(details["deterministic_reverts"], 1)
        self.assertEqual(details["unresolved"], 0)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["text"], "Метформин принимаю.")
        self.assertEqual(changes[0]["approved_terms"][0]["id"], "term-0001")

    def test_optional_contextual_rediarization_repairs_known_speaker_boundaries(self):
        runner = ContextualRediarizationCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "rediarization",
            codex_runner=FakeCodexRunner(),
            diarization_runner=runner,
            auto_advance=False,
        )
        task_id = "focus-group.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "Пожалуйста, кто готов?"},
                {"start": 2.1, "end": 2.8, "speaker": "SPEAKER_00", "text": "Здравствуйте,"},
                {"start": 2.9, "end": 5, "speaker": "SPEAKER_01", "text": "меня зовут Юлия."},
                {"start": 5.1, "end": 5.7, "speaker": "SPEAKER_01", "text": "Спасибо."},
                {"start": 5.8, "end": 8, "speaker": "SPEAKER_00", "text": "Следующий вопрос."},
            ],
        }
        manager.ensure(task_id, task)
        manager.start(task_id, task, "source")
        self._wait_for_manager_step(manager, task_id, "source")
        manager.start(task_id, task, "structure")
        structure = self._wait_for_manager_step(manager, task_id, "structure")

        turns = manager.artifact(task_id, "structure")["turns"]
        self.assertEqual([turn["speaker"]["source_id"] for turn in turns], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
        self.assertEqual(turns[1]["text"], "Здравствуйте, меня зовут Юлия.")
        self.assertEqual(turns[2]["text"], "Спасибо. Следующий вопрос.")
        self.assertEqual(structure["details"]["contextual_rediarization"]["changed"], 2)
        self.assertEqual(structure["details"]["review_turn_count"], 0)
        self.assertEqual(runner.rediarization_calls, 1)

    def test_contextual_rediarization_caps_total_calls_at_12_with_shared_sol_context(self):
        runner = ParallelRediarizationCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "parallel-rediarization",
            codex_runner=runner,
            auto_advance=False,
        )
        task_id = "large-focus-group.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {
                    "start": index * 1.1,
                    "end": index * 1.1 + 1,
                    "speaker": f"SPEAKER_0{index % 2}",
                    "text": f"Короткий фрагмент {index}.",
                }
                for index in range(980)
            ],
        }
        manager.ensure(task_id, task)
        manager.start(task_id, task, "source")
        source = self._wait_for_manager_step(manager, task_id, "source")
        manager.start(task_id, task, "structure")
        self._wait_for_manager_step(manager, task_id, "structure")

        self.assertIn("ФГ о медицинской практике", source["gate"]["transcript_context"])
        self.assertEqual(len(runner.prompts), 12)
        self.assertEqual(runner.maximum, 12)
        total_cases = 0
        for prompt in runner.prompts:
            self.assertIn("Общий контекст исследования: ФГ о медицинской практике", prompt)
            cases = json.loads(prompt.split("Случаи: ", 1)[1])
            total_cases += len(cases)
            self.assertTrue(all(1 <= len(case["context"]) <= 7 for case in cases))
        self.assertEqual(total_cases, 980)

    def test_interior_turn_audit_splits_hidden_dialogue_inside_one_speaker_run(self):
        runner = InteriorTurnSplitCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "interior-turn-split",
            codex_runner=FakeCodexRunner(),
            diarization_runner=runner,
            auto_advance=False,
        )
        task_id = "hidden-dialogue.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Начнём."},
                {"start": 1.1, "end": 2, "speaker": "SPEAKER_01", "text": "Да, конечно."},
                {"start": 3, "end": 5, "speaker": "SPEAKER_01", "text": "То есть АГ, получается?"},
                {"start": 5.1, "end": 5.8, "speaker": "SPEAKER_00", "text": "Да, да, да."},
                {"start": 5.9, "end": 7, "speaker": "SPEAKER_00", "text": "Чистого АГ нет."},
                {"start": 7.1, "end": 9, "speaker": "SPEAKER_00", "text": "Поняла. Все согласны?"},
            ],
        }
        manager.ensure(task_id, task)
        manager.start(task_id, task, "source")
        self._wait_for_manager_step(manager, task_id, "source")
        manager.start(task_id, task, "structure")
        structure = self._wait_for_manager_step(manager, task_id, "structure")

        turns = manager.artifact(task_id, "structure")["turns"]
        self.assertEqual(
            [turn["speaker"]["source_id"] for turn in turns],
            ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01", "SPEAKER_00"],
        )
        self.assertEqual(turns[2]["text"], "То есть АГ, получается?")
        self.assertEqual(turns[3]["text"], "Да, да, да. Чистого АГ нет.")
        self.assertEqual(turns[4]["text"], "Поняла. Все согласны?")
        self.assertEqual(structure["details"]["interior_turn_splits"]["applied"], 1)
        self.assertEqual(runner.interior_split_calls, 1)

    def test_contextual_rediarization_can_be_disabled_per_run(self):
        runner = ContextualRediarizationCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "rediarization-off",
            codex_runner=runner,
            auto_advance=False,
        )
        task_id = "interview-off.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "Ваш ответ?"},
                {"start": 2.1, "end": 2.8, "speaker": "SPEAKER_00", "text": "Здравствуйте,"},
                {"start": 2.9, "end": 5, "speaker": "SPEAKER_01", "text": "меня зовут Юлия."},
            ],
        }
        manager.ensure(task_id, task)
        manager.update_settings(task_id, contextual_rediarization=False)
        manager.start(task_id, task, "source")
        self._wait_for_manager_step(manager, task_id, "source")
        manager.start(task_id, task, "structure")
        structure = self._wait_for_manager_step(manager, task_id, "structure")

        self.assertFalse(structure["details"]["contextual_rediarization"]["enabled"])
        self.assertEqual(runner.rediarization_calls, 0)
        turns = manager.artifact(task_id, "structure")["turns"]
        self.assertIn("Здравствуйте", turns[0]["text"])

    def test_low_contextual_rediarization_is_applied_and_logged_by_xhigh(self):
        runner = ContextualRediarizationCodexRunner(confidence="low")
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "rediarization-low",
            codex_runner=runner,
            auto_advance=False,
        )
        task_id = "focus-group-low.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "Кто готов?"},
                {"start": 2.1, "end": 2.8, "speaker": "SPEAKER_00", "text": "Здравствуйте,"},
                {"start": 2.9, "end": 5, "speaker": "SPEAKER_01", "text": "меня зовут Юлия."},
            ],
        }
        manager.ensure(task_id, task)
        manager.start(task_id, task, "source")
        self._wait_for_manager_step(manager, task_id, "source")
        manager.start(task_id, task, "structure")
        structure = self._wait_for_manager_step(manager, task_id, "structure")
        self.assertEqual(structure["details"]["review_turn_count"], 0)
        self.assertEqual(structure["details"]["contextual_rediarization"]["applied"], 1)
        turns = manager.artifact(task_id, "structure")["turns"]
        self.assertEqual(turns[1]["text"], "Здравствуйте, меня зовут Юлия.")
        self.assertEqual(turns[1]["speaker"]["source_id"], "SPEAKER_01")
        assumptions = manager.get(task_id)["assumptions"]
        decision = next(item for item in assumptions if item["category"] == "speaker_assignment")
        self.assertEqual(decision["confidence"], "low")
        self.assertEqual(decision["owner"], "Sol xhigh")

    def test_xhigh_automatically_returns_only_flagged_turns_for_repair(self):
        runner = StructureRepairCodexRunner()
        manager = NormalizationWorkflowManager(
            Path(self.temporary.name) / "structure-repair",
            codex_runner=runner,
            auto_advance=False,
        )
        task_id = "structure-repair.m4a"
        task = {
            "filename": task_id,
            "status": "completed",
            "result": [
                {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Сейчас,"},
                {"start": 1.1, "end": 2, "speaker": "SPEAKER_01", "text": "секундочку,"},
                {"start": 2.1, "end": 3, "speaker": "SPEAKER_00", "text": "Анна."},
            ],
        }
        manager.ensure(task_id, task)
        manager.start(task_id, task, "source")
        self._wait_for_manager_step(manager, task_id, "source")
        manager.start(task_id, task, "structure")
        deadline = time.time() + 5
        while time.time() < deadline:
            structure = next(item for item in manager.get(task_id)["steps"] if item["id"] == "structure")
            if structure["status"] == "completed" and runner.remediation_calls == 1:
                break
            time.sleep(0.01)
        else:
            self.fail(f"Automatic structure remediation did not complete: {structure.get('error')}")
        self.assertEqual(structure["status"], "completed", structure.get("error"))
        self.assertEqual(runner.remediation_calls, 1)
        chunks = next(item for item in manager.get(task_id)["steps"] if item["id"] == "chunks")
        self.assertEqual(chunks["status"], "ready")
        turns = manager.artifact(task_id, "structure")["turns"]
        self.assertEqual(turns[1]["speaker"]["source_id"], "SPEAKER_00")
        registry = json.loads(manager._artifact_path(task_id, "speaker-registry.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["automatic_turn_overrides"][-1]["turn_id"], "t00002")

    def _wait_for_manager_step(self, manager, task_id, step_id):
        deadline = time.time() + 5
        while time.time() < deadline:
            step = next(item for item in manager.get(task_id)["steps"] if item["id"] == step_id)
            if step["status"] not in {"queued", "running", "reviewing"}:
                self.assertEqual(step["status"], "completed", step.get("error"))
                return step
            time.sleep(0.01)
        self.fail(f"Step {step_id} did not finish")


if __name__ == "__main__":
    unittest.main()
