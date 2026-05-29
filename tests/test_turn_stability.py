import importlib
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_handler_module():
    sys.modules.pop("handler", None)

    fake_runpod = types.ModuleType("runpod")
    fake_runpod.serverless = types.SimpleNamespace(start=lambda *args, **kwargs: None)

    fake_whisperx = types.ModuleType("whisperx")

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
    )

    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda *args, **kwargs: None
    fake_requests.exceptions = types.SimpleNamespace(HTTPError=Exception)

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *args, **kwargs: None

    fake_botocore = types.ModuleType("botocore")
    fake_botocore_config = types.ModuleType("botocore.config")
    fake_pandas = types.ModuleType("pandas")

    class FakeConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeDataFrame:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_pandas.isna = lambda value: value is None
    fake_pandas.DataFrame = FakeDataFrame
    fake_botocore_config.Config = FakeConfig

    sys.modules["runpod"] = fake_runpod
    sys.modules["whisperx"] = fake_whisperx
    sys.modules["torch"] = fake_torch
    sys.modules["requests"] = fake_requests
    sys.modules["boto3"] = fake_boto3
    sys.modules["botocore"] = fake_botocore
    sys.modules["botocore.config"] = fake_botocore_config
    sys.modules["pandas"] = fake_pandas

    return importlib.import_module("handler")


HANDLER = load_handler_module()


def build_word(text, speaker, start, end):
    return {
        "word": text,
        "speaker": speaker,
        "start": start,
        "end": end,
    }


class TurnStabilityTests(unittest.TestCase):
    def test_robust_mapping_uses_ocr_name_when_confident(self):
        self.assertEqual(HANDLER._normalize_known_speakers("Anna; Sergey ;Anna"), ["Anna", "Sergey"])

        pyannote_timeline = [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 7.0, "speaker": "SPEAKER_01"},
        ]
        ocr_timeline = [
            {"start": 0.0, "end": 4.0, "speaker": "Anna"},
            {"start": 4.0, "end": 7.0, "speaker": HANDLER.UNKNOWN_SPEAKER},
        ]

        mapping, details = HANDLER.map_pyannote_speakers_to_ocr_names(pyannote_timeline, ocr_timeline)
        named_timeline = HANDLER.apply_speaker_mapping_to_timeline(pyannote_timeline, mapping)

        self.assertEqual(mapping, {"SPEAKER_00": "Anna"})
        self.assertTrue(details["SPEAKER_00"]["accepted"])
        self.assertFalse(details["SPEAKER_01"]["accepted"])
        self.assertEqual(named_timeline[0]["speaker"], "Anna")
        self.assertEqual(named_timeline[1]["speaker"], HANDLER.UNKNOWN_SPEAKER)
        self.assertEqual(named_timeline[1]["speaker_raw"], "SPEAKER_01")

    def test_false_split_inside_sentence_merges_back(self):
        split_segments = [
            HANDLER._words_to_segment(
                [build_word("Ничего", "SPEAKER_01", 46.93, 47.29)],
                "SPEAKER_01",
            ),
            HANDLER._words_to_segment(
                [build_word("себе.", "SPEAKER_00", 47.33, 48.37)],
                "SPEAKER_00",
            ),
        ]

        repaired = HANDLER.repair_fragmented_turns(split_segments)

        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0]["speaker"], "SPEAKER_00")
        self.assertEqual(repaired[0]["text"], "Ничего себе.")

    def test_prefix_fragment_reassigns_to_next_speaker(self):
        split_segments = [
            HANDLER._words_to_segment(
                [
                    build_word("Это", "SPEAKER_01", 185.55, 185.66),
                    build_word("при", "SPEAKER_01", 185.67, 185.79),
                ],
                "SPEAKER_01",
            ),
            HANDLER._words_to_segment(
                [
                    build_word("каком", "SPEAKER_00", 185.81, 186.25),
                    build_word("давлении?", "SPEAKER_00", 186.26, 186.93),
                ],
                "SPEAKER_00",
            ),
        ]

        repaired = HANDLER.repair_fragmented_turns(split_segments)

        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0]["speaker"], "SPEAKER_00")
        self.assertEqual(repaired[0]["text"], "Это при каком давлении?")

    def test_short_acknowledgement_stays_separate_turn(self):
        segments = [
            HANDLER._words_to_segment(
                [build_word("Продолжайте.", "SPEAKER_00", 1.00, 1.80)],
                "SPEAKER_00",
            ),
            HANDLER._words_to_segment(
                [build_word("Да.", "SPEAKER_01", 1.90, 2.10)],
                "SPEAKER_01",
            ),
            HANDLER._words_to_segment(
                [build_word("Хорошо,", "SPEAKER_00", 2.20, 2.70)],
                "SPEAKER_00",
            ),
        ]

        repaired = HANDLER.repair_fragmented_turns(segments)

        self.assertEqual([seg["speaker"] for seg in repaired], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
        self.assertEqual(repaired[1]["text"], "Да.")

    def test_aba_backchannel_remains_separate(self):
        segments = [
            HANDLER._words_to_segment(
                [build_word("Я", "SPEAKER_00", 10.0, 10.1), build_word("думаю.", "SPEAKER_00", 10.1, 10.8)],
                "SPEAKER_00",
            ),
            HANDLER._words_to_segment(
                [build_word("Угу.", "SPEAKER_01", 10.9, 11.1)],
                "SPEAKER_01",
            ),
            HANDLER._words_to_segment(
                [build_word("Нужно", "SPEAKER_00", 11.2, 11.5), build_word("проверить.", "SPEAKER_00", 11.5, 12.0)],
                "SPEAKER_00",
            ),
        ]

        repaired = HANDLER.repair_fragmented_turns(segments)

        self.assertEqual(len(repaired), 3)
        self.assertEqual([seg["speaker"] for seg in repaired], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])

    def test_unknown_short_acknowledgement_between_same_speakers_is_recovered(self):
        segments = [
            HANDLER._words_to_segment(
                [build_word("Расскажите.", "SPEAKER_00", 0.0, 0.7)],
                "SPEAKER_00",
            ),
            HANDLER._words_to_segment(
                [build_word("Да.", HANDLER.UNKNOWN_SPEAKER, 0.8, 0.95)],
                HANDLER.UNKNOWN_SPEAKER,
            ),
            HANDLER._words_to_segment(
                [build_word("Спасибо.", "SPEAKER_00", 1.0, 1.6)],
                "SPEAKER_00",
            ),
        ]

        repaired = HANDLER.repair_short_acknowledgement_speakers(segments)

        self.assertEqual(repaired[1]["speaker"], "SPEAKER_00")

    def test_splitter_ignores_tiny_single_word_flip(self):
        segments = [
            {
                "speaker": "SPEAKER_00",
                "words": [
                    build_word("Я", "SPEAKER_00", 0.00, 0.10),
                    build_word("думаю", "SPEAKER_00", 0.10, 0.40),
                    build_word("да", "SPEAKER_01", 0.40, 0.58),
                    build_word("правильно.", "SPEAKER_00", 0.58, 1.20),
                ],
            }
        ]

        split_segments = HANDLER.split_by_word_speakers(segments)

        self.assertEqual(len(split_segments), 1)
        self.assertEqual(split_segments[0]["speaker"], "SPEAKER_00")


if __name__ == "__main__":
    unittest.main()
