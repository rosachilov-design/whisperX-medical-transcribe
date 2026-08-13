"""Independent, persistent Codex jobs for conclusions from operator transcripts."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
import zipfile
from html import escape as xml_escape
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from normalization_workflow import CodexRunner


SUPPORTED_CONCLUSION_EXTENSIONS = {".md", ".txt", ".json", ".docx"}
ACTIVE_CONCLUSION_STATUSES = {"queued", "running"}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or "transcript"
    return re.sub(r"[^\w.() -]+", "_", stem, flags=re.UNICODE)[:120]


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            document = archive.read("word/document.xml")
        root = ElementTree.fromstring(document)
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs = []
        for paragraph in root.iter(namespace + "p"):
            text = "".join(node.text or "" for node in paragraph.iter(namespace + "t"))
            if text.strip():
                paragraphs.append(text)
        return "\n\n".join(paragraphs)
    raise ValueError("Поддерживаются файлы .md, .txt, .json и .docx.")


def write_windows_docx(path: Path, content: str) -> None:
    """Create a compact Word-compatible DOCX without an optional runtime dependency."""
    paragraphs = []
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        style = "Normal"
        text = line
        if line.startswith("### "):
            style, text = "Heading3", line[4:]
        elif line.startswith("## "):
            style, text = "Heading2", line[3:]
        elif line.startswith("# "):
            style, text = "Heading1", line[2:]
        elif re.match(r"^[-*] ", line):
            style, text = "ListBullet", "• " + line[2:]
        elif re.match(r"^\d+\. ", line):
            style, text = "ListNumber", line
        style_xml = f'<w:pStyle w:val="{style}"/>' if style != "Normal" else ""
        if text:
            paragraphs.append(
                f'<w:p><w:pPr>{style_xml}</w:pPr><w:r><w:t xml:space="preserve">{xml_escape(text)}</w:t></w:r></w:p>'
            )
        else:
            paragraphs.append("<w:p/>")
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>' + "".join(paragraphs) +
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr>'
        '</w:body></w:document>'
    )
    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="22"/><w:lang w:val="ru-RU"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListNumber"><w:name w:val="List Number"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    document_rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
        archive.writestr("word/_rels/document.xml.rels", document_rels)


class ConclusionsWorkflowManager:
    def __init__(self, root: Path, codex_runner: CodexRunner | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.codex = codex_runner or CodexRunner(model="gpt-5.6-sol", reasoning_effort="high")
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        for state_path in self.root.glob("*/state.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("status") in ACTIVE_CONCLUSION_STATUSES:
                    state["status"] = "failed"
                    state["error"] = "Сервер был перезапущен во время выполнения. Запустите файл повторно."
                    state["finished_at"] = int(time.time())
                    _atomic_json(state_path, state)
                self._tasks[state["id"]] = state
            except (OSError, KeyError, json.JSONDecodeError):
                continue

    def _task_dir(self, task_id: str) -> Path:
        return self.root / task_id

    def _save(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._tasks[state["id"]] = state
            _atomic_json(self._task_dir(state["id"]) / "state.json", state)

    @staticmethod
    def _public(state: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in state.items() if key not in {"source_path", "result_path", "result_paths", "instruction"}}

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            tasks = sorted(self._tasks.values(), key=lambda item: item.get("created_at", 0), reverse=True)
            return [self._public(dict(task)) for task in tasks]

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return self._public(dict(task)) if task else None

    def create(self, filename: str, content: bytes, instruction: str) -> dict[str, Any]:
        safe_name = Path(filename or "transcript.txt").name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_CONCLUSION_EXTENSIONS:
            raise ValueError("Поддерживаются файлы .md, .txt, .json и .docx.")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Добавьте инструкцию для выводов.")

        task_id = uuid.uuid4().hex
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=False)
        source_path = task_dir / ("source" + suffix)
        source_path.write_bytes(content)
        state = {
            "id": task_id,
            "filename": safe_name,
            "status": "queued",
            "progress": 4,
            "message": "Файл добавлен в параллельную очередь",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "created_at": int(time.time()),
            "source_path": str(source_path),
            "instruction": instruction,
        }
        self._save(state)
        threading.Thread(target=self._run, args=(task_id,), daemon=True, name=f"conclusions-{task_id[:8]}").start()
        return self._public(state)

    def _run(self, task_id: str) -> None:
        with self._lock:
            state = self._tasks[task_id]
        try:
            state.update(status="running", progress=18, message="Codex Sol High читает операторскую версию")
            self._save(state)
            source_text = extract_document_text(Path(state["source_path"]))
            if not source_text.strip():
                raise ValueError("В файле не найден текст.")
            state.update(progress=35, message="Codex Sol High формирует выводы")
            self._save(state)
            prompt = (
                "Ты готовишь отдельный аналитический документ с выводами по операторской версии транскрипта.\n"
                "Не исправляй, не нормализуй и не пересобирай транскрипт. Не возвращай изменённую версию источника.\n"
                "Строго следуй пользовательской инструкции. Верни готовый Markdown в поле content и короткий заголовок в title.\n\n"
                f"Если название фокус-группы не указано внутри стенограммы, используй основу имени файла: {_safe_stem(state['filename'])}. Не выдумывай отдельное название.\n\n"
                f"ИНСТРУКЦИЯ:\n{state['instruction']}\n\n"
                f"ОПЕРАТОРСКАЯ ВЕРСИЯ ({state['filename']}):\n---\n{source_text}\n---"
            )
            schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
                "additionalProperties": False,
            }
            result = self.codex.run(self._task_dir(task_id) / "codex", prompt, schema)
            content = str(result.get("content") or "").strip()
            if not content:
                raise RuntimeError("Codex вернул пустой документ.")
            result_stem = f"{_safe_stem(state['filename'])}_conclusions"
            txt_name = result_stem + ".txt"
            docx_name = result_stem + ".docx"
            txt_path = self._task_dir(task_id) / txt_name
            docx_path = self._task_dir(task_id) / docx_name
            txt_path.write_text(content + "\n", encoding="utf-8-sig")
            write_windows_docx(docx_path, content)
            state.update(
                status="completed",
                progress=100,
                message="Выводы готовы",
                title=str(result.get("title") or "Выводы").strip(),
                result_filenames={"txt": txt_name, "docx": docx_name},
                result_paths={"txt": str(txt_path), "docx": str(docx_path)},
                finished_at=int(time.time()),
            )
            state.pop("error", None)
        except Exception as exc:
            state.update(status="failed", progress=100, message="Не удалось сформировать выводы", error=str(exc), finished_at=int(time.time()))
        self._save(state)

    def result_path(self, task_id: str, result_format: str = "docx") -> tuple[Path, str]:
        with self._lock:
            state = self._tasks.get(task_id)
            if not state:
                raise KeyError(task_id)
            result_format = result_format if result_format in {"txt", "docx"} else "docx"
            if state.get("status") != "completed" or not state.get("result_paths", {}).get(result_format):
                raise FileNotFoundError(task_id)
            return Path(state["result_paths"][result_format]), state["result_filenames"][result_format]
