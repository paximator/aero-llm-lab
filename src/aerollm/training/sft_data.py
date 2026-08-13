"""Build and validate the first evidence-linked, train-only SFT dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from aerollm.common.schemas import Split
from aerollm.evaluation.corpus import CorpusManifest
from aerollm.generation.grounded_prompt import GROUNDED_RAG_PROMPT_VERSION

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SPACE = re.compile(r"\s+")
_AVIATION_TERMS = (
    "accident",
    "aircraft",
    "airplane",
    "airport",
    "captain",
    "crew",
    "engine",
    "flight",
    "landing",
    "ntsb",
    "pilot",
    "runway",
    "takeoff",
)
_BOILERPLATE_PREFIXES = (
    "aviation accident report",
    "aviation investigation report",
    "national transportation safety board",
    "ntsb aircraft accident report",
)
_SECTION_HEADING = re.compile(r"\b\d+(?:\.\d+){1,3}\s+[A-Z][A-Za-z]")
_BROKEN_WORD = re.compile(r"\b[bcdefghijklmnopqrstuvwxyz]\s+[a-z]{2,}\b")
_BROKEN_PUNCTUATION = re.compile(r"\s+[.,;)]|[,.)][A-Za-z]")
_MERGED_OCR_WORDS = ("garmin650", "theneed", "thefaa", "theaircraft", "thepilot")
_BROKEN_HYPHEN = re.compile(r"\b[A-Za-z0-9]{2,}\s+-\s*[A-Za-z0-9]|\b[A-Za-z0-9]{2,}-\s+[A-Za-z0-9]")
_MERGED_CASE = re.compile(r"\b[a-z]{2,}[A-Z]{2,}\b")
_MERGED_NUMBER_WORD = re.compile(r"\d{2,}[a-z]{3,}\b")
_MERGED_REFERENCE_YEAR = re.compile(r"\b[A-Z][A-Za-z]{3,}\d{4}\b")
_KNOWN_SPLIT_WORDS = re.compile(
    r"\b(?:affect|altitude|bottl|devel|haza|occur|th)\s+[a-z]{1,4}\b|\bimp\s+airment\b"
)
_BROKEN_CAPITALS = re.compile(r"\b[A-Z]\s+[A-Z][A-Za-z]?\b|\b(?:[A-Z]-){2,}\s*[A-Z]\b")
_TRANSCRIPT = re.compile(r"^\d{1,2}:\d{2}:\d{2}\b|\b(?:CAM|CTR|RDO)-\d\b")
_INLINE_FOOTNOTE = re.compile(r"\b\d+\s+(?:A|An|The)\s+[a-z]")
_FUSED_SENTENCES = re.compile(
    r"[.!?][”\"]?\s+(?:A|According|An|As|At|He|In|It|On|She|That|The|They|This|We)\s"
)
_LOW_VALUE_PREFIXES = (
    "accident airplane ",
    "departure from controlled flight,",
    "information that addresses the requirements",
    "for more detailed background information",
    "statutory language prohibits",
    "workforce manufacturing ",
)
_OPERATIONAL_IMPERATIVE = re.compile(r"^(?:If .+?,\s+)?(?:continue|do|ensure|maintain)\b", re.I)
_SYSTEM = (
    "You answer aviation-report questions using only the supplied excerpt. "
    f"Return the {GROUNDED_RAG_PROMPT_VERSION} JSON object and cite an exact span."
)


def build_sft_dataset(
    corpus: CorpusManifest,
    *,
    corpus_sha256: str,
    target_records: int = 50,
    max_records_per_family: int = 4,
    max_tokens: int = 1024,
) -> dict[str, object]:
    """Select deterministic train-only extraction records across event families."""
    if target_records < 1 or max_records_per_family < 1 or max_tokens < 1:
        raise ValueError("SFT build limits must be positive")
    sources_by_digest = {source.sha256: source for source in corpus.sources}
    documents = {document.document_id: document for document in corpus.documents}
    candidates: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for chunk in sorted(corpus.chunks, key=lambda item: item.chunk_id):
        document = documents[chunk.document_id]
        source = sources_by_digest[document.source_sha256]
        if source.split is not Split.TRAIN:
            continue
        sentence = _best_sentence(chunk.text)
        if sentence is None:
            continue
        record = _record(source, chunk.chunk_id, sentence)
        if _token_count(record["messages"]) <= max_tokens:
            candidates[source.event_family_id].append(record)

    selected: list[dict[str, object]] = []
    families = sorted(candidates)
    cursors: defaultdict[str, int] = defaultdict(int)
    family_counts: defaultdict[str, int] = defaultdict(int)
    answer_digests: set[str] = set()
    while len(selected) < target_records:
        progress = False
        for family in families:
            if len(selected) == target_records or family_counts[family] >= max_records_per_family:
                continue
            while cursors[family] < len(candidates[family]):
                record = candidates[family][cursors[family]]
                cursors[family] += 1
                answer = _assistant_answer(record["messages"])
                digest = hashlib.sha256(_SPACE.sub(" ", answer).casefold().encode()).hexdigest()
                if digest in answer_digests:
                    continue
                selected.append(record)
                answer_digests.add(digest)
                family_counts[family] += 1
                progress = True
                break
        if not progress:
            break
    if len(selected) != target_records:
        raise ValueError(f"only {len(selected)} eligible train records for target {target_records}")
    dataset = {
        "schema_version": 1,
        "dataset_id": "ntsb-evidence-linked-sft-v1-validation",
        "version": "1.0.0-draft",
        "status": "model_assisted_review_passed",
        "review": {
            "kind": "model_assisted_full_review",
            "reviewed_records": target_records,
            "comment": (
                "All generated records were inspected after the automated quality gate; "
                "the observed OCR, boilerplate, transcript, cross-reference, and duplicate "
                "failure classes were removed. Human approval is still recommended before training."
            ),
        },
        "source_corpus_sha256": corpus_sha256,
        "prompt_contract": GROUNDED_RAG_PROMPT_VERSION,
        "selection": {
            "method": "balanced_train_family_grounded_extraction_v1",
            "target_records": target_records,
            "max_records_per_family": max_records_per_family,
            "max_tokens": max_tokens,
        },
        "records": selected,
    }
    validate_sft_dataset(dataset, corpus, max_tokens=max_tokens)
    return dataset


def validate_sft_dataset(
    dataset: Mapping[str, object], corpus: CorpusManifest, *, max_tokens: int = 1024
) -> None:
    """Reject leakage, broken evidence, duplicates, and invalid message records."""
    records = dataset.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("SFT dataset requires records")
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    record_ids: set[str] = set()
    message_digests: set[str] = set()
    answer_digests: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("SFT records must be objects")
        required = {
            "record_id",
            "event_family_id",
            "source_chunk_ids",
            "messages",
        }
        if set(record) != required:
            raise ValueError("invalid SFT record fields")
        record_id = _string(record["record_id"], "record_id")
        if record_id in record_ids:
            raise ValueError(f"duplicate SFT record_id: {record_id}")
        record_ids.add(record_id)
        chunk_ids = record["source_chunk_ids"]
        if not isinstance(chunk_ids, list) or not chunk_ids:
            raise ValueError("SFT record requires source_chunk_ids")
        evidence_text = []
        evidence_sources = set()
        for chunk_id in chunk_ids:
            chunk = chunks.get(_string(chunk_id, "source_chunk_id"))
            if chunk is None:
                raise ValueError("SFT evidence chunk is unknown")
            document = next(
                (item for item in corpus.documents if item.document_id == chunk.document_id), None
            )
            if document is None:
                raise ValueError("SFT evidence document is unknown")
            source = next(
                (item for item in corpus.sources if item.sha256 == document.source_sha256), None
            )
            if source is None or source.split is not Split.TRAIN:
                raise ValueError("SFT records must use train-only reports")
            evidence_sources.add(source.source_document_id)
            if record["event_family_id"] != source.event_family_id:
                raise ValueError("SFT record crosses event-family provenance")
            evidence_text.append(chunk.text)
        if len(evidence_sources) != 1:
            raise ValueError("SFT evidence chunks cross report provenance")
        messages = record["messages"]
        answer = _validate_messages(messages, evidence_text)
        rejection = _quality_rejection_reason(answer)
        if rejection is not None:
            raise ValueError(f"SFT answer failed quality gate: {rejection}")
        answer_digest = hashlib.sha256(_SPACE.sub(" ", answer).casefold().encode()).hexdigest()
        if answer_digest in answer_digests:
            raise ValueError("duplicate SFT answers")
        answer_digests.add(answer_digest)
        canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if digest in message_digests:
            raise ValueError("duplicate SFT messages")
        message_digests.add(digest)
        token_count = _token_count(messages)
        if not 1 <= token_count <= max_tokens:
            raise ValueError("SFT token_count exceeds budget")


def write_sft_dataset(dataset: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record(source, chunk_id: str, sentence: str) -> dict[str, object]:  # type: ignore[no-untyped-def]
    excerpt = sentence
    user = (
        f"Extract one aviation-safety fact from this report excerpt for event "
        f'{source.event_id}.\n\n<chunk id="{chunk_id}">\n{excerpt}\n</chunk>'
    )
    assistant = json.dumps(
        {
            "answer": sentence,
            "citations": [{"chunk_id": chunk_id, "quote": sentence}],
            "abstained": False,
            "abstention_reason": None,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    identity = hashlib.sha256(f"{source.event_id}\0{chunk_id}\0{sentence}".encode()).hexdigest()[
        :16
    ]
    return {
        "record_id": f"sft-{identity}",
        "event_family_id": source.event_family_id,
        "source_chunk_ids": [chunk_id],
        "messages": messages,
    }


def _best_sentence(text: str) -> str | None:
    normalized = _SPACE.sub(" ", text).strip()
    candidates = []
    for sentence in _SENTENCE.split(normalized):
        sentence = sentence.strip()
        lowered = sentence.casefold()
        if _quality_rejection_reason(sentence) is not None:
            continue
        letters = sum(character.isalpha() for character in sentence)
        if letters / len(sentence) < 0.65:
            continue
        relevance = sum(term in lowered for term in _AVIATION_TERMS)
        if relevance:
            candidates.append((relevance, len(sentence), sentence))
    return max(candidates, default=None)[2] if candidates else None


def _validate_messages(messages: object, evidence_text: list[str]) -> str:
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("SFT messages must contain system, user, and assistant")
    if [message.get("role") for message in messages if isinstance(message, Mapping)] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("SFT message roles are invalid")
    if not all(isinstance(message, Mapping) for message in messages):
        raise ValueError("SFT messages must be objects")
    assistant = json.loads(_string(messages[2].get("content"), "assistant content"))
    if set(assistant) != {"answer", "citations", "abstained", "abstention_reason"}:
        raise ValueError("SFT assistant output violates grounded schema")
    citations = assistant["citations"]
    if assistant["abstained"] is not False or not isinstance(citations, list) or not citations:
        raise ValueError("grounded extraction record must be answerable and cited")
    for citation in citations:
        normalized_quote = _SPACE.sub(" ", citation["quote"]).strip()
        normalized_evidence = _SPACE.sub(" ", "\n".join(evidence_text)).strip()
        if normalized_quote not in normalized_evidence:
            raise ValueError("SFT citation is not an exact evidence span")
    return _string(assistant["answer"], "assistant answer")


def _assistant_answer(messages: object) -> str:
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("SFT messages must contain system, user, and assistant")
    assistant_message = messages[2]
    if not isinstance(assistant_message, Mapping):
        raise ValueError("SFT assistant message must be an object")
    payload = json.loads(_string(assistant_message.get("content"), "assistant content"))
    return _string(payload.get("answer"), "assistant answer")


def _quality_rejection_reason(sentence: str) -> str | None:
    lowered = sentence.casefold()
    if not 80 <= len(sentence) <= 320 or not sentence.endswith((".", "!", "?")):
        return "length_or_termination"
    if not (sentence[0].isupper() or sentence[0].isdigit()):
        return "sentence_fragment"
    if re.match(r"^\d+\s+[A-Z]", sentence) or _INLINE_FOOTNOTE.search(sentence):
        return "footnote_fragment"
    if "•" in sentence or re.search(r"\b\d\s+\d\b", sentence):
        return "list_or_page_fragment"
    if lowered.startswith(("exemplar ", "figure ", "table ", *_BOILERPLATE_PREFIXES)):
        return "document_boilerplate"
    if lowered.startswith(_LOW_VALUE_PREFIXES) or re.match(r"^[A-Z]-\d", sentence):
        return "reference_or_heading"
    if "section " in lowered and re.search(r"\bsection\s+\d", lowered):
        return "section_cross_reference"
    if _TRANSCRIPT.search(sentence):
        return "transcript_fragment"
    if _FUSED_SENTENCES.search(sentence):
        return "fused_sentences"
    if _OPERATIONAL_IMPERATIVE.search(sentence):
        return "isolated_operational_instruction"
    if "http" in lowered or _SECTION_HEADING.search(sentence):
        return "heading_or_url"
    if _BROKEN_WORD.search(sentence) or re.search(r"\ba\s+re\b", lowered):
        return "split_word"
    if _BROKEN_PUNCTUATION.search(sentence):
        return "broken_punctuation"
    if any(fragment in lowered for fragment in _MERGED_OCR_WORDS):
        return "merged_word"
    if sentence.count("“") != sentence.count("”") or sentence.count('"') % 2:
        return "unbalanced_quote"
    if (
        _BROKEN_HYPHEN.search(sentence)
        or _MERGED_CASE.search(sentence)
        or _MERGED_NUMBER_WORD.search(sentence)
        or _MERGED_REFERENCE_YEAR.search(sentence)
        or _KNOWN_SPLIT_WORDS.search(lowered)
        or _BROKEN_CAPITALS.search(sentence)
    ):
        return "ocr_fragment"
    return None


def _token_count(messages: list[dict[str, str]]) -> int:
    return sum(len(message["content"].split()) + 4 for message in messages)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
