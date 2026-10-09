from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.ocr.provider import OCRToken
from app.schemas import FieldResult

logger = logging.getLogger(__name__)

NUMBER_RE = re.compile(r"(?<!\d)([\dОO]{3}\s*/\s*[\dОOСC]{6})(?!\d)", re.IGNORECASE)
DATE_RE = re.compile(
    r"(?<!\d)(\d{2}[.\-/]\d{2}[.\-/]\d{4})(?:\s*(\d{1,2})[:\-](\d{2})(?:[:\-](\d{2}))?)?(?!\d)"
)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?7|8)[\s()\-]*(\d)[\s()\-]*(\d)[\s()\-]*(\d)"
    r"[\s()\-]*(\d{3})[\s()\-]*(\d{2})[\s()\-]*(\d{2})(?!\d)"
)
AUTHOR_RE = re.compile(
    r"(?:^|\s)автор\s*[:\-]?\s*(.*?)"
    r"(?=\s*[:\-]?\s*редактор\s*[:\-]?)",
    re.IGNORECASE,
)

LABELS: dict[str, tuple[str, ...]] = {
    "number": ("номер",),
    "date": ("дата",),
    "counterparty": ("контрагент",),
    "phone": ("телефон", "тел.", "тел"),
    "comment": ("что сделано", "комментарий"),
    "manager": ("автор", "менеджер", "ответственный"),
}
OCR_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("aта", "дaта"),
    "phone": ("тел", "тeл", "тeлeфон"),
    "manager": ("автор", "aвтор", "менеджер", "менeджер"),
}
TOP_MENU_MARKERS = ("провести", "создать", "основании", "печать", "перейти", "уведомить")
COMMENT_STOP_MARKERS = (
    "пп клиента", "конфигурация", "типовая", "раб мест", "раб.мест",
    "кол.баз", "номер релиза", "статус отзыва", "дата сдачи отзыва",
    "универс", "шаблон", "вручную", "статус протокола",
    "статус проведенных работ", "по наряду", "статус подтверждения",
)
INVALID_COUNTERPARTIES = {
    "ип выезда", "тип выезда", "контрагент", "что сделано", "комментарий",
}


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass(frozen=True)
class LabelMatch:
    field: str
    token: OCRToken
    box: Box
    label_text: str


def _box(token: OCRToken) -> Box | None:
    if not token.box:
        return None
    try:
        xs = [float(point[0]) for point in token.box]
        ys = [float(point[1]) for point in token.box]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return Box(min(xs), min(ys), max(xs), max(ys))


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_homoglyphs(text: str) -> str:
    replacements = {
        "A": "А", "a": "а", "B": "В", "c": "с", "C": "С",
        "e": "е", "E": "Е", "o": "о", "O": "О", "p": "р",
        "P": "Р", "x": "х", "X": "Х", "y": "у", "K": "К",
        "M": "М", "T": "Т",
    }
    return "".join(replacements.get(char, char) for char in text)


def _normalize_search(text: str) -> str:
    text = _normalize_homoglyphs(text).lower()
    return _normalize_spaces(
        text.replace("ё", "е").replace("—", "-").replace("–", "-")
    )


def _normalize_compact(text: str) -> str:
    return re.sub(r"[^а-яa-z0-9]+", "", _normalize_search(text))


def _clean_value(text: str) -> str:
    return _normalize_spaces(text.strip().lstrip(" :;,-–—"))


def _looks_like_number(text: str) -> bool:
    return NUMBER_RE.search(text) is not None


def _looks_like_date(text: str) -> bool:
    return DATE_RE.search(text) is not None


def _looks_like_phone(text: str) -> bool:
    return PHONE_RE.search(text) is not None


def _is_top_menu_token(text: str) -> bool:
    normalized = _normalize_search(text)
    return sum(marker in normalized for marker in TOP_MENU_MARKERS) >= 2


def _exact_label_match(text: str) -> tuple[str, str] | None:
    normalized = _normalize_search(text)
    if not normalized:
        return None
    for field, labels in LABELS.items():
        for label in labels:
            if normalized == _normalize_search(label):
                return field, label
    for field, aliases in OCR_LABEL_ALIASES.items():
        for alias in aliases:
            if normalized == _normalize_search(alias):
                return field, alias
    return None


def _embedded_label_matches(token: OCRToken) -> list[tuple[str, str]]:
    text = token.text
    if not text.strip() or _is_top_menu_token(text):
        return []
    normalized = _normalize_search(text)
    result: list[tuple[str, str]] = []

    for field, labels in LABELS.items():
        for label in labels:
            label_normalized = _normalize_search(label)
            if len(label_normalized) < 4 or label_normalized not in normalized:
                continue
            if field == "date" and re.search(r"дата\s+сдачи\s+отзыва", normalized):
                continue
            # «Номер релиза» не является номером заявки.
            if field == "number" and re.search(r"номер\s+релиза", normalized):
                continue
            result.append((field, label))
    return result


def find_labels(tokens: list[OCRToken]) -> list[LabelMatch]:
    result: list[LabelMatch] = []
    for token in tokens:
        text = token.text.strip()
        if not text:
            continue
        box = _box(token)
        if box is None:
            continue
        exact = _exact_label_match(text)
        if exact is not None:
            field, label = exact
            if field == "date" and "сдачи" in _normalize_search(text):
                continue
            result.append(LabelMatch(field, token, box, label))
            continue
        for field, label in _embedded_label_matches(token):
            result.append(LabelMatch(field, token, box, label))
    return result


def _extract_date_from_labeled_token(token: OCRToken) -> FieldResult:
    original = token.text.strip()
    if not original:
        return FieldResult()
    normalized = _normalize_homoglyphs(original)
    if not re.match(r"^\s*дата\s*[:\-]?", normalized.lower()):
        return FieldResult()
    match = DATE_RE.search(normalized.lower())
    if not match:
        return FieldResult()
    raw = match.group(0)
    return FieldResult(
        value=normalize_date(raw),
        confidence=max(0.0, min(1.0, float(token.confidence))),
        raw_text=original,
    )


def _extract_date(tokens: list[OCRToken]) -> FieldResult:
    for token in tokens:
        result = _extract_date_from_labeled_token(token)
        if result.value:
            return result
    return FieldResult()


def _extract_inline_value_same_token(field: str, token: OCRToken) -> FieldResult:
    text = token.text.strip()
    if not text:
        return FieldResult()
    normalized = _normalize_search(text)
    labels = list(LABELS.get(field, ())) + list(OCR_LABEL_ALIASES.get(field, ()))
    for label in sorted(labels, key=len, reverse=True):
        label_normalized = _normalize_search(label)
        pattern = re.compile(
            rf"^\s*{re.escape(label_normalized)}\s*(?::|-|–|—)?\s*(.+)$",
            re.IGNORECASE,
        )
        match = pattern.match(normalized)
        if not match:
            continue
        value = _clean_value(match.group(1))
        if not value:
            continue
        if field == "number" and not _looks_like_number(value):
            continue
        if field == "date" and not _looks_like_date(value):
            continue
        if field == "phone" and not _looks_like_phone(value):
            continue
        return FieldResult(
            value=value,
            confidence=max(0.0, min(1.0, float(token.confidence))),
            raw_text=token.text,
        )
    return FieldResult()


def _same_line(label_box: Box, candidate_box: Box) -> bool:
    distance = abs(candidate_box.center_y - label_box.center_y)
    reference_height = max(label_box.height, candidate_box.height, 1.0)
    return distance <= reference_height * 1.5


def _find_right_candidates(label: LabelMatch, tokens: list[OCRToken]) -> list[OCRToken]:
    candidates: list[tuple[float, OCRToken]] = []
    for token in tokens:
        if token is label.token or not token.text.strip():
            continue
        box = _box(token)
        if box is None or box.left < label.box.right:
            continue
        if not _same_line(label.box, box) or _exact_label_match(token.text.strip()):
            continue
        score = (box.left - label.box.right) + abs(box.center_y - label.box.center_y) * 3.0
        candidates.append((score, token))
    candidates.sort(key=lambda item: item[0])
    return [token for _, token in candidates]


def _extract_by_label(field: str, label: LabelMatch, tokens: list[OCRToken]) -> FieldResult:
    same_token = _extract_inline_value_same_token(field, label.token)
    if same_token.value:
        return same_token
    for token in _find_right_candidates(label, tokens):
        value = _clean_value(token.text)
        if not value:
            continue
        if field == "number" and not _looks_like_number(value):
            continue
        if field == "date" and not _looks_like_date(value):
            continue
        if field == "phone" and not _looks_like_phone(value):
            continue
        return FieldResult(
            value=value,
            confidence=max(0.0, min(1.0, float(token.confidence))),
            raw_text=token.text,
        )
    return FieldResult()


def _is_comment_stop(text: str) -> bool:
    normalized = _normalize_search(text)
    return bool(normalized) and any(marker in normalized for marker in COMMENT_STOP_MARKERS)


def _extract_comment(label: LabelMatch, tokens: list[OCRToken]) -> FieldResult:
    candidates: list[tuple[OCRToken, Box]] = []
    for token in tokens:
        if token is label.token or not token.text.strip():
            continue
        box = _box(token)
        if box is None or box.top <= label.box.bottom:
            continue
        candidates.append((token, box))
    candidates.sort(key=lambda item: (item[1].top, item[1].left))

    selected: list[tuple[OCRToken, Box]] = []
    for token, box in candidates:
        if _is_comment_stop(token.text):
            break
        exact = _exact_label_match(token.text.strip())
        if exact is not None and exact[0] != "comment":
            break
        selected.append((token, box))
    if not selected:
        return FieldResult()

    rows: list[list[tuple[OCRToken, Box]]] = []
    for token, box in selected:
        if not rows:
            rows.append([(token, box)])
            continue
        row = rows[-1]
        center_y = sum(item[1].center_y for item in row) / len(row)
        if abs(box.center_y - center_y) <= max(8.0, box.height * 0.8):
            row.append((token, box))
        else:
            rows.append([(token, box)])

    lines = []
    for row in rows:
        row.sort(key=lambda item: item[1].left)
        line = _clean_value(" ".join(token.text.strip() for token, _ in row if token.text.strip()))
        if line:
            lines.append(line)
    value = "\n".join(lines)
    if not value:
        return FieldResult()
    confidence = sum(float(token.confidence) for token, _ in selected) / len(selected)
    return FieldResult(
        value=value,
        confidence=max(0.0, min(1.0, confidence)),
        raw_text=value,
    )


def _looks_like_counterparty(text: str) -> bool:
    normalized = _normalize_search(text)
    if not normalized or len(normalized) > 100:
        return False
    # Элементы интерфейса не являются названием контрагента.
    if normalized in INVALID_COUNTERPARTIES:
        return False
    if re.search(r"(?:^|[\s.,])(?:ооо|000|оо0|0оо)\s*$", normalized, re.IGNORECASE):
        return True
    if re.match(r"^ип(?:\s|\.|$)", normalized, re.IGNORECASE):
        return True
    if re.match(r"^(?:ао|оао|зао)(?:\s|\.|$)", normalized, re.IGNORECASE):
        return True
    return False


def normalize_counterparty(value: str) -> str:
    value = _normalize_homoglyphs(_normalize_spaces(value))
    value = re.sub(r"(?:000|оо0|0оо|оoо|ooo)$", "ООО", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*\.\s*ООО\s*$", " ООО", value, flags=re.IGNORECASE)
    return _clean_value(value)


def _extract_counterparty_fallback(tokens: list[OCRToken]) -> FieldResult:
    candidates: list[tuple[float, OCRToken]] = []
    for token in tokens:
        if not token.text.strip() or not _looks_like_counterparty(token.text):
            continue
        score = float(token.confidence)
        normalized = _normalize_search(token.text)
        if re.search(r"(?:000|ооо|оо0|0оо)\s*$", normalized, re.IGNORECASE):
            score += 0.5
        candidates.append((score, token))
    if not candidates:
        return FieldResult()
    candidates.sort(key=lambda item: item[0], reverse=True)
    token = candidates[0][1]
    return FieldResult(
        value=normalize_counterparty(token.text),
        confidence=max(0.0, min(1.0, float(token.confidence))),
        raw_text=token.text,
    )


def _extract_manager_from_author(tokens: list[OCRToken]) -> FieldResult:
    if not tokens:
        return FieldResult()
    full_text = " ".join(token.text.strip() for token in tokens if token.text.strip())
    normalized = _normalize_homoglyphs(full_text)
    match = AUTHOR_RE.search(normalized)
    if not match:
        logger.info("MANAGER: Автор не найден в OCR")
        return FieldResult()
    value = _clean_value(match.group(1)).rstrip(" :;-").strip()
    if not value:
        return FieldResult()
    normalized_value = _normalize_search(value)
    confidences = [
        float(token.confidence)
        for token in tokens
        if normalized_value and normalized_value in _normalize_search(token.text)
    ]
    confidence = sum(confidences) / len(confidences) if confidences else 0.80
    return FieldResult(
        value=value,
        confidence=max(0.0, min(1.0, confidence)),
        raw_text=match.group(0),
    )


def normalize_number(value: str) -> str:
    value = value.upper().strip().replace("О", "0").replace("O", "0")
    value = re.sub(r"\s+", "", value)
    match = re.search(r"(\d{3})/(\d{6})", value)
    return f"{match.group(1)}/{match.group(2)}" if match else value


def normalize_date(value: str) -> str:
    match = DATE_RE.search(value.strip())
    if not match:
        return value.strip()
    date_part, hour, minute, second = match.groups()
    date_part = date_part.replace("/", ".").replace("-", ".")
    if hour and minute:
        return f"{date_part} {hour}:{minute}:{second}" if second else f"{date_part} {hour}:{minute}"
    return date_part


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 11:
        return value.strip()
    if digits.startswith("8"):
        digits = "7" + digits[1:]
    if not digits.startswith("7"):
        return value.strip()
    return f"+7 ({digits[1:4]}) {digits[4:7]}-{digits[7:9]}-{digits[9:11]}"


def _extract_pattern(field: str, full_text: str) -> FieldResult:
    pattern = {"number": (NUMBER_RE, normalize_number, 0.75),
               "date": (DATE_RE, normalize_date, 0.70),
               "phone": (PHONE_RE, normalize_phone, 0.75)}
    if field not in pattern:
        return FieldResult()
    regex, normalizer, confidence = pattern[field]
    match = regex.search(full_text)
    if not match:
        return FieldResult()
    raw = match.group(1) if field == "number" else match.group(0)
    return FieldResult(value=normalizer(raw), confidence=confidence, raw_text=raw)


def extract_from_full_text(tokens: list[OCRToken]) -> dict[str, FieldResult]:
    fields: dict[str, FieldResult] = {
        name: FieldResult()
        for name in ("number", "date", "counterparty", "phone", "comment", "manager")
    }
    if not tokens:
        return fields

    labels = find_labels(tokens)
    logger.info("========== FOUND LABELS ==========")
    for label in labels:
        logger.info("LABEL field=%s text=%r box=%s token=%r",
                    label.field, label.label_text, label.box, label.token.text)
    logger.info("========== END FOUND LABELS ==========")

    fields["date"] = _extract_date(tokens)

    for field in ("number", "counterparty", "phone"):
        for label in (item for item in labels if item.field == field):
            result = _extract_by_label(field, label, tokens)
            if result.value:
                fields[field] = result
                break

    comment_labels = [item for item in labels if item.field == "comment"]
    comment_labels.sort(key=lambda item: 0 if item.label_text == "что сделано" else 1)
    for label in comment_labels:
        result = _extract_comment(label, tokens)
        if result.value:
            fields["comment"] = result
            break

    full_text = " ".join(token.text for token in tokens if token.text.strip())
    if fields["number"].value:
        fields["number"] = FieldResult(
            value=normalize_number(fields["number"].value),
            confidence=fields["number"].confidence,
            raw_text=fields["number"].raw_text,
        )
    else:
        fields["number"] = _extract_pattern("number", full_text)

    if not fields["date"].value:
        fields["date"] = _extract_pattern("date", full_text)

    if fields["phone"].value:
        phone_match = PHONE_RE.search(fields["phone"].value)
        if phone_match:
            fields["phone"] = FieldResult(
                value=normalize_phone(phone_match.group(0)),
                confidence=fields["phone"].confidence,
                raw_text=fields["phone"].raw_text,
            )
        else:
            fields["phone"] = FieldResult()
    if not fields["phone"].value:
        fields["phone"] = _extract_pattern("phone", full_text)

    if fields["counterparty"].value:
        value = normalize_counterparty(fields["counterparty"].value)
        if value.lower() in INVALID_COUNTERPARTIES:
            fields["counterparty"] = FieldResult()
        else:
            fields["counterparty"] = FieldResult(
                value=value,
                confidence=fields["counterparty"].confidence,
                raw_text=fields["counterparty"].raw_text,
            )
    else:
        fields["counterparty"] = _extract_counterparty_fallback(tokens)

    fields["manager"] = _extract_manager_from_author(tokens)
    logger.info(
        "EXTRACTION DETAILS: labels=%s",
        [{"field": item.field, "label": item.label_text,
          "token": item.token.text, "box": item.box} for item in labels],
    )
    return fields
