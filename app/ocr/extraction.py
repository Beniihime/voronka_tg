from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.ocr.provider import OCRToken
from app.schemas import FieldResult

logger = logging.getLogger(__name__)


# =============================================================================
# REGEX
# =============================================================================

NUMBER_RE = re.compile(
    r"(?<!\d)"
    r"([\dОO]{3}\s*/\s*[\dОOСC]{6})"
    r"(?!\d)",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"(?<!\d)"
    r"(\d{2}[.\-/]\d{2}[.\-/]\d{4}"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)"
    r"(?!\d)"
)

PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?7|8)"
    r"[\s()\-]*"
    r"(\d)"
    r"[\s()\-]*"
    r"(\d)"
    r"[\s()\-]*"
    r"(\d)"
    r"[\s()\-]*"
    r"(\d{3})"
    r"[\s()\-]*"
    r"(\d{2})"
    r"[\s()\-]*"
    r"(\d{2})"
    r"(?!\d)"
)


# =============================================================================
# LABELS
# =============================================================================

# ВАЖНО:
# "№" больше не используем как обычный label.
# Иначе он находится внутри адресов и других текстов.
LABELS: dict[str, tuple[str, ...]] = {
    "number": (
        "номер",
    ),
    "date": (
        "дата",
    ),
    "counterparty": (
        "контрагент",
    ),
    "phone": (
        "телефон",
        "тел.",
        "тел",
    ),
    "comment": (
        "что сделано",
        "комментарий",
    ),
    "manager": (
        "менеджер",
        "ответственный",
        "автор",
    ),
}


# Допустимые OCR-варианты.
OCR_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "date": (
        "ата",
        "aта",
        "дaта",
    ),
    "phone": (
        "тел",
        "тeл",
        "тeлeфон",
    ),
    "counterparty": (
        "контрaгент",
        "контрагeнт",
    ),
    "manager": (
        "менeджер",
        "менеджeр",
        "менедж",
    ),
}


# Что точно говорит о верхнем меню, а не о поле комментария.
TOP_MENU_MARKERS = (
    "провести",
    "создать",
    "основании",
    "печать",
    "перейти",
    "уведомить",
)


# Что говорит о начале следующего блока после комментария.
COMMENT_STOP_MARKERS = (
    "пп клиента",
    "конфигурация",
    "типовая",
    "раб мест",
    "раб.мест",
    "кол.баз",
    "номер релиза",
    "статус отзыва",
    "дата сдачи отзыва",
    "универс",
    "шаблон",
    "вручную",
    "статус протокола",
    "статус проведенных работ",
    "по наряду",
    "статус подтверждения",
)


# =============================================================================
# GEOMETRY
# =============================================================================

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

    return Box(
        left=min(xs),
        top=min(ys),
        right=max(xs),
        bottom=max(ys),
    )


# =============================================================================
# TEXT
# =============================================================================

def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_search(text: str) -> str:
    text = text.lower().strip()

    replacements = {
        "ё": "е",
        "—": "-",
        "–": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return _normalize_spaces(text)


def _normalize_compact(text: str) -> str:
    text = _normalize_search(text)
    return re.sub(r"[^а-яa-z0-9]+", "", text)


def _clean_value(text: str) -> str:
    text = text.strip()
    text = text.lstrip(" :;,-–—")
    text = _normalize_spaces(text)
    return text.strip()


# =============================================================================
# FORMAT VALIDATORS
# =============================================================================

def _looks_like_number(text: str) -> bool:
    return NUMBER_RE.search(text) is not None


def _looks_like_date(text: str) -> bool:
    return DATE_RE.search(text) is not None


def _looks_like_phone(text: str) -> bool:
    return PHONE_RE.search(text) is not None


# =============================================================================
# LABEL MATCHING
# =============================================================================

def _is_top_menu_token(text: str) -> bool:
    normalized = _normalize_search(text)

    matches = 0

    for marker in TOP_MENU_MARKERS:
        if marker in normalized:
            matches += 1

    return matches >= 2


def _exact_label_match(
    text: str,
) -> tuple[str, str] | None:

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


def _fuzzy_label_match(
    text: str,
) -> tuple[str, str] | None:

    normalized = _normalize_compact(text)

    if not normalized:
        return None

    # Одно-/двухбуквенные фрагменты fuzzy-search запрещаем.
    if len(normalized) < 4:
        return None

    best_field: str | None = None
    best_label: str | None = None
    best_score = 0.0

    candidates: list[
        tuple[str, str]
    ] = []

    for field, labels in LABELS.items():
        for label in labels:
            candidates.append(
                (field, label)
            )

    for field, aliases in OCR_LABEL_ALIASES.items():
        for alias in aliases:
            candidates.append(
                (field, alias)
            )

    for field, label in candidates:
        target = _normalize_compact(label)

        if not target:
            continue

        # Fuzzy только для приблизительно одинаковых по размеру слов.
        if abs(len(normalized) - len(target)) > 4:
            continue

        score = SequenceMatcher(
            None,
            normalized,
            target,
        ).ratio()

        if score > best_score:
            best_score = score
            best_field = field
            best_label = label

    if best_score >= 0.78 and best_field and best_label:
        return best_field, best_label

    return None


def _embedded_label_matches(
    token: OCRToken,
) -> list[tuple[str, str]]:

    text = token.text

    if not text.strip():
        return []

    normalized = _normalize_search(text)

    result: list[tuple[str, str]] = []

    # -------------------------------------------------------------------------
    # Верхнее меню:
    #
    # "Провести ... Дополнительно ..."
    #
    # "дополнительно" здесь НЕ является comment label.
    # -------------------------------------------------------------------------

    if _is_top_menu_token(text):
        return []

    # -------------------------------------------------------------------------
    # Ищем только многосимвольные labels.
    # "№" здесь принципиально нет.
    # -------------------------------------------------------------------------

    for field, labels in LABELS.items():
        for label in labels:
            label_normalized = _normalize_search(label)

            if len(label_normalized) < 4:
                continue

            if label_normalized not in normalized:
                continue

            # Для "дата" не принимаем:
            #
            # "Дата сдачи отзыва"
            #
            # как отдельный date label.
            if (
                field == "date"
                and re.search(
                    r"дата\s+сдачи\s+отзыва",
                    normalized,
                )
            ):
                continue

            result.append(
                (
                    field,
                    label,
                )
            )

    return result


def find_labels(
    tokens: list[OCRToken],
) -> list[LabelMatch]:

    result: list[LabelMatch] = []

    for token in tokens:
        text = token.text.strip()

        if not text:
            continue

        box = _box(token)

        if box is None:
            continue

        # ---------------------------------------------------------------------
        # 1. Точное совпадение целого токена.
        # ---------------------------------------------------------------------

        exact = _exact_label_match(text)

        if exact is not None:
            field, label = exact

            result.append(
                LabelMatch(
                    field=field,
                    token=token,
                    box=box,
                    label_text=label,
                )
            )

            continue

        # ---------------------------------------------------------------------
        # 2. Для коротких/искажённых label пробуем fuzzy.
        # ---------------------------------------------------------------------

        fuzzy = _fuzzy_label_match(text)

        if fuzzy is not None:
            field, label = fuzzy

            # Длинный текст с похожим словом не принимаем.
            if len(_normalize_search(text)) <= len(
                _normalize_search(label)
            ) + 5:
                result.append(
                    LabelMatch(
                        field=field,
                        token=token,
                        box=box,
                        label_text=label,
                    )
                )

                continue

        # ---------------------------------------------------------------------
        # 3. Склеенные OCR-блоки.
        #
        # Например:
        #
        # "Акты /Отчеты сотрудникаЧто сделано ... Комментарий ..."
        # ---------------------------------------------------------------------

        embedded = _embedded_label_matches(
            token
        )

        for field, label in embedded:
            result.append(
                LabelMatch(
                    field=field,
                    token=token,
                    box=box,
                    label_text=label,
                )
            )

    return result


# =============================================================================
# INLINE VALUE
# =============================================================================

def _extract_inline_value_from_same_token(
    field: str,
    token: OCRToken,
) -> FieldResult:

    text = token.text.strip()

    if not text:
        return FieldResult()

    normalized = _normalize_search(text)

    possible_labels = list(
        LABELS.get(field, ())
    )

    possible_labels.extend(
        OCR_LABEL_ALIASES.get(field, ())
    )

    for label in sorted(
        possible_labels,
        key=len,
        reverse=True,
    ):
        label_normalized = _normalize_search(
            label
        )

        pattern = re.compile(
            rf"^\s*"
            rf"{re.escape(label_normalized)}"
            rf"\s*(?::|-|–|—)?\s*"
            rf"(.+)$",
            re.IGNORECASE,
        )

        match = pattern.match(
            normalized
        )

        if not match:
            continue

        value = _clean_value(
            match.group(1)
        )

        if not value:
            continue

        if field == "date" and not _looks_like_date(
            value
        ):
            continue

        if field == "number" and not _looks_like_number(
            value
        ):
            continue

        if field == "phone" and not _looks_like_phone(
            value
        ):
            continue

        return FieldResult(
            value=value,
            confidence=max(
                0.0,
                min(
                    1.0,
                    float(token.confidence),
                ),
            ),
            raw_text=token.text,
        )

    return FieldResult()


# =============================================================================
# SPATIAL SEARCH
# =============================================================================

def _is_same_line(
    label_box: Box,
    candidate_box: Box,
) -> bool:

    vertical_distance = abs(
        label_box.center_y
        - candidate_box.center_y
    )

    line_height = max(
        label_box.height,
        candidate_box.height,
        1.0,
    )

    return vertical_distance <= line_height * 1.5


def _find_right_candidates(
    label: LabelMatch,
    tokens: list[OCRToken],
) -> list[OCRToken]:

    result: list[
        tuple[float, OCRToken]
    ] = []

    for token in tokens:
        if token is label.token:
            continue

        if not token.text.strip():
            continue

        box = _box(token)

        if box is None:
            continue

        if _exact_label_match(
            token.text
        ) is not None:
            continue

        if box.left < label.box.right:
            continue

        if not _is_same_line(
            label.box,
            box,
        ):
            continue

        horizontal_distance = (
            box.left - label.box.right
        )

        vertical_distance = abs(
            box.center_y
            - label.box.center_y
        )

        score = (
            horizontal_distance
            + vertical_distance * 3.0
        )

        result.append(
            (
                score,
                token,
            )
        )

    result.sort(
        key=lambda item: item[0]
    )

    return [
        token
        for _, token in result
    ]


def _extract_by_label(
    field: str,
    label: LabelMatch,
    tokens: list[OCRToken],
) -> FieldResult:

    # -------------------------------------------------------------------------
    # Если label + value в одном OCR token.
    # -------------------------------------------------------------------------

    same_token = _extract_inline_value_from_same_token(
        field,
        label.token,
    )

    if same_token.value:
        return same_token

    # -------------------------------------------------------------------------
    # Значение справа.
    # -------------------------------------------------------------------------

    candidates = _find_right_candidates(
        label,
        tokens,
    )

    for token in candidates:
        value = _clean_value(
            token.text
        )

        if not value:
            continue

        if field == "number" and not _looks_like_number(
            value
        ):
            continue

        if field == "date" and not _looks_like_date(
            value
        ):
            continue

        if field == "phone" and not _looks_like_phone(
            value
        ):
            continue

        return FieldResult(
            value=value,
            confidence=max(
                0.0,
                min(
                    1.0,
                    float(token.confidence),
                ),
            ),
            raw_text=token.text,
        )

    return FieldResult()


# =============================================================================
# COMMENT STOP
# =============================================================================

def _is_comment_stop_token(
    text: str,
) -> bool:

    normalized = _normalize_search(
        text
    )

    if not normalized:
        return False

    for marker in COMMENT_STOP_MARKERS:
        if marker in normalized:
            return True

    # Немного fuzzy для OCR ошибок:
    compact = _normalize_compact(
        normalized
    )

    for marker in COMMENT_STOP_MARKERS:
        target = _normalize_compact(
            marker
        )

        if len(target) < 5:
            continue

        if len(compact) > len(target) + 8:
            continue

        score = SequenceMatcher(
            None,
            compact,
            target,
        ).ratio()

        if score >= 0.80:
            return True

    return False


# =============================================================================
# COMMENT
# =============================================================================

def _extract_comment(
    label: LabelMatch,
    tokens: list[OCRToken],
) -> FieldResult:

    candidates: list[
        tuple[OCRToken, Box]
    ] = []

    for token in tokens:
        if token is label.token:
            continue

        if not token.text.strip():
            continue

        box = _box(token)

        if box is None:
            continue

        if box.top <= label.box.bottom:
            continue

        candidates.append(
            (
                token,
                box,
            )
        )

    if not candidates:
        return FieldResult()

    candidates.sort(
        key=lambda item: (
            item[1].top,
            item[1].left,
        )
    )

    # -------------------------------------------------------------------------
    # Идём сверху вниз.
    #
    # Останавливаемся на следующем UI-блоке.
    # -------------------------------------------------------------------------

    selected: list[
        tuple[OCRToken, Box]
    ] = []

    for token, box in candidates:

        if _is_comment_stop_token(
            token.text
        ):
            break

        # Если встретили целый структурный label — прекращаем комментарий.
        exact = _exact_label_match(
            token.text
        )

        if exact is not None:
            field, _ = exact

            if field not in {
                "comment",
            }:
                break

        selected.append(
            (
                token,
                box,
            )
        )

    if not selected:
        return FieldResult()

    # -------------------------------------------------------------------------
    # Группировка по строкам.
    # -------------------------------------------------------------------------

    rows: list[
        list[tuple[OCRToken, Box]]
    ] = []

    for token, box in selected:
        if not rows:
            rows.append(
                [(token, box)]
            )
            continue

        current = rows[-1]

        row_center = sum(
            item[1].center_y
            for item in current
        ) / len(current)

        tolerance = max(
            8.0,
            box.height * 0.8,
        )

        if abs(
            box.center_y
            - row_center
        ) <= tolerance:
            current.append(
                (token, box)
            )
        else:
            rows.append(
                [(token, box)]
            )

    lines: list[str] = []

    for row in rows:
        row.sort(
            key=lambda item: item[1].left
        )

        line = " ".join(
            token.text.strip()
            for token, _ in row
            if token.text.strip()
        )

        line = _clean_value(
            line
        )

        if line:
            lines.append(line)

    value = "\n".join(lines)

    if not value:
        return FieldResult()

    confidence = sum(
        token.confidence
        for token, _ in selected
    ) / len(selected)

    return FieldResult(
        value=value,
        confidence=max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        ),
        raw_text=value,
    )


# =============================================================================
# FALLBACK COUNTERPARTY
# =============================================================================

def _looks_like_counterparty(
    text: str,
) -> bool:

    normalized = _normalize_search(
        text
    )

    if not normalized:
        return False

    # Юридические формы.
    if re.search(
        r"\b(?:ооо|оао|ао|ип|пao|зао)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True

    # OCR может дать 000 вместо ООО.
    if re.search(
        r"(?:^|[\s.])000(?:$|[\s.])",
        normalized,
        re.IGNORECASE,
    ):
        return True

    return False


def _extract_counterparty_fallback(
    tokens: list[OCRToken],
) -> FieldResult:

    candidates: list[
        tuple[float, OCRToken]
    ] = []

    for token in tokens:
        text = token.text.strip()

        if not text:
            continue

        if not _looks_like_counterparty(
            text
        ):
            continue

        score = (
            token.confidence
            + 0.5
        )

        candidates.append(
            (
                score,
                token,
            )
        )

    if not candidates:
        return FieldResult()

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    token = candidates[0][1]

    return FieldResult(
        value=normalize_counterparty(
            token.text
        ),
        confidence=max(
            0.0,
            min(
                1.0,
                float(token.confidence),
            ),
        ),
        raw_text=token.text,
    )


# =============================================================================
# FALLBACK MANAGER
# =============================================================================

# На практике здесь лучше потом заменить на список сотрудников
# из Google Sheets, когда мы подключим его к боту.
KNOWN_NAMES = {
    "александр",
    "алексей",
    "андрей",
    "анна",
    "артем",
    "виктор",
    "виталий",
    "галина",
    "дарья",
    "денис",
    "дмитрий",
    "евгений",
    "екатерина",
    "елена",
    "иван",
    "игорь",
    "ирина",
    "максим",
    "марина",
    "мария",
    "наталья",
    "николай",
    "олег",
    "павел",
    "роман",
    "светлана",
    "сергей",
    "татьяна",
    "юлия",
}


def _looks_like_person_name(
    text: str,
) -> bool:

    normalized = _normalize_search(
        text
    )

    return normalized in KNOWN_NAMES


def _extract_manager_fallback(
    tokens: list[OCRToken],
) -> FieldResult:

    candidates: list[
        tuple[float, OCRToken]
    ] = []

    for token in tokens:
        text = token.text.strip()

        if not text:
            continue

        if not _looks_like_person_name(
            text
        ):
            continue

        box = _box(token)

        if box is None:
            continue

        # Небольшой приоритет более уверенному OCR.
        score = (
            token.confidence * 100.0
        )

        candidates.append(
            (
                score,
                token,
            )
        )

    if not candidates:
        return FieldResult()

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    token = candidates[0][1]

    return FieldResult(
        value=_clean_value(
            token.text
        ),
        confidence=max(
            0.0,
            min(
                1.0,
                float(token.confidence),
            ),
        ),
        raw_text=token.text,
    )


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_number(
    value: str,
) -> str:

    value = value.upper().strip()

    value = value.replace(
        "О",
        "0",
    )

    value = value.replace(
        "O",
        "0",
    )

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    match = re.search(
        r"(\d{3})/(\d{6})",
        value,
    )

    if not match:
        return value

    return (
        f"{match.group(1)}/"
        f"{match.group(2)}"
    )


def normalize_date(
    value: str,
) -> str:

    value = value.strip()

    value = value.replace(
        "/",
        ".",
    )

    value = value.replace(
        "-",
        ".",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def normalize_phone(
    value: str,
) -> str:

    digits = re.sub(
        r"\D",
        "",
        value,
    )

    if len(digits) != 11:
        return value.strip()

    if digits.startswith("8"):
        digits = (
            "7"
            + digits[1:]
        )

    if not digits.startswith("7"):
        return value.strip()

    return (
        f"+7 ({digits[1:4]}) "
        f"{digits[4:7]}-"
        f"{digits[7:9]}-"
        f"{digits[9:11]}"
    )


def normalize_counterparty(
    value: str,
) -> str:

    value = _normalize_spaces(
        value
    )

    # 000 / ОО0 / 0O0 -> ООО
    value = re.sub(
        r"\b[0оo]{3}\b",
        "ООО",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\s*\.\s*(ООО)\b",
        r" \1",
        value,
        flags=re.IGNORECASE,
    )

    return _clean_value(
        value
    )


# =============================================================================
# PATTERN FALLBACK
# =============================================================================

def _extract_pattern(
    field: str,
    full_text: str,
) -> FieldResult:

    if field == "number":
        match = NUMBER_RE.search(
            full_text
        )

        if not match:
            return FieldResult()

        raw = match.group(1)

        return FieldResult(
            value=normalize_number(raw),
            confidence=0.75,
            raw_text=raw,
        )

    if field == "date":
        match = DATE_RE.search(
            full_text
        )

        if not match:
            return FieldResult()

        raw = match.group(1)

        return FieldResult(
            value=normalize_date(raw),
            confidence=0.75,
            raw_text=raw,
        )

    if field == "phone":
        match = PHONE_RE.search(
            full_text
        )

        if not match:
            return FieldResult()

        raw = match.group(0)

        return FieldResult(
            value=normalize_phone(raw),
            confidence=0.75,
            raw_text=raw,
        )

    return FieldResult()


# =============================================================================
# MAIN
# =============================================================================

def extract_from_full_text(
    tokens: list[OCRToken],
) -> dict[str, FieldResult]:

    fields: dict[str, FieldResult] = {
        "number": FieldResult(),
        "date": FieldResult(),
        "counterparty": FieldResult(),
        "phone": FieldResult(),
        "comment": FieldResult(),
        "manager": FieldResult(),
    }

    if not tokens:
        return fields

    # =========================================================================
    # 1. ИЩЕМ LABELS
    # =========================================================================

    labels = find_labels(tokens)

    logger.info(
        "========== FOUND LABELS =========="
    )

    for label in labels:
        logger.info(
            "LABEL field=%s text=%r box=%s token=%r",
            label.field,
            label.label_text,
            label.box,
            label.token.text,
        )

    logger.info(
        "========== END FOUND LABELS =========="
    )

    # =========================================================================
    # 2. INLINE LABELS
    # =========================================================================

    for field in (
        "number",
        "date",
        "counterparty",
        "phone",
        "manager",
    ):
        field_labels = [
            label
            for label in labels
            if label.field == field
        ]

        for label in field_labels:
            result = _extract_by_label(
                field,
                label,
                tokens,
            )

            if not result.value:
                continue

            fields[field] = result
            break

    # =========================================================================
    # 3. COMMENT
    # =========================================================================

    comment_labels = [
        label
        for label in labels
        if label.field == "comment"
    ]

    # Предпочитаем именно "что сделано".
    comment_labels.sort(
        key=lambda label: (
            0
            if label.label_text == "что сделано"
            else 1
        )
    )

    for label in comment_labels:
        result = _extract_comment(
            label,
            tokens,
        )

        if result.value:
            fields["comment"] = result
            break

    # =========================================================================
    # 4. FULL TEXT
    # =========================================================================

    full_text = " ".join(
        token.text
        for token in tokens
        if token.text.strip()
    )

    # =========================================================================
    # 5. NUMBER
    # =========================================================================

    if fields["number"].value:
        fields["number"] = FieldResult(
            value=normalize_number(
                fields["number"].value
            ),
            confidence=fields["number"].confidence,
            raw_text=fields["number"].raw_text,
        )
    else:
        fields["number"] = _extract_pattern(
            "number",
            full_text,
        )

    # =========================================================================
    # 6. DATE
    # =========================================================================

    if fields["date"].value:
        date_match = DATE_RE.search(
            fields["date"].value
        )

        if date_match:
            fields["date"] = FieldResult(
                value=normalize_date(
                    date_match.group(1)
                ),
                confidence=fields["date"].confidence,
                raw_text=fields["date"].raw_text,
            )
        else:
            fields["date"] = FieldResult()

    if not fields["date"].value:
        fields["date"] = _extract_pattern(
            "date",
            full_text,
        )

    # =========================================================================
    # 7. PHONE
    # =========================================================================

    if fields["phone"].value:
        phone_match = PHONE_RE.search(
            fields["phone"].value
        )

        if phone_match:
            fields["phone"] = FieldResult(
                value=normalize_phone(
                    phone_match.group(0)
                ),
                confidence=fields["phone"].confidence,
                raw_text=fields["phone"].raw_text,
            )
        else:
            fields["phone"] = FieldResult()

    if not fields["phone"].value:
        fields["phone"] = _extract_pattern(
            "phone",
            full_text,
        )

    # =========================================================================
    # 8. COUNTERPARTY FALLBACK
    # =========================================================================

    if not fields["counterparty"].value:
        fields["counterparty"] = (
            _extract_counterparty_fallback(
                tokens
            )
        )

    # =========================================================================
    # 9. MANAGER FALLBACK
    # =========================================================================

    if not fields["manager"].value:
        fields["manager"] = (
            _extract_manager_fallback(
                tokens
            )
        )

    # =========================================================================
    # 10. LOG
    # =========================================================================

    logger.info(
        "EXTRACTION DETAILS: labels=%s",
        [
            {
                "field": label.field,
                "label": label.label_text,
                "token": label.token.text,
                "box": label.box,
            }
            for label in labels
        ],
    )

    return fields