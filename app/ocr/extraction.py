from __future__ import annotations

import logging
import re
from dataclasses import dataclass

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


# Поддерживает:
#
# 07.10.2026
# 07.10.2026 12:06
# 07.10.2026 12:06:40
# 07.10.202612-06-40
# 07.10.202612:06:40
#
DATE_RE = re.compile(
    r"(?<!\d)"
    r"(\d{2}[.\-/]\d{2}[.\-/]\d{4})"
    r"(?:"
    r"\s*"
    r"(\d{1,2})"
    r"[:\-]"
    r"(\d{2})"
    r"(?:[:\-](\d{2}))?"
    r")?"
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


# Менеджер ВСЕГДА берётся из:
#
# Автор: Аксёнова Виктория Алексеевна:
# Редактор: Давыдова Анастасия Сергеевна
#
AUTHOR_RE = re.compile(
    r"(?:^|\s)"
    r"автор"
    r"\s*[:\-]?\s*"
    r"(.*?)"
    r"(?=\s*[:\-]?\s*редактор\s*[:\-]?)"
    ,
    re.IGNORECASE,
)


# =============================================================================
# LABELS
# =============================================================================

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
        "автор",
        "менеджер",
        "ответственный",
    ),
}


# OCR может искажать отдельные labels.
#
# ВАЖНО:
# сюда не добавляем слишком общие варианты,
# иначе снова появятся ложные совпадения.
OCR_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "date": (
        "aта",
        "дaта",
        "aта",
    ),
    "phone": (
        "тел",
        "тeл",
        "тeлeфон",
    ),
    "manager": (
        "автор",
        "aвтор",
        "менеджер",
        "менeджер",
    ),
}


# =============================================================================
# UI / COMMENT STOP MARKERS
# =============================================================================

# Верхнее меню.
#
# Если OCR-токен содержит несколько этих слов,
# то "Дополнительно" в нём НЕ считается comment label.
TOP_MENU_MARKERS = (
    "провести",
    "создать",
    "основании",
    "печать",
    "перейти",
    "уведомить",
)


# После комментария начинается таблица.
#
# Всё, что находится начиная с этих заголовков,
# в comment больше не попадает.
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
        return max(
            0.0,
            self.right - self.left,
        )

    @property
    def height(self) -> float:
        return max(
            0.0,
            self.bottom - self.top,
        )

    @property
    def center_x(self) -> float:
        return (
            self.left
            + self.right
        ) / 2.0

    @property
    def center_y(self) -> float:
        return (
            self.top
            + self.bottom
        ) / 2.0


@dataclass(frozen=True)
class LabelMatch:
    field: str
    token: OCRToken
    box: Box
    label_text: str


def _box(
    token: OCRToken,
) -> Box | None:

    if not token.box:
        return None

    try:
        xs = [
            float(point[0])
            for point in token.box
        ]

        ys = [
            float(point[1])
            for point in token.box
        ]

    except (
        TypeError,
        ValueError,
        IndexError,
    ):
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
# TEXT NORMALIZATION
# =============================================================================

def _normalize_spaces(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _normalize_homoglyphs(
    text: str,
) -> str:
    """
    Исправляет типичные латинские символы,
    которые OCR принимает за кириллицу.
    """

    replacements = {
        "A": "А",
        "a": "а",
        "B": "В",
        "c": "с",
        "C": "С",
        "e": "е",
        "E": "Е",
        "o": "о",
        "O": "О",
        "p": "р",
        "P": "Р",
        "x": "х",
        "X": "Х",
        "y": "у",
        "K": "К",
        "M": "М",
        "T": "Т",
    }

    return "".join(
        replacements.get(
            char,
            char,
        )
        for char in text
    )


def _normalize_search(
    text: str,
) -> str:

    text = _normalize_homoglyphs(
        text
    )

    text = text.lower()

    replacements = {
        "ё": "е",
        "—": "-",
        "–": "-",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    return _normalize_spaces(
        text
    )


def _normalize_compact(
    text: str,
) -> str:

    text = _normalize_search(
        text
    )

    return re.sub(
        r"[^а-яa-z0-9]+",
        "",
        text,
    )


def _clean_value(
    text: str,
) -> str:

    text = text.strip()

    text = text.lstrip(
        " :;,-–—"
    )

    text = _normalize_spaces(
        text
    )

    return text.strip()


# =============================================================================
# FORMAT CHECKS
# =============================================================================

def _looks_like_number(
    text: str,
) -> bool:

    return (
        NUMBER_RE.search(text)
        is not None
    )


def _looks_like_date(
    text: str,
) -> bool:

    return (
        DATE_RE.search(text)
        is not None
    )


def _looks_like_phone(
    text: str,
) -> bool:

    return (
        PHONE_RE.search(text)
        is not None
    )


# =============================================================================
# LABEL MATCHING
# =============================================================================

def _is_top_menu_token(
    text: str,
) -> bool:

    normalized = _normalize_search(
        text
    )

    matches = 0

    for marker in TOP_MENU_MARKERS:
        if marker in normalized:
            matches += 1

    return matches >= 2


def _exact_label_match(
    text: str,
) -> tuple[str, str] | None:

    normalized = _normalize_search(
        text
    )

    if not normalized:
        return None

    for field, labels in LABELS.items():
        for label in labels:
            if (
                normalized
                == _normalize_search(label)
            ):
                return field, label

    for field, aliases in OCR_LABEL_ALIASES.items():
        for alias in aliases:
            if (
                normalized
                == _normalize_search(alias)
            ):
                return field, alias

    return None


def _embedded_label_matches(
    token: OCRToken,
) -> list[tuple[str, str]]:
    """
    Ищет labels внутри большого OCR-блока.

    Например:

        Акты /Отчеты сотрудникаЧто сделано
        (заполняется исполнителем)Ремонт оборудования
        КомментарийДополнительно

    """

    text = token.text

    if not text.strip():
        return []

    if _is_top_menu_token(text):
        return []

    normalized = _normalize_search(
        text
    )

    result: list[
        tuple[str, str]
    ] = []

    # -------------------------------------------------------------------------
    # Обычные labels.
    # -------------------------------------------------------------------------

    for field, labels in LABELS.items():
        for label in labels:
            label_normalized = (
                _normalize_search(label)
            )

            # Однобуквенные / слишком короткие варианты
            # принципиально не ищем внутри длинного текста.
            if len(label_normalized) < 4:
                continue

            if (
                label_normalized
                not in normalized
            ):
                continue

            # "Дата сдачи отзыва" не является реквизитом "Дата".
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
        # 1. Полный OCR token == label.
        # ---------------------------------------------------------------------

        exact = _exact_label_match(
            text
        )

        if exact is not None:
            field, label = exact

            # "Дата сдачи отзыва" не принимаем.
            if (
                field == "date"
                and "сдачи"
                in _normalize_search(text)
            ):
                continue

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
        # 2. Label находится внутри большого OCR token.
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
# DATE EXTRACTION
# =============================================================================

def _extract_date_from_labeled_token(
    token: OCRToken,
) -> FieldResult:
    """
    Дата имеет отдельное правило.

    Разрешены варианты:

        Дата: 07.10.2026 12:06:40
        Дата:07.10.2026 12:06:40
        Дaта:07.10.202612-06-40
    """

    original = token.text.strip()

    if not original:
        return FieldResult()

    normalized = _normalize_homoglyphs(
        original
    )

    normalized_lower = normalized.lower()

    if not re.match(
        r"^\s*дата\s*[:\-]?",
        normalized_lower,
    ):
        return FieldResult()

    match = DATE_RE.search(
        normalized_lower
    )

    if not match:
        return FieldResult()

    raw = match.group(0)

    return FieldResult(
        value=normalize_date(
            raw
        ),
        confidence=max(
            0.0,
            min(
                1.0,
                float(token.confidence),
            ),
        ),
        raw_text=original,
    )


def _extract_date(
    tokens: list[OCRToken],
) -> FieldResult:
    """
    Приоритет:

        1. OCR token, начинающийся с "Дата".
        2. Только затем общий regex fallback.

    Это защищает от старых дат в истории/связанных документах.
    """

    for token in tokens:
        result = _extract_date_from_labeled_token(
            token
        )

        if result.value:
            return result

    return FieldResult()


# =============================================================================
# INLINE VALUE
# =============================================================================

def _extract_inline_value_same_token(
    field: str,
    token: OCRToken,
) -> FieldResult:

    text = token.text.strip()

    if not text:
        return FieldResult()

    normalized = _normalize_search(
        text
    )

    labels = list(
        LABELS.get(
            field,
            (),
        )
    )

    labels.extend(
        OCR_LABEL_ALIASES.get(
            field,
            (),
        )
    )

    for label in sorted(
        labels,
        key=len,
        reverse=True,
    ):
        label_normalized = (
            _normalize_search(label)
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

        if (
            field == "number"
            and not _looks_like_number(value)
        ):
            continue

        if (
            field == "date"
            and not _looks_like_date(value)
        ):
            continue

        if (
            field == "phone"
            and not _looks_like_phone(value)
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
# SPATIAL EXTRACTION
# =============================================================================

def _same_line(
    label_box: Box,
    candidate_box: Box,
) -> bool:

    vertical_distance = abs(
        candidate_box.center_y
        - label_box.center_y
    )

    reference_height = max(
        label_box.height,
        candidate_box.height,
        1.0,
    )

    return (
        vertical_distance
        <= reference_height * 1.5
    )


def _find_right_candidates(
    label: LabelMatch,
    tokens: list[OCRToken],
) -> list[OCRToken]:

    candidates: list[
        tuple[float, OCRToken]
    ] = []

    for token in tokens:

        if token is label.token:
            continue

        text = token.text.strip()

        if not text:
            continue

        box = _box(token)

        if box is None:
            continue

        # Не используем другой label как значение.
        if (
            _exact_label_match(text)
            is not None
        ):
            continue

        # Значение должно быть справа.
        if (
            box.left
            < label.box.right
        ):
            continue

        # И примерно на той же строке.
        if not _same_line(
            label.box,
            box,
        ):
            continue

        horizontal_distance = (
            box.left
            - label.box.right
        )

        vertical_distance = abs(
            box.center_y
            - label.box.center_y
        )

        # Чем ближе справа и чем лучше
        # совпадает строка — тем выше кандидат.
        score = (
            horizontal_distance
            + vertical_distance * 3.0
        )

        candidates.append(
            (
                score,
                token,
            )
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    return [
        token
        for _, token in candidates
    ]


def _extract_by_label(
    field: str,
    label: LabelMatch,
    tokens: list[OCRToken],
) -> FieldResult:

    # -------------------------------------------------------------------------
    # 1. Label + value в одном OCR token.
    # -------------------------------------------------------------------------

    same_token = (
        _extract_inline_value_same_token(
            field,
            label.token,
        )
    )

    if same_token.value:
        return same_token

    # -------------------------------------------------------------------------
    # 2. Значение справа.
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

        if (
            field == "number"
            and not _looks_like_number(value)
        ):
            continue

        if (
            field == "date"
            and not _looks_like_date(value)
        ):
            continue

        if (
            field == "phone"
            and not _looks_like_phone(value)
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
# COMMENT
# =============================================================================

def _is_comment_stop(
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

    return False


def _extract_comment(
    label: LabelMatch,
    tokens: list[OCRToken],
) -> FieldResult:
    """
    Берёт строки ниже "Что сделано".

    Прекращает сбор при начале нижней таблицы.
    """

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

        if (
            box.top
            <= label.box.bottom
        ):
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

    selected: list[
        tuple[OCRToken, Box]
    ] = []

    for token, box in candidates:

        # Началась нижняя таблица.
        if _is_comment_stop(
            token.text
        ):
            break

        # Если появился новый UI label
        # не относящийся к comment,
        # тоже прекращаем сбор.
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
    # Группируем OCR tokens по строкам.
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

        current_row = rows[-1]

        row_center = sum(
            item[1].center_y
            for item in current_row
        ) / len(current_row)

        tolerance = max(
            8.0,
            box.height * 0.8,
        )

        if abs(
            box.center_y
            - row_center
        ) <= tolerance:

            current_row.append(
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

    value = "\n".join(
        lines
    )

    if not value:
        return FieldResult()

    confidence = (
        sum(
            float(token.confidence)
            for token, _ in selected
        )
        / len(selected)
    )

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
# COUNTERPARTY
# =============================================================================

def _looks_like_counterparty(
    text: str,
) -> bool:
    """
    Fallback для случая, когда "Контрагент" OCR не распознал.

    Например:

        ПромТЭК. 000
        ПромТЭК ООО
        ИП Иванов Иван Иванович
        АО Ромашка
    """

    normalized = _normalize_search(
        text
    )

    if not normalized:
        return False

    # Слишком длинный блок не является хорошим
    # названием контрагента.
    if len(normalized) > 100:
        return False

    # ООО / 000 в конце.
    if re.search(
        r"(?:^|[\s.,])"
        r"(?:ооо|000|оо0|0оо)"
        r"\s*$",
        normalized,
        re.IGNORECASE,
    ):
        return True

    # ИП только в начале.
    if re.match(
        r"^ип(?:\s|\.|$)",
        normalized,
        re.IGNORECASE,
    ):
        return True

    # АО / ОАО / ЗАО только в начале.
    if re.match(
        r"^(?:ао|оао|зао)"
        r"(?:\s|\.|$)",
        normalized,
        re.IGNORECASE,
    ):
        return True

    return False


def normalize_counterparty(
    value: str,
) -> str:

    value = _normalize_spaces(
        value
    )

    value = _normalize_homoglyphs(
        value
    )

    # OCR:
    #
    # 000
    # ОО0
    # 0ОО
    # OOO
    #
    # -> ООО
    value = re.sub(
        r"(?:000|оо0|0оо|оoо|ooo)$",
        "ООО",
        value,
        flags=re.IGNORECASE,
    )

    # ПромТЭК. 000
    # ПромТЭК. ООО
    # ПромТЭК . ООО
    #
    # -> ПромТЭК ООО
    value = re.sub(
        r"\s*\.\s*ООО\s*$",
        " ООО",
        value,
        flags=re.IGNORECASE,
    )

    return _clean_value(
        value
    )


def _extract_counterparty_fallback(
    tokens: list[OCRToken],
) -> FieldResult:

    candidates: list[
        tuple[float, OCRToken]
    ] = []

    for token in tokens:

        if not token.text.strip():
            continue

        if not _looks_like_counterparty(
            token.text
        ):
            continue

        # Приоритет confidence OCR.
        score = float(
            token.confidence
        )

        # Немного предпочитаем ООО / 000,
        # так как это наиболее частый случай.
        normalized = _normalize_search(
            token.text
        )

        if re.search(
            r"(?:000|ооо|оо0|0оо)\s*$",
            normalized,
            re.IGNORECASE,
        ):
            score += 0.5

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
# MANAGER = AUTHOR
# =============================================================================

def _extract_manager_from_author(
    tokens: list[OCRToken],
) -> FieldResult:
    """
    Менеджер берётся ТОЛЬКО из реквизита "Автор".

    Пример:

        Автор: Аксёнова Виктория Алексеевна:
        Редактор: Давыдова Анастасия Сергеевна

    Результат:

        Аксёнова Виктория Алексеевна
    """

    if not tokens:
        return FieldResult()

    full_text = " ".join(
        token.text.strip()
        for token in tokens
        if token.text.strip()
    )

    normalized = _normalize_homoglyphs(
        full_text
    )

    match = AUTHOR_RE.search(
        normalized
    )

    if not match:
        logger.info(
            "MANAGER: Автор не найден в OCR"
        )

        return FieldResult()

    value = _clean_value(
        match.group(1)
    )

    # Убираем возможный двоеточие перед "Редактор".
    value = value.rstrip(
        " :;-"
    ).strip()

    if not value:
        return FieldResult()

    # -------------------------------------------------------------------------
    # Confidence.
    #
    # Ищем OCR token, в котором встречается имя автора.
    # -------------------------------------------------------------------------

    confidence_values: list[float] = []

    normalized_value = (
        _normalize_search(value)
    )

    for token in tokens:

        token_normalized = (
            _normalize_search(
                token.text
            )
        )

        if (
            normalized_value
            and normalized_value
            in token_normalized
        ):
            confidence_values.append(
                float(token.confidence)
            )

    if confidence_values:
        confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )
    else:
        confidence = 0.80

    return FieldResult(
        value=value,
        confidence=max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        ),
        raw_text=match.group(0),
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

    match = DATE_RE.search(
        value
    )

    if not match:
        return value

    date_part = match.group(1)

    hour = match.group(2)
    minute = match.group(3)
    second = match.group(4)

    date_part = date_part.replace(
        "/",
        ".",
    )

    date_part = date_part.replace(
        "-",
        ".",
    )

    if hour and minute:

        if second:
            return (
                f"{date_part} "
                f"{hour}:{minute}:{second}"
            )

        return (
            f"{date_part} "
            f"{hour}:{minute}"
        )

    return date_part


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
            value=normalize_number(
                raw
            ),
            confidence=0.75,
            raw_text=raw,
        )

    if field == "date":

        match = DATE_RE.search(
            full_text
        )

        if not match:
            return FieldResult()

        raw = match.group(0)

        return FieldResult(
            value=normalize_date(
                raw
            ),
            confidence=0.70,
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
            value=normalize_phone(
                raw
            ),
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
    """
    Один OCR -> extraction.

    Никаких дополнительных OCR проходов.

    Основной принцип:

        label -> значение рядом

    Специальные поля:

        date        -> реквизит "Дата:"
        manager     -> "Автор:" ... "Редактор:"
        comment     -> ниже "Что сделано"
        counterparty -> label либо fallback по названию организации
    """

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
    # 1. LABELS
    # =========================================================================

    labels = find_labels(
        tokens
    )

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
    # 2. DATE
    #
    # ВАЖНО:
    # дата сначала ищется по реквизиту "Дата:".
    # =========================================================================

    fields["date"] = _extract_date(
        tokens
    )

    # =========================================================================
    # 3. Обычные inline-поля.
    #
    # Номер здесь можно найти через label "Номер",
    # хотя чаще он будет найден regex fallback.
    #
    # Контрагент / телефон ищутся по label,
    # если label OCR действительно увидел.
    # =========================================================================

    for field in (
        "number",
        "counterparty",
        "phone",
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
    # 4. COMMENT
    #
    # Если найдено несколько comment labels,
    # предпочитаем "Что сделано".
    # =========================================================================

    comment_labels = [
        label
        for label in labels
        if label.field == "comment"
    ]

    comment_labels.sort(
        key=lambda label: (
            0
            if label.label_text
            == "что сделано"
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
    # 5. FULL OCR TEXT
    # =========================================================================

    full_text = " ".join(
        token.text
        for token in tokens
        if token.text.strip()
    )

    # =========================================================================
    # 6. NUMBER FALLBACK
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
    # 7. DATE FALLBACK
    #
    # До сюда попадём только если "Дата:" вообще не нашлась.
    # =========================================================================

    if not fields["date"].value:

        fields["date"] = _extract_pattern(
            "date",
            full_text,
        )

    # =========================================================================
    # 8. PHONE FALLBACK
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
    # 9. COUNTERPARTY FALLBACK
    #
    # Только если label "Контрагент" не дал нормальное значение.
    # =========================================================================

    if fields["counterparty"].value:

        fields["counterparty"] = FieldResult(
            value=normalize_counterparty(
                fields["counterparty"].value
            ),
            confidence=fields["counterparty"].confidence,
            raw_text=fields["counterparty"].raw_text,
        )

    else:

        fields["counterparty"] = (
            _extract_counterparty_fallback(
                tokens
            )
        )

    # =========================================================================
    # 10. MANAGER
    #
    # НИКАКИХ:
    #
    #     Татьяна
    #     известные имена
    #     ближайший человек
    #
    # Только:
    #
    #     Автор: ... Редактор:
    # =========================================================================

    fields["manager"] = (
        _extract_manager_from_author(
            tokens
        )
    )

    # =========================================================================
    # 11. DEBUG
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