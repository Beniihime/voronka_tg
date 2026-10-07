from __future__ import annotations

import re
from dataclasses import dataclass

from app.ocr.provider import OCRToken
from app.schemas import FieldResult


# =============================================================================
# REGEX
# =============================================================================

NUMBER_RE = re.compile(
    r"(?<!\d)([\dОO]{3}\s*/\s*[\dОOСC]{6})(?!\d)",
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
# "описание" здесь намеренно отсутствует.
#
# На интерфейсе:
#
# Описание: ...
#
# и отдельно:
#
# Что сделано (заполняется исполнителем)
# [большое белое поле]
#
# Нам нужен именно второй блок.

LABELS: dict[str, tuple[str, ...]] = {
    "number": (
        "номер",
        "№",
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
        "дополнительно",
    ),
    "manager": (
        "менеджер",
        "автор",
    ),
}


# Поля, для которых значение ожидается справа от label.
INLINE_FIELDS = {
    "number",
    "date",
    "counterparty",
    "phone",
    "manager",
}


# Поля, для которых значение обычно может находиться ниже label.
BELOW_FIELDS = {
    "comment",
}


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
    """
    Преобразует polygon/rectangle PaddleOCR в единый Box.

    Поддерживает:
        [[x1, y1], [x2, y2], ...]
    и:
        [x1, y1, x2, y2]
    """

    if not token.box:
        return None

    points = token.box

    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
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
# TEXT NORMALIZATION
# =============================================================================

def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_label_text(text: str) -> str:
    """
    Нормализация текста именно для сравнения с названием поля.
    """

    text = text.lower().strip()

    replacements = {
        "ё": "е",
        "—": "-",
        "–": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = _normalize_spaces(text)

    # Убираем двоеточие в конце:
    #
    # Контрагент:
    # Контрагент
    #
    # считаются одинаковыми.
    text = text.strip(" :;,.!?-")

    return text


def _clean_value(text: str) -> str:
    text = text.strip()

    # Убираем label-разделители в начале.
    text = text.lstrip(" :;,-")

    text = _normalize_spaces(text)

    return text.strip()


def _is_empty(text: str) -> bool:
    return not text or not text.strip()


# =============================================================================
# LABEL MATCHING
# =============================================================================

def _exact_label_match(text: str) -> tuple[str, str] | None:
    """
    Проверяет, является ли весь OCR-токен названием поля.
    """

    normalized = _normalize_label_text(text)

    if not normalized:
        return None

    for field, variants in LABELS.items():
        for label in variants:
            if normalized == _normalize_label_text(label):
                return field, label

    return None


def _inline_label_match(text: str) -> tuple[str, str, str] | None:
    """
    Ищет label внутри одного OCR-токена.

    Например:

        "Контрагент: ИП Какула Евгений Владимирович"

    превращается в:

        field = counterparty
        label = Контрагент
        value = ИП Какула Евгений Владимирович

    Также работает для:

        "Номер: 043/008217"
        "Дата: 25.09.2026 9:29:15"
        "Телефон: +7 (962) 030-64-59"
    """

    original = text.strip()

    if not original:
        return None

    normalized = original.lower()

    for field, variants in LABELS.items():
        # Для каждого label сначала пробуем длинные варианты.
        # Например "что сделано" раньше "что".
        sorted_variants = sorted(
            variants,
            key=len,
            reverse=True,
        )

        for label in sorted_variants:
            label_normalized = label.lower()

            # Допускаем:
            #
            # Контрагент:
            # Контрагент :
            # Контрагент - ...
            # Контрагент: ...
            #
            pattern = (
                rf"^\s*{re.escape(label_normalized)}"
                rf"\s*(?::|-|–|—)?\s*(.*)$"
            )

            match = re.match(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            value = match.group(1).strip()

            # Если после label ничего нет, это обычный label.
            if not value:
                return None

            # Возвращаем исходный текст значения,
            # насколько это возможно.
            prefix_match = re.match(
                pattern,
                original,
                flags=re.IGNORECASE,
            )

            if prefix_match:
                value = prefix_match.group(1).strip()

            return field, label, value

    return None


def _is_label_text(text: str) -> bool:
    """
    True, если весь OCR-токен является названием поля.
    """

    return _exact_label_match(text) is not None


def find_labels(tokens: list[OCRToken]) -> list[LabelMatch]:
    """
    Находит отдельные OCR-блоки, являющиеся названиями полей.

    Inline labels типа:
        "Контрагент: ООО Ромашка"

    здесь не возвращаются как отдельный LabelMatch,
    потому что они обрабатываются отдельно в extraction.
    """

    result: list[LabelMatch] = []

    for token in tokens:
        match = _exact_label_match(token.text)

        if match is None:
            continue

        box = _box(token)

        if box is None:
            continue

        field, label_text = match

        result.append(
            LabelMatch(
                field=field,
                token=token,
                box=box,
                label_text=label_text,
            )
        )

    return result


# =============================================================================
# INLINE VALUES
# =============================================================================

def _extract_inline_values(
    tokens: list[OCRToken],
) -> dict[str, FieldResult]:
    """
    Обрабатывает ситуацию, когда PaddleOCR вернул label и value
    одним блоком.

    Например:

        "Контрагент: ИП Какула Евгений Владимирович"

    """

    result: dict[str, FieldResult] = {}

    for token in tokens:
        parsed = _inline_label_match(token.text)

        if parsed is None:
            continue

        field, label, value = parsed

        value = _clean_value(value)

        if not value:
            continue

        # Если поле уже найдено более качественным способом,
        # не перезаписываем его.
        if field in result and result[field].value:
            continue

        confidence = max(
            0.0,
            min(
                1.0,
                float(token.confidence),
            ),
        )

        result[field] = FieldResult(
            value=value,
            confidence=confidence,
            raw_text=token.text,
        )

    return result


# =============================================================================
# LINE GEOMETRY
# =============================================================================

def _same_line(
    label: Box,
    candidate: Box,
) -> bool:
    """
    Проверяет, находятся ли блоки примерно на одной строке.
    """

    vertical_distance = abs(
        candidate.center_y - label.center_y
    )

    reference_height = max(
        label.height,
        candidate.height,
        1.0,
    )

    return vertical_distance <= reference_height * 0.75


def _candidate_to_right(
    label: Box,
    candidate: Box,
) -> bool:
    """
    Кандидат должен находиться справа от label.
    """

    tolerance = max(
        label.height * 0.30,
        2.0,
    )

    return candidate.left >= label.right - tolerance


def _candidate_below(
    label: Box,
    candidate: Box,
) -> bool:
    """
    Проверяет, находится ли блок непосредственно под label.
    """

    if candidate.top < label.bottom:
        return False

    distance = candidate.top - label.bottom

    max_distance = max(
        label.height * 3.0,
        30.0,
    )

    return distance <= max_distance


# =============================================================================
# CANDIDATE GROUPING
# =============================================================================

def _group_inline_candidates(
    label_match: LabelMatch,
    tokens: list[OCRToken],
) -> list[list[OCRToken]]:
    """
    Собирает OCR-блоки справа от label.

    Пример:

        Контрагент   ООО   Ромашка

    станет:

        [ООО, Ромашка]
    """

    label = label_match.box

    candidates: list[OCRToken] = []

    for token in tokens:
        if token is label_match.token:
            continue

        box = _box(token)

        if box is None:
            continue

        if _is_label_text(token.text):
            continue

        if not _candidate_to_right(label, box):
            continue

        if not _same_line(label, box):
            continue

        candidates.append(token)

    candidates.sort(
        key=lambda token: (
            _box(token).left,  # type: ignore[union-attr]
        )
    )

    if not candidates:
        return []

    groups: list[list[OCRToken]] = []

    current: list[OCRToken] = []
    previous_box: Box | None = None

    for token in candidates:
        box = _box(token)

        if box is None:
            continue

        if previous_box is None:
            current = [token]
            previous_box = box
            continue

        gap = box.left - previous_box.right

        max_gap = max(
            label.height * 5.0,
            previous_box.height * 5.0,
            25.0,
        )

        if gap <= max_gap:
            current.append(token)
        else:
            if current:
                groups.append(current)

            current = [token]

        previous_box = box

    if current:
        groups.append(current)

    return groups


# =============================================================================
# COMMENT / MULTILINE FIELD
# =============================================================================

def _collect_comment_tokens(
    label_match: LabelMatch,
    tokens: list[OCRToken],
) -> list[OCRToken]:
    """
    Собирает текст из области под label "Что сделано".

    В отличие от обычных полей здесь значение может состоять
    из нескольких строк.

    Например:

        Что сделано (заполняется исполнителем)

        Обновление 1С-БСХП с релиза 3.0.200.23.
        платформа 8.3.27.1936.
        Обновляется обычно сама, но сейчас не может найти
        файл обновления, обновить нам
    """

    label = label_match.box

    candidates: list[tuple[OCRToken, Box]] = []

    for token in tokens:
        if token is label_match.token:
            continue

        box = _box(token)

        if box is None:
            continue

        if box.top < label.bottom:
            continue

        # Другой label не является текстом комментария.
        if _is_label_text(token.text):
            continue

        # Комментарий должен находиться ниже label.
        vertical_distance = box.top - label.bottom

        max_vertical_distance = max(
            label.height * 20.0,
            250.0,
        )

        if vertical_distance > max_vertical_distance:
            continue

        candidates.append((token, box))

    if not candidates:
        return []

    # Сортируем сначала по строке, затем слева направо.
    candidates.sort(
        key=lambda item: (
            item[1].top,
            item[1].left,
        )
    )

    # -------------------------------------------------------------------------
    # Находим первую строку комментария.
    #
    # Это важно, чтобы не начать собирать произвольный текст из нижней
    # части интерфейса.
    # -------------------------------------------------------------------------

    first_token, first_box = candidates[0]

    row_tolerance = max(
        first_box.height * 0.8,
        label.height * 0.8,
        10.0,
    )

    first_row_y = first_box.center_y

    selected: list[tuple[OCRToken, Box]] = []

    for token, box in candidates:
        if abs(box.center_y - first_row_y) <= row_tolerance:
            selected.append((token, box))
        else:
            break

    # -------------------------------------------------------------------------
    # После первой строки разрешаем последующие строки.
    #
    # Ограничение по горизонтальной области здесь специально мягкое:
    # комментарий может занимать почти всю ширину окна.
    # -------------------------------------------------------------------------

    if selected:
        min_left = min(
            box.left
            for _, box in selected
        )

        max_right = max(
            box.right
            for _, box in selected
        )
    else:
        min_left = label.left
        max_right = label.right

    last_row_y = first_row_y

    for token, box in candidates[len(selected):]:
        vertical_gap = box.top - last_row_y

        max_row_gap = max(
            box.height * 2.5,
            label.height * 3.0,
            40.0,
        )

        if vertical_gap > max_row_gap:
            break

        # Если блок находится совсем далеко сбоку от области комментария,
        # считаем его менее вероятным.
        horizontal_too_far = (
            box.right < min_left - label.width * 2.0
            or box.left > max_right + label.width * 20.0
        )

        if horizontal_too_far:
            continue

        selected.append((token, box))
        last_row_y = max(
            last_row_y,
            box.center_y,
        )

        min_left = min(
            min_left,
            box.left,
        )

        max_right = max(
            max_right,
            box.right,
        )

    return [
        token
        for token, _ in selected
    ]


def _merge_multiline_tokens(
    tokens: list[OCRToken],
) -> str:
    """
    Объединяет OCR-блоки в многострочный текст.

    OCR-токены группируются по строкам.
    """

    if not tokens:
        return ""

    prepared: list[tuple[OCRToken, Box]] = []

    for token in tokens:
        box = _box(token)

        if box is None:
            continue

        if not token.text.strip():
            continue

        prepared.append((token, box))

    if not prepared:
        return ""

    prepared.sort(
        key=lambda item: (
            item[1].top,
            item[1].left,
        )
    )

    rows: list[list[tuple[OCRToken, Box]]] = []

    for token, box in prepared:
        if not rows:
            rows.append([(token, box)])
            continue

        current_row = rows[-1]

        row_center = sum(
            item[1].center_y
            for item in current_row
        ) / len(current_row)

        tolerance = max(
            box.height * 0.7,
            10.0,
        )

        if abs(box.center_y - row_center) <= tolerance:
            current_row.append((token, box))
        else:
            rows.append([(token, box)])

    result_lines: list[str] = []

    for row in rows:
        row.sort(
            key=lambda item: item[1].left
        )

        line = " ".join(
            token.text.strip()
            for token, _ in row
            if token.text.strip()
        )

        line = _clean_value(line)

        if line:
            result_lines.append(line)

    return "\n".join(result_lines)


# =============================================================================
# SCORING
# =============================================================================

def _looks_like_number(text: str) -> bool:
    return NUMBER_RE.search(text) is not None


def _looks_like_date(text: str) -> bool:
    return DATE_RE.search(text) is not None


def _looks_like_phone(text: str) -> bool:
    return PHONE_RE.search(text) is not None


def _field_format_score(
    field: str,
    text: str,
) -> float:
    """
    Дополнительный score за соответствие ожидаемому формату.
    """

    if not text:
        return 0.0

    if field == "number":
        return 50.0 if _looks_like_number(text) else 0.0

    if field == "date":
        return 50.0 if _looks_like_date(text) else 0.0

    if field == "phone":
        return 50.0 if _looks_like_phone(text) else 0.0

    return 0.0


def _score_inline_candidate(
    field: str,
    label: LabelMatch,
    tokens: list[OCRToken],
) -> float:
    """
    Score группы OCR-токенов справа от label.
    """

    if not tokens:
        return float("-inf")

    boxes = [
        _box(token)
        for token in tokens
    ]

    boxes = [
        box
        for box in boxes
        if box is not None
    ]

    if not boxes:
        return float("-inf")

    candidate_left = min(
        box.left
        for box in boxes
    )

    candidate_center_y = (
        min(box.top for box in boxes)
        + max(box.bottom for box in boxes)
    ) / 2.0

    label_box = label.box

    score = 0.0

    # -------------------------------------------------------------------------
    # Близость по горизонтали
    # -------------------------------------------------------------------------

    horizontal_distance = max(
        0.0,
        candidate_left - label_box.right,
    )

    normalized_distance = (
        horizontal_distance
        / max(label_box.height, 1.0)
    )

    score += max(
        0.0,
        35.0 - normalized_distance * 4.0,
    )

    # -------------------------------------------------------------------------
    # Совпадение строки
    # -------------------------------------------------------------------------

    vertical_distance = abs(
        candidate_center_y
        - label_box.center_y
    )

    normalized_vertical = (
        vertical_distance
        / max(label_box.height, 1.0)
    )

    score += max(
        0.0,
        35.0 - normalized_vertical * 20.0,
    )

    # -------------------------------------------------------------------------
    # Confidence OCR
    # -------------------------------------------------------------------------

    avg_confidence = sum(
        max(0.0, min(1.0, token.confidence))
        for token in tokens
    ) / len(tokens)

    score += avg_confidence * 20.0

    # -------------------------------------------------------------------------
    # Формат поля
    # -------------------------------------------------------------------------

    text = " ".join(
        token.text
        for token in tokens
    )

    score += _field_format_score(
        field,
        text,
    )

    return score


# =============================================================================
# EXTRACT VALUE BY LABEL
# =============================================================================

def extract_labeled_value(
    field: str,
    label_match: LabelMatch,
    tokens: list[OCRToken],
) -> FieldResult:
    """
    Извлекает значение обычного inline-поля.

    Приоритет:

        1. значение справа;
        2. значение на той же строке;
        3. fallback ниже.
    """

    groups = _group_inline_candidates(
        label_match,
        tokens,
    )

    best_group: list[OCRToken] = []
    best_score = float("-inf")

    for group in groups:
        score = _score_inline_candidate(
            field,
            label_match,
            group,
        )

        if score > best_score:
            best_score = score
            best_group = group

    if not best_group:
        return FieldResult()

    value = _clean_value(
        " ".join(
            token.text
            for token in best_group
        )
    )

    if not value:
        return FieldResult()

    confidence = sum(
        token.confidence
        for token in best_group
    ) / len(best_group)

    return FieldResult(
        value=value,
        confidence=max(
            0.0,
            min(1.0, confidence),
        ),
        raw_text=value,
    )


# =============================================================================
# PATTERN EXTRACTION
# =============================================================================

def extract_pattern(
    field: str,
    text: str,
) -> FieldResult:
    """
    Последний fallback.

    Используется только по уже распознанному OCR-тексту.
    Нового OCR здесь нет.
    """

    if field == "number":
        match = NUMBER_RE.search(text)

        if not match:
            return FieldResult()

        raw = match.group(1)

        return FieldResult(
            value=normalize_number(raw),
            confidence=0.75,
            raw_text=raw,
        )

    if field == "date":
        match = DATE_RE.search(text)

        if not match:
            return FieldResult()

        raw = match.group(1)

        return FieldResult(
            value=normalize_date(raw),
            confidence=0.75,
            raw_text=raw,
        )

    if field == "phone":
        match = PHONE_RE.search(text)

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
# NORMALIZATION
# =============================================================================

def normalize_number(value: str) -> str:
    """
    Приводит номер к:

        043/008217
    """

    value = value.upper().strip()

    # Частые ошибки OCR.
    value = value.replace("О", "0")
    value = value.replace("O", "0")

    # Убираем пробелы.
    value = re.sub(r"\s+", "", value)

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


def normalize_date(value: str) -> str:
    """
    Приводит дату к единому виду:

        25.09.2026 9:29:15
    """

    value = value.strip()

    value = value.replace("/", ".")
    value = value.replace("-", ".")

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def normalize_phone(value: str) -> str:
    """
    Приводит:

        89620306459
        +7 962 030-64-59
        8 (962) 030-64-59

    к:

        +7 (962) 030-64-59
    """

    digits = re.sub(
        r"\D",
        "",
        value,
    )

    if len(digits) != 11:
        return value.strip()

    if digits.startswith("8"):
        digits = "7" + digits[1:]

    if not digits.startswith("7"):
        return value.strip()

    return (
        f"+7 ({digits[1:4]}) "
        f"{digits[4:7]}-"
        f"{digits[7:9]}-"
        f"{digits[9:11]}"
    )


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_from_full_text(
    tokens: list[OCRToken],
) -> dict[str, FieldResult]:
    """
    Главная функция извлечения данных.

    ВАЖНО:

        OCR здесь НЕ выполняется.

    На вход поступает результат единственного вызова PaddleOCR:

        OCRToken[]
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
    # 1. INLINE LABEL + VALUE В ОДНОМ OCR-БЛОКЕ
    # =========================================================================
    #
    # Например:
    #
    # "Контрагент: ИП Какула Евгений Владимирович"
    #
    # Это должно иметь высокий приоритет.
    # =========================================================================

    inline_values = _extract_inline_values(tokens)

    for field, result in inline_values.items():
        fields[field] = result

    # =========================================================================
    # 2. ОТДЕЛЬНЫЕ LABEL
    # =========================================================================

    labels = find_labels(tokens)

    # Если label встретился несколько раз,
    # сохраняем первый.
    labels_by_field: dict[str, LabelMatch] = {}

    for label in labels:
        if label.field not in labels_by_field:
            labels_by_field[label.field] = label

    # =========================================================================
    # 3. ОБЫЧНЫЕ INLINE-ПОЛЯ
    # =========================================================================

    for field in INLINE_FIELDS:
        # Если уже нашли значение внутри одного OCR-блока,
        # повторно не ищем.
        if fields[field].value:
            continue

        label = labels_by_field.get(field)

        if label is None:
            continue

        fields[field] = extract_labeled_value(
            field,
            label,
            tokens,
        )

    # =========================================================================
    # 4. COMMENT
    # =========================================================================
    #
    # Для комментария отдельная логика.
    #
    # Ищем:
    #
    # Что сделано (заполняется исполнителем)
    #
    # и затем собираем текст ПОД ним.
    # =========================================================================

    comment_label = labels_by_field.get("comment")

    if comment_label is not None:
        comment_tokens = _collect_comment_tokens(
            comment_label,
            tokens,
        )

        comment_text = _merge_multiline_tokens(
            comment_tokens,
        )

        if comment_text:
            confidence = sum(
                token.confidence
                for token in comment_tokens
            ) / len(comment_tokens)

            fields["comment"] = FieldResult(
                value=comment_text,
                confidence=max(
                    0.0,
                    min(1.0, confidence),
                ),
                raw_text=comment_text,
            )

    # =========================================================================
    # 5. FULL OCR TEXT
    # =========================================================================
    #
    # Это всё ещё тот же самый OCR.
    #
    # Мы просто собираем уже полученные токены в строку.
    # =========================================================================

    full_text = " ".join(
        token.text
        for token in tokens
        if token.text.strip()
    )

    # =========================================================================
    # 6. FALLBACK ДЛЯ NUMBER
    # =========================================================================

    if not fields["number"].value:
        fields["number"] = extract_pattern(
            "number",
            full_text,
        )
    else:
        normalized = normalize_number(
            fields["number"].value
        )

        if _looks_like_number(normalized):
            fields["number"] = FieldResult(
                value=normalized,
                confidence=min(
                    1.0,
                    fields["number"].confidence + 0.10,
                ),
                raw_text=fields["number"].raw_text,
            )

    # =========================================================================
    # 7. FALLBACK ДЛЯ DATE
    # =========================================================================

    if not fields["date"].value:
        fields["date"] = extract_pattern(
            "date",
            full_text,
        )
    else:
        normalized = normalize_date(
            fields["date"].value
        )

        fields["date"] = FieldResult(
            value=normalized,
            confidence=fields["date"].confidence,
            raw_text=fields["date"].raw_text,
        )

    # =========================================================================
    # 8. FALLBACK ДЛЯ PHONE
    # =========================================================================

    if not fields["phone"].value:
        fields["phone"] = extract_pattern(
            "phone",
            full_text,
        )
    else:
        normalized = normalize_phone(
            fields["phone"].value
        )

        fields["phone"] = FieldResult(
            value=normalized,
            confidence=fields["phone"].confidence,
            raw_text=fields["phone"].raw_text,
        )

    return fields