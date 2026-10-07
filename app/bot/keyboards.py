import uuid

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)


FIELDS = {
    "number": "Номер",
    "date": "Дата",
    "counterparty": "Контрагент",
    "phone": "Телефон",
    "comment": "Комментарий",
    "manager": "Менеджер",
    "task_status": "Статус задачи",
}


def review_keyboard(
    job_id: uuid.UUID,
) -> InlineKeyboardMarkup:
    value = str(job_id)

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="👤 Выбрать менеджера",
                    callback_data=f"select:manager:{value}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="📋 Выбрать статус",
                    callback_data=f"select:task_status:{value}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="✅ Всё верно",
                    callback_data=f"confirm:{value}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="✏️ Исправить",
                    callback_data=f"edit:{value}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data=f"cancel:{value}",
                )
            ],
        ]
    )


def dropdown_keyboard(
    job_id: uuid.UUID,
    field_name: str,
    options: list[str],
) -> InlineKeyboardMarkup:
    value = str(job_id)

    rows = []

    for index, option in enumerate(options):
        rows.append(
            [
                InlineKeyboardButton(
                    text=option[:64],
                    callback_data=(
                        f"option:{field_name}:"
                        f"{value}:{index}"
                    ),
                )
            ]
        )

    rows.append(
        [
            InlineKeyboardButton(
                text="⬅️ Назад",
                callback_data=f"dropdown_back:{value}",
            )
        ]
    )

    return InlineKeyboardMarkup(
        inline_keyboard=rows
    )


def fields_keyboard(
    job_id: uuid.UUID,
) -> InlineKeyboardMarkup:
    value = str(job_id)

    rows = [
        [
            InlineKeyboardButton(
                text=label,
                callback_data=(
                    f"field:{value}:{name}"
                ),
            )
        ]
        for name, label in FIELDS.items()
    ]

    rows.append(
        [
            InlineKeyboardButton(
                text="Назад",
                callback_data=f"back:{value}",
            )
        ]
    )

    return InlineKeyboardMarkup(
        inline_keyboard=rows
    )


def duplicate_keyboard(
    job_id: uuid.UUID,
) -> InlineKeyboardMarkup:
    value = str(job_id)

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Добавить всё равно",
                    callback_data=f"force:{value}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="Отмена",
                    callback_data=f"cancel:{value}",
                )
            ],
        ]
    )