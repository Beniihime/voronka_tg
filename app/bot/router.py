import asyncio
import logging
import uuid
from pathlib import Path

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from app.bot.keyboards import (
    FIELDS,
    duplicate_keyboard,
    fields_keyboard,
    review_keyboard,
    dropdown_keyboard,
)
from app.bot.states import EditField
from app.config import get_settings
from app.db.models import JobStatus
from app.db.session import SessionLocal
from app.ocr.pipeline import RecognitionPipeline
from app.schemas import RecognitionResult
from app.services.google_sheets import GoogleSheetsService
from app.services.repository import JobRepository
from app.services.validation import ValidationError, validate_result

logger = logging.getLogger(__name__)
router = Router()
settings = get_settings()
# PaddleOCR loads several native models. Keep one engine for the whole bot
# process instead of allocating a new model set for every uploaded image.
recognition_pipeline = RecognitionPipeline()
recognition_lock = asyncio.Lock()


def format_card(result: RecognitionResult, has_low_confidence: bool = False) -> str:
    notice = "\n\n⚠️ Некоторые поля распознаны с низкой уверенностью. Проверьте данные." if has_low_confidence else ""
    return (
        "Проверьте распознанные данные:\n\n"
        f"Номер: {result.number.value or '—'}\n"
        f"Дата: {result.date.value or '—'}\n"
        f"Контрагент: {result.counterparty.value or '—'}\n"
        f"Телефон: {result.phone.value or '—'}\n"
        f"Менеджер: {result.manager.value or '—'}\n"
        f"Статус задачи: {result.task_status.value or '—'}\n"
        f"Комментарий: {result.comment.value or '—'}"
        f"{notice}"
    )


def parse_job_id(data: str) -> uuid.UUID:
    return uuid.UUID(data.split(":")[1])


async def load_owned_job(job_id: uuid.UUID, user_id: int):
    session = SessionLocal()
    repo = JobRepository(session)
    job = await repo.get_job(job_id, user_id)
    return session, repo, job


async def recognize_image(image_path: Path) -> RecognitionResult:
    """Serialise native OCR access and reuse its loaded models."""
    async with recognition_lock:
        return await asyncio.to_thread(recognition_pipeline.recognize, image_path)


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    if message.from_user is None or message.bot is None:
        return
    await message.answer("Изображение получено. Распознаю данные...")
    settings.image_storage_path.mkdir(parents=True, exist_ok=True)
    image_path = settings.image_storage_path / f"{message.from_user.id}_{message.photo[-1].file_unique_id}.jpg"
    job_id: uuid.UUID | None = None
    try:
        await message.bot.download(message.photo[-1], destination=image_path)
        async with SessionLocal() as session:
            repo = JobRepository(session)
            job = await repo.create_job(message.from_user.id, message.from_user.username, str(image_path))
            await session.commit()
            job_id = job.id
            logger.info("Received image for job=%s user=%s", job.id, message.from_user.id)
            result = await recognize_image(image_path)
            await repo.save_recognition(job, result)
            job.status = JobStatus.AWAITING_CONFIRMATION
            await session.commit()
            await message.answer(format_card(result, result.has_low_confidence), reply_markup=review_keyboard(job.id))
    except Exception:
        logger.exception("Image recognition failed for Telegram user %s", message.from_user.id)
        if job_id is not None:
            async with SessionLocal() as failure_session:
                failed_job = await JobRepository(failure_session).get_job(job_id)
                if failed_job is not None:
                    failed_job.status = JobStatus.FAILED
                    await failure_session.commit()
        await message.answer("Не удалось распознать изображение. Попробуйте отправить более чёткий скриншот.")


@router.callback_query(F.data.startswith("edit:"))
async def choose_field(callback: CallbackQuery) -> None:
    job_id = parse_job_id(callback.data or "")
    await callback.message.edit_text("Что исправить?", reply_markup=fields_keyboard(job_id))
    await callback.answer()


@router.callback_query(F.data.startswith("back:"))
async def back_to_card(callback: CallbackQuery) -> None:
    job_id = parse_job_id(callback.data or "")
    if callback.from_user is None:
        return
    session, repo, job = await load_owned_job(job_id, callback.from_user.id)
    try:
        if job is None:
            await callback.answer("Заявка не найдена", show_alert=True)
            return
        result = repo.result_for(job)
        await callback.message.edit_text(format_card(result, result.has_low_confidence), reply_markup=review_keyboard(job.id))
        await callback.answer()
    finally:
        await session.close()


@router.callback_query(F.data.startswith("field:"))
async def request_field_value(callback: CallbackQuery, state: FSMContext) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) != 3 or parts[2] not in FIELDS:
        await callback.answer("Некорректное поле", show_alert=True)
        return
    await state.set_state(EditField.waiting_for_value)
    await state.update_data(job_id=parts[1], field_name=parts[2])
    await callback.message.edit_text(f"Введите правильное значение для поля «{FIELDS[parts[2]]}».")
    await callback.answer()

@router.callback_query(
    F.data.startswith("select:")
)
async def select_dropdown(
    callback: CallbackQuery,
) -> None:
    """
    Открывает список значений из Google Sheets.
    """

    if callback.from_user is None:
        return

    parts = (
        callback.data or ""
    ).split(":")

    if len(parts) != 3:
        await callback.answer(
            "Некорректный запрос",
            show_alert=True,
        )
        return

    _, field_name, job_id_raw = parts

    if field_name not in {
        "manager",
        "task_status",
    }:
        await callback.answer(
            "Для этого поля нет списка",
            show_alert=True,
        )
        return

    try:
        job_id = uuid.UUID(job_id_raw)
    except ValueError:
        await callback.answer(
            "Некорректная заявка",
            show_alert=True,
        )
        return

    session, repo, job = await load_owned_job(
        job_id,
        callback.from_user.id,
    )

    try:
        if job is None:
            await callback.answer(
                "Заявка не найдена",
                show_alert=True,
            )
            return

        if job.status not in {
            JobStatus.AWAITING_CONFIRMATION,
            JobStatus.AWAITING_DUPLICATE_DECISION,
        }:
            await callback.answer(
                "Эта заявка уже обработана.",
                show_alert=True,
            )
            return

        sheets = GoogleSheetsService()

        options = await sheets.get_dropdown_options(
            field_name
        )

        if not options:
            await callback.answer(
                "В таблице не найден список значений.",
                show_alert=True,
            )
            return

        field_title = {
            "manager": "менеджера",
            "task_status": "статус",
        }[field_name]

        await callback.message.edit_text(
            f"Выберите {field_title}:",
            reply_markup=dropdown_keyboard(
                job.id,
                field_name,
                options,
            ),
        )

        await callback.answer()

    except Exception:
        logger.exception(
            "Could not load dropdown for field=%s",
            field_name,
        )

        await callback.answer(
            "Не удалось получить список из Google Sheets.",
            show_alert=True,
        )

    finally:
        await session.close()

@router.callback_query(
    F.data.startswith("option:")
)
async def apply_dropdown_option(
    callback: CallbackQuery,
) -> None:
    """
    Пользователь нажал на конкретное значение dropdown.
    """

    if callback.from_user is None:
        return

    parts = (
        callback.data or ""
    ).split(":")

    if len(parts) != 4:
        await callback.answer(
            "Некорректный выбор",
            show_alert=True,
        )
        return

    _, field_name, job_id_raw, index_raw = parts

    if field_name not in {
        "manager",
        "task_status",
    }:
        await callback.answer(
            "Некорректное поле",
            show_alert=True,
        )
        return

    try:
        job_id = uuid.UUID(job_id_raw)
        option_index = int(index_raw)
    except ValueError:
        await callback.answer(
            "Некорректный выбор",
            show_alert=True,
        )
        return

    session, repo, job = await load_owned_job(
        job_id,
        callback.from_user.id,
    )

    try:
        if job is None:
            await callback.answer(
                "Заявка не найдена",
                show_alert=True,
            )
            return

        if job.status not in {
            JobStatus.AWAITING_CONFIRMATION,
            JobStatus.AWAITING_DUPLICATE_DECISION,
        }:
            await callback.answer(
                "Эта заявка уже обработана.",
                show_alert=True,
            )
            return

        sheets = GoogleSheetsService()

        options = await sheets.get_dropdown_options(
            field_name
        )

        if option_index < 0 or option_index >= len(options):
            await callback.answer(
                "Этот вариант больше недоступен. "
                "Откройте список заново.",
                show_alert=True,
            )
            return

        selected_value = options[option_index]

        await repo.correct_field(
            job,
            field_name,
            selected_value,
        )

        job.status = JobStatus.AWAITING_CONFIRMATION

        await session.commit()

        result = repo.result_for(job)

        await callback.message.edit_text(
            format_card(
                result,
                result.has_low_confidence,
            ),
            reply_markup=review_keyboard(job.id),
        )

        await callback.answer(
            f"Выбрано: {selected_value}"
        )

    except Exception:
        logger.exception(
            "Could not apply dropdown option "
            "field=%s job=%s",
            field_name,
            job_id,
        )

        await callback.answer(
            "Не удалось сохранить выбранное значение.",
            show_alert=True,
        )

    finally:
        await session.close()

@router.callback_query(
    F.data.startswith("dropdown_back:")
)
async def dropdown_back(
    callback: CallbackQuery,
) -> None:
    if callback.from_user is None:
        return

    job_id = parse_job_id(
        callback.data or ""
    )

    session, repo, job = await load_owned_job(
        job_id,
        callback.from_user.id,
    )

    try:
        if job is None:
            await callback.answer(
                "Заявка не найдена",
                show_alert=True,
            )
            return

        result = repo.result_for(job)

        await callback.message.edit_text(
            format_card(
                result,
                result.has_low_confidence,
            ),
            reply_markup=review_keyboard(job.id),
        )

        await callback.answer()

    finally:
        await session.close()

@router.message(EditField.waiting_for_value, F.text)
async def save_field_value(message: Message, state: FSMContext) -> None:
    if message.from_user is None:
        return
    data = await state.get_data()
    try:
        job_id = uuid.UUID(data["job_id"])
        field_name = data["field_name"]
    except (KeyError, ValueError):
        await state.clear()
        await message.answer("Срок редактирования истёк. Отправьте изображение заново.")
        return
    session, repo, job = await load_owned_job(job_id, message.from_user.id)
    try:
        if job is None or job.status not in {JobStatus.AWAITING_CONFIRMATION, JobStatus.AWAITING_DUPLICATE_DECISION}:
            await message.answer("Эта заявка уже недоступна для редактирования.")
            return
        await repo.correct_field(job, field_name, message.text.strip())
        job.status = JobStatus.AWAITING_CONFIRMATION
        await session.commit()
        result = repo.result_for(job)
        await message.answer(format_card(result, result.has_low_confidence), reply_markup=review_keyboard(job.id))
    finally:
        await session.close()
        await state.clear()


async def write_job(callback: CallbackQuery, job_id: uuid.UUID, skip_duplicate_check: bool) -> None:
    if callback.from_user is None:
        return
    session, repo, job = await load_owned_job(job_id, callback.from_user.id)
    try:
        if job is None:
            await callback.answer("Заявка не найдена", show_alert=True)
            return
        expected_status = JobStatus.AWAITING_DUPLICATE_DECISION if skip_duplicate_check else JobStatus.AWAITING_CONFIRMATION
        if job.status != expected_status:
            await callback.answer("Эта заявка уже обработана или недоступна.", show_alert=True)
            return
        result = repo.result_for(job)
        try:
            result = validate_result(result)
        except ValidationError as exc:
            await callback.message.answer(f"Нельзя сохранить: {exc} Исправьте поле и повторите.", reply_markup=fields_keyboard(job.id))
            await callback.answer()
            return
        await repo.persist_effective_values(job, result)
        sheets = GoogleSheetsService()
        if not skip_duplicate_check and await sheets.has_duplicate(result.number.value):
            job.status = JobStatus.AWAITING_DUPLICATE_DECISION
            await session.commit()
            await callback.message.edit_text(
                f"Заявка {result.number.value} уже есть в таблице.", reply_markup=duplicate_keyboard(job.id)
            )
            await callback.answer()
            return
        await sheets.append(result)
        await repo.mark_confirmed(job)
        job.status = JobStatus.CONFIRMED
        await session.commit()
        await callback.message.edit_text(f"Готово. Заявка {result.number.value} добавлена в таблицу.")
        await callback.answer()
    except Exception:
        logger.exception("Could not write job=%s to Google Sheets", job_id)
        await callback.message.answer("Не удалось записать заявку в Google Sheets. Проверьте настройки и повторите.")
        await callback.answer()
    finally:
        await session.close()


@router.callback_query(F.data.startswith("confirm:"))
async def confirm_job(callback: CallbackQuery) -> None:
    await write_job(callback, parse_job_id(callback.data or ""), skip_duplicate_check=False)


@router.callback_query(F.data.startswith("force:"))
async def force_job(callback: CallbackQuery) -> None:
    await write_job(callback, parse_job_id(callback.data or ""), skip_duplicate_check=True)


@router.callback_query(F.data.startswith("cancel:"))
async def cancel_job(callback: CallbackQuery) -> None:
    if callback.from_user is None:
        return
    job_id = parse_job_id(callback.data or "")
    session, _, job = await load_owned_job(job_id, callback.from_user.id)
    try:
        if job is None:
            await callback.answer("Заявка не найдена", show_alert=True)
            return
        if job.status not in {JobStatus.AWAITING_CONFIRMATION, JobStatus.AWAITING_DUPLICATE_DECISION}:
            await callback.answer("Эта заявка уже обработана.", show_alert=True)
            return
        job.status = JobStatus.CANCELLED
        await session.commit()
        await callback.message.edit_text("Операция отменена. Изображение и результат сохранены для анализа.")
        await callback.answer()
    finally:
        await session.close()
