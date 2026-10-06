import asyncio
import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.schemas import RecognitionResult

logger = logging.getLogger(__name__)


class GoogleSheetsService:
    """Only Google Sheets access lives here; calls are offloaded from the event loop."""

    HEADER = [
        "Дата", "Наименование клиента", "что сделано", "Статус задачи",
        "План (текущий месяц)", "Факт (текущий месяц)", "Потенциал",
        "Менеджер", "Наряд", "Статус наряда/акта", "Комментарий",
    ]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _service(self):
        if not self.settings.google_spreadsheet_id:
            raise RuntimeError("GOOGLE_SPREADSHEET_ID is not configured")
        path = Path(self.settings.google_credentials_file)
        if not path.is_file():
            raise RuntimeError(
                f"Google service-account credentials file was not found at {path}. "
                "Place the JSON key in ./secrets and set GOOGLE_CREDENTIALS_FILE to its /app/secrets path."
            )
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        credentials = Credentials.from_service_account_file(
            str(path), scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        return build("sheets", "v4", credentials=credentials, cache_discovery=False)

    def _find_number(self, number: str) -> bool:
        values = self._service().spreadsheets().values().get(
            spreadsheetId=self.settings.google_spreadsheet_id,
            range=f"'{self.settings.google_sheet_name}'!I:I",
        ).execute().get("values", [])
        return any(row and row[0].strip() == number for row in values[1:])

    async def has_duplicate(self, number: str) -> bool:
        return await asyncio.to_thread(self._find_number, number)

    def _append(self, result: RecognitionResult) -> None:
        values = [[
            result.date.value,
            result.counterparty.value,
            "",  # No separate source value exists for this CRM column.
            result.task_status.value,
            "", "", "",
            result.manager.value,
            result.number.value,
            "",
            result.comment.value,
        ]]
        self._service().spreadsheets().values().append(
            spreadsheetId=self.settings.google_spreadsheet_id,
            range=f"'{self.settings.google_sheet_name}'!A:K",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": values},
        ).execute()
        logger.info("Appended application %s to Google Sheets", result.number.value)

    async def append(self, result: RecognitionResult) -> None:
        await asyncio.to_thread(self._append, result)
