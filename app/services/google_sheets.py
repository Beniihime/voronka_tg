import asyncio
import logging
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.schemas import RecognitionResult

logger = logging.getLogger(__name__)


class GoogleSheetsService:
    """Google Sheets access."""

    HEADER = [
        "Дата",
        "Наименование клиента",
        "что сделано",
        "Статус задачи",
        "План (текущий месяц)",
        "Факт (текущий месяц)",
        "Потенциал",
        "Менеджер",
        "Наряд",
        "Статус наряда/акта",
        "Комментарий",
    ]

    # Поля RecognitionResult -> столбцы Google Sheets.
    DROPDOWN_COLUMNS = {
        "task_status": "D",
        "manager": "H",
    }

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _service(self):
        if not self.settings.google_spreadsheet_id:
            raise RuntimeError(
                "GOOGLE_SPREADSHEET_ID is not configured"
            )

        path = Path(
            self.settings.google_credentials_file
        )

        if not path.is_file():
            raise RuntimeError(
                f"Google service-account credentials file "
                f"was not found at {path}."
            )

        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials.from_service_account_file(
            str(path),
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets"
            ],
        )

        return build(
            "sheets",
            "v4",
            credentials=credentials,
            cache_discovery=False,
        )

    # =====================================================================
    # DROPDOWN
    # =====================================================================

    def _get_dropdown_options(
        self,
        field_name: str,
    ) -> list[str]:
        """
        Получает варианты dropdown непосредственно из Google Sheets.

        manager     -> H
        task_status -> D
        """

        column = self.DROPDOWN_COLUMNS.get(field_name)

        if column is None:
            raise ValueError(
                f"Dropdown is not configured for field "
                f"{field_name!r}"
            )

        service = self._service()

        # Получаем только нужный столбец.
        #
        # dataValidation находится в CellData.
        response = (
            service.spreadsheets()
            .get(
                spreadsheetId=self.settings.google_spreadsheet_id,
                ranges=[
                    f"{self.settings.google_sheet_name}!{column}2:{column}"
                ],
                includeGridData=True,
                fields=(
                    "sheets("
                    "data("
                    "rowData("
                    "values("
                    "dataValidation"
                    ")"
                    ")"
                    ")"
                    ")"
                ),
            )
            .execute()
        )

        options: list[str] = []

        for sheet in response.get("sheets", []):
            for data in sheet.get("data", []):
                for row in data.get("rowData", []):
                    for cell in row.get("values", []):
                        validation = cell.get(
                            "dataValidation"
                        )

                        if not validation:
                            continue

                        condition = validation.get(
                            "condition",
                            {},
                        )

                        condition_type = condition.get(
                            "type"
                        )

                        if condition_type != "ONE_OF_LIST":
                            continue

                        for item in condition.get(
                            "values",
                            [],
                        ):
                            value = item.get(
                                "userEnteredValue"
                            )

                            if value is None:
                                continue

                            value = str(value).strip()

                            if value and value not in options:
                                options.append(value)

        return options

    async def get_dropdown_options(
        self,
        field_name: str,
    ) -> list[str]:
        return await asyncio.to_thread(
            self._get_dropdown_options,
            field_name,
        )

    # =====================================================================
    # EXISTING
    # =====================================================================

    def _find_number(
        self,
        number: str,
    ) -> bool:
        values = (
            self._service()
            .spreadsheets()
            .values()
            .get(
                spreadsheetId=self.settings.google_spreadsheet_id,
                range=(
                    f"'{self.settings.google_sheet_name}'!I:I"
                ),
            )
            .execute()
            .get("values", [])
        )

        return any(
            row and row[0].strip() == number
            for row in values[1:]
        )

    async def has_duplicate(
        self,
        number: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._find_number,
            number,
        )

    def _append(
        self,
        result: RecognitionResult,
    ) -> None:
        values = [[
            result.date.value,
            result.counterparty.value,
            "",
            result.task_status.value,
            "",
            "",
            "",
            result.manager.value,
            result.number.value,
            "",
            result.comment.value,
        ]]

        (
            self._service()
            .spreadsheets()
            .values()
            .append(
                spreadsheetId=self.settings.google_spreadsheet_id,
                range=(
                    f"'{self.settings.google_sheet_name}'!A:K"
                ),
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body={"values": values},
            )
            .execute()
        )

        logger.info(
            "Appended application %s to Google Sheets",
            result.number.value,
        )

    async def append(
        self,
        result: RecognitionResult,
    ) -> None:
        await asyncio.to_thread(
            self._append,
            result,
        )