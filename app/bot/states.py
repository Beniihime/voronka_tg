from aiogram.fsm.state import State, StatesGroup


class EditField(StatesGroup):
    waiting_for_value = State()
