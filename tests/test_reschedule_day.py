from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from common.timeutils import resolve_reschedule_day

TZ = ZoneInfo("Europe/Moscow")


@pytest.mark.parametrize(
    ("source", "day"),
    [
        ("Перенеси её на завтра в 12:20", 11),
        ("Перенеси её на сегодня в 12:20", 10),
        ("Перенеси её на послезавтра в 12:20", 12),
        ("На завтра в 12.20", 11),
        ("Перенеси «Завтра» на послезавтра в 12:20", 12),
        ("Перенеси на 15 ноября в 12:20", 15),
        ("Перенеси на 15.11 в 12:20", 15),
        ("Перенеси на день позже", 15),
        ("Переименуй её в «Завтра»", 15),
    ],
)
def test_relative_day_uses_request_local_date_only(source, day):
    proposed = datetime(2030, 11, 15, 12, 20, tzinfo=TZ)
    reference = datetime(2030, 11, 10, 1, 10, tzinfo=TZ)
    result = resolve_reschedule_day(source, proposed, reference)
    assert result.day == day and (result.hour, result.minute) == (12, 20)


def test_relative_day_uses_local_day_and_handles_year_boundary():
    reference = datetime.fromisoformat("2030-12-31T22:10:00+00:00").astimezone(TZ)
    proposed = datetime(2031, 1, 3, 12, 20, tzinfo=TZ)
    assert resolve_reschedule_day("На завтра в 12:20", proposed, reference) == proposed.replace(
        day=2
    )


def test_conflicting_relative_days_require_clarification():
    now = datetime(2030, 11, 10, tzinfo=TZ)
    with pytest.raises(ValueError, match="Уточните"):
        resolve_reschedule_day("Перенеси на завтра или послезавтра", now, now)
