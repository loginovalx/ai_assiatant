from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from api.ics import build_ics
from common.schemas import Event, EventCreate, EventStatus, EventType


@pytest.mark.parametrize("name", [" ", "   ", "\t\n"])
def test_event_name_must_remain_nonempty_after_trimming(name):
    with pytest.raises(ValidationError, match="name"):
        EventCreate(user_id="user-1", type=EventType.TASK, name=name)


def test_event_name_is_trimmed_and_empty_description_is_allowed():
    event = EventCreate(
        user_id="user-1", type=EventType.TASK, name="  Prepare report  ", description="  "
    )

    assert event.name == "Prepare report"
    assert event.description == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Planning\rSTATUS:CANCELLED"),
        ("description", "Notes\rEND:VEVENT\rBEGIN:VEVENT\rSUMMARY:Injected"),
    ],
)
def test_ics_text_cannot_inject_properties_or_components(field, value):
    event = Event(
        id=uuid4(),
        user_id="user-1",
        type=EventType.TASK,
        name=value if field == "name" else "Planning",
        description=value if field == "description" else "",
        start_at=datetime(2026, 10, 31, 10),
        status=EventStatus.ACTIVE,
        created_at=datetime(2026, 10, 31, 10),
        updated_at=datetime(2026, 10, 31, 10),
    )

    ics = build_ics(event)

    assert "\r\nSTATUS:CANCELLED\r\n" not in ics
    assert "\r\nEND:VEVENT\r\nBEGIN:VEVENT\r\n" not in ics
    assert "\r\nSUMMARY:Injected\r\n" not in ics
