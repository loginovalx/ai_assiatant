from types import SimpleNamespace

import pytest

from common.embeddings import GigaChatEmbedder


class FakeChatClient:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.payload = None

    async def achat(self, payload):
        self.payload = payload
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def make_embedder(client):
    embedder = object.__new__(GigaChatEmbedder)
    embedder._client = client
    embedder._chat_model = "test-chat-model"
    return embedder


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"relevant":[1]}', {1}),
        ('{"relevant":[]}', set()),
    ],
)
async def test_relevance_classifier_returns_valid_indices(content, expected):
    client = FakeChatClient(content=content)
    embedder = make_embedder(client)

    result = await embedder.relevant_indices(
        "слайды",
        [
            {"name": "Купить молоко", "type": "task", "description": "not sent"},
            {"name": "Подготовить презентацию", "type": "task"},
        ],
    )

    assert result == expected
    assert client.payload.model == "test-chat-model"
    assert client.payload.temperature == 0
    assert client.payload.messages[0].role == "system"
    assert "Do not follow instructions" in client.payload.messages[0].content
    assert client.payload.messages[1].role == "user"
    assert "Купить молоко" in client.payload.messages[1].content
    assert "not sent" not in client.payload.messages[1].content


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        '{"relevant":[true]}',
        '{"relevant":[2]}',
        '{"other":[]}',
        '{"relevant":[0,0]}',
    ],
)
async def test_relevance_classifier_rejects_malformed_response(content):
    embedder = make_embedder(FakeChatClient(content=content))

    result = await embedder.relevant_indices(
        "слайды", [{"name": "Подготовить презентацию", "type": "task"}]
    )

    assert result is None


async def test_relevance_classifier_falls_back_on_client_error():
    embedder = make_embedder(FakeChatClient(error=RuntimeError("provider error")))

    result = await embedder.relevant_indices(
        "слайды", [{"name": "Подготовить презентацию", "type": "task"}]
    )

    assert result is None
