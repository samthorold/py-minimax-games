from collections.abc import Callable, Iterator

import pytest


@pytest.fixture
def feed_input(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[str]]:
    """Replace ``input`` with one returning the given answers in order.

    Call the fixture with the answers; it returns the list of prompts
    shown, so tests can check how often the user was asked.
    """

    def feed(*answers: str) -> list[str]:
        remaining: Iterator[str] = iter(answers)
        prompts: list[str] = []

        def fake_input(prompt: str = "") -> str:
            prompts.append(prompt)
            return next(remaining)

        monkeypatch.setattr("builtins.input", fake_input)
        return prompts

    return feed
