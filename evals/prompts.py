"""Prompt strategies, attached per model spec.

A prompt strategy turns an extracted cloze *prefix* (the text up to the masked word)
into the actual model input. Plain models receive the prefix verbatim; the
instruction-tuned Talkie model receives a fixed few-shot chat prompt so it answers
with a single bare word instead of assistant-style filler. Because the strategy lives
on the spec, it applies uniformly to both the cloze and composition harnesses.
"""

from typing import Protocol


class PromptStrategy(Protocol):
    def cloze_prompt(self, prefix: str) -> str:
        """Wrap an extracted prefix into model-ready input text."""
        ...


class PlainPrefixPrompt:
    """Return the prefix unchanged (Talkie base/web, Typewriter)."""

    def cloze_prompt(self, prefix: str) -> str:
        return prefix


class ITFewShotPrompt:
    """Few-shot multi-turn chat prompt for the instruction-tuned Talkie model.

    Uses ``talkie.Message`` / ``talkie.format_chat`` (imported lazily so this module
    loads on machines without the ``talkie`` package). The three exemplars coerce the
    IT model to reply with a single completion word.
    """

    _EXEMPLARS = [
        ('What one word completes this text? "The cat sat on the"', "mat"),
        ('What one word completes this text? "She opened the door and stepped"', "outside"),
        ('What one word completes this text? "He picked up his pen and began to"', "write"),
    ]

    def cloze_prompt(self, prefix: str) -> str:
        from talkie import Message, format_chat

        messages = []
        for user, assistant in self._EXEMPLARS:
            messages.append(Message(role="user", content=user))
            messages.append(Message(role="assistant", content=assistant))
        messages.append(
            Message(role="user", content=f'What one word completes this text? "{prefix}"'))
        return format_chat(messages)
