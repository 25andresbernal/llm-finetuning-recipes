#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/chat/completions server for the fine-tuned adapter.

Real mode loads the base model plus a LoRA adapter with Unsloth and requires
a GPU and the ``[train]`` extra (Unsloth's inference path shares the same
dependencies as training). Fake mode (``--fake``, on by default when no
adapter path is given) returns canned responses from a small fixture so the
request and response shapes, the stop-phrase guarding, and the client
integration can all be tested on any machine, with no model loaded.

Run it:

    uv run python serve/server.py --fake
    curl -s localhost:8000/v1/chat/completions \\
      -H "content-type: application/json" \\
      -d '{"model": "conversational-agent-qlora", "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "I need to schedule a repair."}
          ]}'

The FastAPI ``app`` object is created eagerly so tests can import it and
drive it with a TestClient without starting uvicorn. Nothing about the
fake path touches torch or transformers.
"""

from __future__ import annotations

import argparse
import itertools
import time
import uuid
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

# Turns that would let the model start speaking as the caller instead of
# the agent. Generation is stopped as soon as one of these appears, so a
# single agent turn can never turn into a simulated back-and-forth.
STOP_PHRASES = [
    "\nCaller:",
    "\ncaller:",
    "<|start_header_id|>user",
    "<|eot_id|>",
]

FAKE_RESPONSES = [
    "What day this week works best for you?",
    "I have you down for Wednesday at 2 PM.",
    "Is this something urgent, or can it wait a few days?",
    "You're all set, the technician will call before arriving.",
    "Can you confirm the name the appointment is under?",
]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = 64
    temperature: float | None = 0.0


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class ServerState:
    """Holds the loaded model (real mode) or nothing at all (fake mode)."""

    def __init__(self, fake: bool = True, adapter_path: str | None = None) -> None:
        self.fake = fake
        self.adapter_path = adapter_path
        self._fake_cycle = itertools.cycle(FAKE_RESPONSES)
        self._model = None
        self._tokenizer = None

    def load_real_model(self) -> None:  # pragma: no cover - requires a GPU
        """Load the base model and adapter. Only called outside --fake mode."""
        import torch
        from transformers import AutoTokenizer
        from unsloth import FastLanguageModel

        if not self.adapter_path:
            raise RuntimeError("--adapter is required outside --fake mode")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.adapter_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.adapter_path)
        self._torch = torch

    def generate(self, messages: list[ChatMessage], max_tokens: int) -> str:
        if self.fake:
            return next(self._fake_cycle)
        return self._generate_real(messages, max_tokens)  # pragma: no cover

    def _generate_real(  # pragma: no cover - requires a GPU
        self, messages: list[ChatMessage], max_tokens: int
    ) -> str:
        prompt = self._build_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        stopping_criteria = build_stopping_criteria(self._tokenizer, STOP_PHRASES)
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            stopping_criteria=stopping_criteria,
        )
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return strip_stop_phrases(text, STOP_PHRASES)

    def _build_prompt(self, messages: list[ChatMessage]) -> str:  # pragma: no cover
        parts = ["<|begin_of_text|>"]
        for m in messages:
            parts.append(f"<|start_header_id|>{m.role}<|end_header_id|>\n\n{m.content}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)


def strip_stop_phrases(text: str, stop_phrases: list[str]) -> str:
    """Cut text at the first stop phrase, so a partial match never leaks through."""
    cut_at = len(text)
    for phrase in stop_phrases:
        idx = text.find(phrase)
        if idx != -1:
            cut_at = min(cut_at, idx)
    return text[:cut_at].strip()


def build_stopping_criteria(tokenizer, stop_phrases: list[str]):  # pragma: no cover
    """A transformers StoppingCriteriaList that halts generation at any stop phrase.

    This is what actually prevents the model from simulating the other
    speaker at generation time, before ``strip_stop_phrases`` even runs: as
    soon as the decoded tail of the generation matches a stop phrase,
    generation halts for that sequence. Only imported in real mode.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnPhrases(StoppingCriteria):
        def __init__(self, tokenizer, phrases: list[str], prompt_len: int) -> None:
            self.tokenizer = tokenizer
            self.phrases = phrases
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            generated = input_ids[0][self.prompt_len :]
            text = self.tokenizer.decode(generated, skip_special_tokens=False)
            return any(phrase in text for phrase in self.phrases)

    return StoppingCriteriaList([StopOnPhrases(tokenizer, stop_phrases, 0)])


def create_app(fake: bool = True, adapter_path: str | None = None) -> FastAPI:
    app = FastAPI(title="conversational-agent-qlora server")
    state = ServerState(fake=fake, adapter_path=adapter_path)
    if not fake:
        state.load_real_model()  # pragma: no cover
    app.state.server_state = state

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "fake": state.fake}

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        text = state.generate(request.messages, request.max_tokens or 64)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--fake",
        action="store_true",
        help="serve canned responses instead of loading a model (default if --adapter is omitted)",
    )
    parser.add_argument(
        "--adapter", default=None, help="path or HF repo id of the fine-tuned adapter"
    )
    args = parser.parse_args()

    fake = args.fake or args.adapter is None
    app = create_app(fake=fake, adapter_path=args.adapter)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
