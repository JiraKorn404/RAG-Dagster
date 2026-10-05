"""The chat model: Ollama through langchain-ollama, with the state of every call reported."""

import httpx
from langchain_ollama import ChatOllama
from langgraph.config import get_stream_writer

from rag_lab.agent.events import AnswerToken, ModelState, Thinking
from rag_lab.config import AgentConfig


def chat_model(base_url: str, cfg: AgentConfig, think: bool) -> ChatOllama:
    return ChatOllama(
        model=cfg.model,
        base_url=base_url,
        temperature=cfg.temperature,
        num_ctx=cfg.num_ctx,
        keep_alive=cfg.keep_alive,
        reasoning=think,
    )


def model_loaded(base_url: str, model: str) -> bool | None:
    """Whether Ollama already has the model in memory (a cold model makes the first call slow)."""
    try:
        listed = httpx.get(f"{base_url.rstrip('/')}/api/ps", timeout=5).json()["models"]
    except (httpx.HTTPError, ValueError, KeyError):
        return None
    return any(m.get("name") == model for m in listed)


def call_model(
    llm: ChatOllama, base_url: str, cfg: AgentConfig, node: str, think: bool, messages: list, stream: bool
) -> tuple[str, str]:
    """Run one call and report a ModelState. With `stream` the thinking and the answer are reported as
    they arrive. Returns (answer, thinking)."""
    emit = get_stream_writer()
    loaded = model_loaded(base_url, cfg.model)
    text, thinking, usage, meta = "", "", {}, {}
    if stream:
        for chunk in llm.stream(messages):
            piece = chunk.additional_kwargs.get("reasoning_content") or ""
            if piece:
                thinking += piece
                emit(Thinking(piece))
            if chunk.content:
                text += chunk.content
                emit(AnswerToken(chunk.content))
            if chunk.usage_metadata:  # only the chunk that ends the generation has the counts
                usage, meta = chunk.usage_metadata, chunk.response_metadata
    else:
        reply = llm.invoke(messages)
        text, thinking = reply.content, reply.additional_kwargs.get("reasoning_content") or ""
        usage, meta = reply.usage_metadata or {}, reply.response_metadata
    seconds = (meta.get("eval_duration") or 0) / 1e9
    prompt_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    emit(
        ModelState(
            node=node,
            model=cfg.model,
            think=think,
            loaded=loaded,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            num_ctx=cfg.num_ctx,
            tokens_per_s=output_tokens / seconds if seconds else None,
            context_full=prompt_tokens >= cfg.num_ctx,
        )
    )
    return text, thinking
