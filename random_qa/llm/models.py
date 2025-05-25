import httpx
import os
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

from typing import Sequence

from random_qa.llm.base import CustomLLM
from random_qa.oauth import get_oauth_clients

LLM_LOCAL_CHAT_ENDPOINT = os.getenv("LLM_LOCAL_CHAT_ENDPOINT")
LLM_LOCAL_COMPLETION_ENDPOINT = os.getenv("LLM_LOCAL_COMPLETION_ENDPOINT")


def _codellama_messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    # Source: https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf#chat-prompt
    prompt = "<s>"
    for m in messages:
        prompt += f"Source: {m.role.value}\n\n {m.content.strip()}"
        prompt += " <step> "
    prompt += "Source: assistant\nDestination: user\n\n "
    return prompt


def get_model_by_name(model_name: str, model_params: dict) -> CustomLLM:
    """Obtain a LLM instance for a given model name"""
    local_sync_client, local_async_client = httpx.Client(timeout=120), httpx.AsyncClient(timeout=120)
    oauth_sync_client, oauth_async_client = get_oauth_clients(timeout=60)
    if model_name == "codellama-70b":
        return CustomLLM(
            model="codellama-70b",
            api_completion_endpoint=LLM_LOCAL_COMPLETION_ENDPOINT,
            api_chat_endpoint=LLM_LOCAL_CHAT_ENDPOINT,
            sync_client=local_sync_client,
            async_client=local_async_client,
            # customize stop-sequences and prompt formatting because llama-cpp-server does not support the chat template
            # of the CodeLlama model
            default_model_params={"stop": ["Source: assistant", "<step>"], **model_params},
            messages_to_prompt=_codellama_messages_to_prompt
        )
    elif model_name == "qwen2.5-coder-32b" or model_name == "llama3.3-70b":
        return CustomLLM(
            model=model_name,
            api_completion_endpoint=LLM_LOCAL_COMPLETION_ENDPOINT,
            api_chat_endpoint=LLM_LOCAL_CHAT_ENDPOINT,
            sync_client=local_sync_client,
            async_client=local_async_client,
        )
    elif model_name == "gpt-4o":
        return CustomLLM(
            model="gpt-4o",
            api_chat_endpoint=os.getenv("LLM_GPT4o_CHAT_ENDPOINT"),
            api_completion_endpoint=os.getenv("LLM_GPT4o_COMPLETION_ENDPOINT"),
            sync_client=oauth_sync_client,
            async_client=oauth_async_client,
            default_model_params=model_params
        )
    elif model_name == "gpt-3.5-turbo":
        return CustomLLM(
            model="gpt-3.5-turbo",
            api_chat_endpoint=os.getenv("LLM_GPT35_CHAT_ENDPOINT"),
            api_completion_endpoint=os.getenv("LLM_GPT35_COMPLETION_ENDPOINT"),
            sync_client=oauth_sync_client,
            async_client=oauth_async_client,
            default_model_params=model_params
        )
    elif model_name.startswith("openai:"):
        return OpenAI(model=model_name[8:], api_key=os.getenv("OPENAI_API_KEY"), **model_params)
    else:
        raise ValueError(f"Unknown model name {model_name!r}")
