
import httpx
import json
import tiktoken


from llama_index.core.llms import (
    LLM, CompletionResponse, LLMMetadata, ChatMessage, ChatResponse, MessageRole,
    ChatResponseGen, ChatResponseAsyncGen, CompletionResponseGen, CompletionResponseAsyncGen,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.embeddings import BaseEmbedding

from pydantic import PrivateAttr, Field
from typing import Any, Iterator, Sequence, Generator, AsyncGenerator


class CustomLLM(LLM):
    model: str = Field(description="Name of the LLM")
    default_model_params: dict | None = Field(description="Set default values for LLM parameters", default=None)
    api_chat_endpoint: str = Field(description="URL of the chat completion endpoint")
    api_completion_endpoint: str = Field(description="URL of the plain completion endpoint")

    _sync_client: httpx.Client = PrivateAttr()
    _async_client: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self, model: str, api_chat_endpoint: str, api_completion_endpoint: str,
        default_model_params: dict | None = None, sync_client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None, **kwargs
    ):
        super().__init__(
            model=model, default_model_params=default_model_params, api_chat_endpoint=api_chat_endpoint,
            api_completion_endpoint=api_completion_endpoint, **kwargs
        )
        self._sync_client = sync_client or httpx.Client()
        self._async_client = async_client or httpx.AsyncClient()

    @property
    def metadata(self) -> LLMMetadata:
        # Deactivate chat endpoint for CodeLlama models due to incorrect chat template in llama-cpp-server
        is_chat_model = not self.model.lower().startswith("codellama")
        return LLMMetadata(
            model_name=self.model,
            context_window=16385,
            num_output=4096,
            is_chat_model=is_chat_model
        )

    def count_tokens(self, text: str) -> int:
        """Tokenize an input string and count the number of tokens.

        Parameters:
            text: Input string that should tokenized
        
        Returns: Number of tokens
        """
        if self.model.startswith("gpt-"):
            # OpenAI models (e.g., "gpt-3.5-turbo", "gpt-4o")
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        else:
            raise NotImplementedError(f"Tokenizer unknown for model {self.model}")

    def _build_chat_request(self, messages: Sequence[ChatMessage], **kwargs: dict) -> httpx.Request:
        """Helper function that prepares a POST request for the chat completion endpoint

        Parameters:
            messages: List of messages that will be inserted to the payload of the request
            kwargs: Optional parameters that should be passed to the API
        
        Returns: Request object
        """
        _messages = []
        # Insert system prompt as the first message if one is defined
        if self.system_prompt is not None:
            _messages.append({"role": MessageRole.SYSTEM.value, "content": self.system_prompt})
        # Add messages passed to this function to the list
        _messages.extend({"role": m.role.value, "content": m.content} for m in messages)
        # Combine default parameters with kwargs
        params = self.default_model_params or {}
        params.update(kwargs)
        # Build request object
        return httpx.Request("POST", self.api_chat_endpoint, json={"messages": _messages, **params})
    
    def _build_completion_request(self, prompt: str, **kwargs: dict) -> httpx.Request:
        """Helper function that prepares a POST request for the completion endpoint

        Parameters:
            prompt: Prompt for the LLM
            kwargs: Optional parameters that should be passed to the API
        
        Returns: Request object
        """
        # Combine default parameters with kwargs
        params = self.default_model_params or {}
        params.update(kwargs)
        # Build request object
        return httpx.Request("POST", self.api_completion_endpoint, json={"prompt": prompt, **params})
    
    @staticmethod
    def _extract_response_metadata(raw: dict[str, Any]) -> dict[str, Any]:
        return {key: raw.get(key) for key in ("model", "system_fingerprint", "usage")}

    @staticmethod
    def _response_to_chat_response(response: httpx.Response) -> ChatResponse:
        """Helper function that transforms a Response object from the API to a ChatResponse object
        
        Parameters:
            response: Response object containing the JSON payload

        Returns: ChatResponse object
        """
        # TODO: Improve parsing and error handling
        if response.is_error:
            response.raise_for_status()

        raw = response.json()
        message = raw["choices"][0]["message"]
        return ChatResponse(
            # NOTE: Content attribute can be missing due to filter rules
            message=ChatMessage(role=MessageRole(message["role"]), content=message.get("content")),
            raw=raw,
            additional_kwargs={
                "duration": response.elapsed.total_seconds(),
                **CustomLLM._extract_response_metadata(raw)
            }
        )

    @staticmethod
    def _response_to_completion_response(response: httpx.Response) -> CompletionResponse:
        """Helper function that transforms a Response object from the API to a CompletionResponse object
        
        Parameters:
            response: Response object containing the JSON payload

        Returns: CompletionResponse object
        """
        if response.is_error:
            response.raise_for_status()

        raw = response.json()
        return CompletionResponse(
            text=raw["choices"][0]["text"],
            raw=raw,
            additional_kwargs={
                "duration": response.elapsed.total_seconds(),
                **CustomLLM._extract_response_metadata(raw)
            }
        )

    @staticmethod
    def _completion_response_to_chat_response(response: CompletionResponse) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response.text),
            raw=response.raw,
            delta=response.delta,
            additional_kwargs=response.additional_kwargs
        )

    @staticmethod
    def _stream_response_generator() -> Generator[dict | None, str, None]:
        """Create a generator processes lines from a SSE (server-sent events) stream and outputs the JSON objects from
        the stream. The generator will yield `None` if the chunk has not been completed yet.
        
        Reference: https://html.spec.whatwg.org/multipage/server-sent-events.html
        """
        def gen() -> Generator[dict | None, str, None]:
            chunk = ""
            line = yield
            while True:
                # Empty line indicates a separator (i.e., the current chunk is complete and the next will start)
                if line == "":
                    if chunk.strip() == "[DONE]":
                        # Reached end of stream, quit generator safely
                        yield
                        break
                    
                    # Yield completed chunk as dictionary and read next line
                    line = yield json.loads(chunk)
                    # Reset chunk
                    chunk = ""
                elif line.startswith("data:"):
                    # Append line to current chunk
                    chunk += line[5:].lstrip() + "\n"
                    line = yield

        # Auto-initialize the generator instance
        instance = gen()
        instance.send(None)
        return instance

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: dict) -> ChatResponse:
        if not self.metadata.is_chat_model:
            # Convert messages to prompt and use completion endpoint instead of chat endpoint
            prompt = self.messages_to_prompt(messages)
            response = self.complete(prompt, **kwargs)
            return self._completion_response_to_chat_response(response)
        else:
            request = self._build_chat_request(messages, **kwargs)
            response = self._sync_client.send(request)
            return self._response_to_chat_response(response)
    
    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if not self.metadata.is_chat_model:
            # Convert messages to prompt and use completion endpoint instead of chat endpoint
            prompt = self.messages_to_prompt(messages)
            response = await self.acomplete(prompt, **kwargs)
            return self._completion_response_to_chat_response(response)
        else:
            request = self._build_chat_request(messages, **kwargs)
            response = await self._async_client.send(request)
            return self._response_to_chat_response(response)
        
    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            for response in self.stream_complete(prompt, **kwargs):
                yield self._completion_response_to_chat_response(response)
        else:
            request = self._build_chat_request(messages, stream=True, **kwargs)
            response = self._sync_client.send(request, stream=True)
            generator = self._stream_response_generator()
            content = ""
            for line in response.iter_lines():
                chunk = generator.send(line)
                if chunk is None:
                    continue

                delta = chunk["choices"][0]["delta"].get("content", "")
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                    raw=chunk,
                    delta=delta,
                )
    
    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        """Asynchronous streaming of a chat completion

        Example:
        ```
        async for response in await llm.astream_chat(...):
            print(response.delta, end="", flush=True)
        ```
        """
        async def gen() -> ChatResponseAsyncGen:
            if not self.metadata.is_chat_model:
                prompt = self.messages_to_prompt(messages)
                async for response in await self.astream_complete(prompt, **kwargs):
                    yield self._completion_response_to_chat_response(response)
            else:
                request = self._build_chat_request(messages, stream=True, **kwargs)
                response = await self._async_client.send(request, stream=True)
                generator = self._stream_response_generator()
                content = ""
                async for line in response.aiter_lines():
                    chunk = generator.send(line)
                    if chunk is None:
                        continue
                    
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    content += delta
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                        raw=chunk,
                        delta=delta,
                    )
        # NOTE: The generator must be instantiated this way due to the internal handling in llama_index
        return gen()
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        request = self._build_completion_request(prompt, **kwargs)
        response = self._sync_client.send(request)
        return self._response_to_completion_response(response)
    
    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        request = self._build_completion_request(prompt, **kwargs)
        response = await self._async_client.send(request)
        return self._response_to_completion_response(response)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        request = self._build_completion_request(prompt, stream=True, **kwargs)
        response = self._sync_client.send(request, stream=True)
        generator = self._stream_response_generator()
        content = ""
        for line in response.iter_lines():
            chunk = generator.send(line)
            if chunk is None:
                continue

            delta = chunk["choices"][0]["text"]
            content += delta
            yield CompletionResponse(text=content, delta=delta, raw=chunk)

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, **kwargs) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            request = self._build_completion_request(prompt, stream=True, **kwargs)
            response = await self._async_client.send(request, stream=True)
            generator = self._stream_response_generator()
            content = ""
            async for line in response.aiter_lines():
                chunk = generator.send(line)
                if chunk is None:
                    continue
                delta = chunk["choices"][0]["text"]
                content += delta
                yield CompletionResponse(text=content, delta=delta, raw=chunk)
        # NOTE: The generator must be instantiated this way due to the internal handling in llama_index
        return gen()


class CustomEmbedding(BaseEmbedding):
    api_url: str = Field(description="URL of the embedding endpoint")

    _sync_client: httpx.Client = PrivateAttr()
    _async_client: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self,
        api_url: str,
        sync_client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        **kwargs
    ):
        super().__init__(api_url=api_url, **kwargs)
        self._sync_client = sync_client or httpx.Client()
        self._async_client = async_client or httpx.AsyncClient()

    @classmethod
    def class_name(cls) -> str:
        return "custom_embedding"
    
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self._sync_client.post(self.api_url, json={"input": texts})
        data = response.json()
        return list(map(lambda item: item["embedding"], data["data"]))
    
    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = await self._async_client.post(self.api_url, json={"input": texts})
        data = response.json()
        return list(map(lambda item: item["embedding"], data["data"]))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._aget_text_embeddings([query])[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await self._aget_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]
