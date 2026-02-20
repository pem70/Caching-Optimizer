import tiktoken

from app.models.schemas import Message


# Cache encoding objects to avoid repeated initialization
_encoding_cache: dict[str, tiktoken.Encoding] = {}

# OpenAI message format overhead:
# every message costs 4 tokens: <|im_start|>{role}\n ... <|im_end|>\n
TOKENS_PER_MESSAGE = 4
# every reply is primed with <|im_start|>assistant<|im_sep|>
TOKENS_PER_REPLY = 3


def get_encoding(model: str) -> tiktoken.Encoding:
    if model not in _encoding_cache:
        try:
            _encoding_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _encoding_cache[model] = tiktoken.get_encoding("o200k_base")
    return _encoding_cache[model]


def count_text_tokens(text: str, model: str) -> int:
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_message_tokens(message: Message, model: str) -> int:
    encoding = get_encoding(model)
    tokens = TOKENS_PER_MESSAGE
    tokens += len(encoding.encode(message.content))
    tokens += len(encoding.encode(message.role.value))
    if message.name:
        tokens += len(encoding.encode(message.name))
    return tokens


def count_messages_tokens(messages: list[Message], model: str) -> int:
    total = sum(count_message_tokens(m, model) for m in messages)
    total += TOKENS_PER_REPLY
    return total