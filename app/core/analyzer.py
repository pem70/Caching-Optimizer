from app.models.schemas import Message, Role, Volatility


def classify_messages(
    messages: list[Message],
    static_prefix_count: int | None = None,
) -> list[Message]:
    """Give each message a volatility label.

    If static_prefix_count is provided, the first N messages are forced STATIC.
    Otherwise, use heuristics:
      - system messages        → STATIC
      - middle user/assistant  → SEMI_STATIC  (few-shot / history)
      - last user message      → DYNAMIC      (current turn)
      - tool messages          → DYNAMIC
    """
    result = [m.model_copy() for m in messages]
    n = len(result)

    # User override: force first N as static
    if static_prefix_count is not None:
        for i, msg in enumerate(result):
            if msg.volatility is not None:
                continue
            if i < static_prefix_count:
                msg.volatility = Volatility.STATIC
            elif i == n - 1 and msg.role == Role.USER:
                msg.volatility = Volatility.DYNAMIC
            else:
                msg.volatility = Volatility.SEMI_STATIC
        return result

    # Auto-detection
    for i, msg in enumerate(result):
        if msg.volatility is not None:
            continue

        if msg.role == Role.SYSTEM:
            msg.volatility = Volatility.STATIC

        elif msg.role == Role.TOOL:
            msg.volatility = Volatility.DYNAMIC

        elif i == n - 1 and msg.role == Role.USER:
            msg.volatility = Volatility.DYNAMIC

        else:
            msg.volatility = Volatility.SEMI_STATIC

    return result