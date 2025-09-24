def read_errors(log_text: str) -> list[str]:
    return [ln for ln in log_text.splitlines() if "%error" in ln.lower()]
