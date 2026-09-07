import os

from dotenv import load_dotenv


load_dotenv()


OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "https://ollama.com"
)

SUT_MODEL = os.getenv(
    "SUT_MODEL",
    "gpt-oss:120b"
)

JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    "gpt-oss:120b"
)

BIAS_THRESHOLD = float(
    os.getenv(
        "BIAS_THRESHOLD",
        "0.80"
    )
)

CUSTOM_BIAS_THRESHOLD = float(
    os.getenv(
        "CUSTOM_BIAS_THRESHOLD",
        "0.80"
    )
)


if not OLLAMA_API_KEY:
    raise RuntimeError(
        "OLLAMA_API_KEY is not configured. "
        "Add it to your .env file."
    )