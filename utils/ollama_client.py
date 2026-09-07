from ollama import Client

from config.settings import (
    OLLAMA_API_KEY,
    OLLAMA_HOST,
    SUT_MODEL
)


client = Client(
    host=OLLAMA_HOST,
    headers={
        "Authorization": (
            "Bearer " + OLLAMA_API_KEY
        )
    }
)


def generate_response(prompt: str) -> str:

    response = client.chat(
        model=SUT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return response["message"]["content"]