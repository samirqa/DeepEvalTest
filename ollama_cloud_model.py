from typing import Optional

from ollama import Client

from deepeval.models import DeepEvalBaseLLM

from config.settings import (
    OLLAMA_API_KEY,
    OLLAMA_HOST,
    JUDGE_MODEL
)


class OllamaCloudModel(DeepEvalBaseLLM):

    def __init__(
        self,
        model: str = JUDGE_MODEL,
        temperature: float = 0
    ):

        
        self.model_name = model
        self.temperature = temperature

        self.client = Client(
            host=OLLAMA_HOST,
            headers={
                "Authorization":
                    "Bearer " + OLLAMA_API_KEY
            }
        )

    def load_model(self):

        
        return self.client

    def generate(
        self,
        prompt: str,
        schema: Optional[object] = None
    ):

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": self.temperature
            }
        )

        return response["message"]["content"]

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[object] = None
    ):

        return self.generate(
            prompt,
            schema
        )

    def get_model_name(self):

        return self.model_name