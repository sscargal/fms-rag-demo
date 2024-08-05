#one file that you can specify memory settings in using valves?

from pydantic import BaseModel
from pydantic import Field
from typing import Optional
from typing import List
from typing import Union
from typing import Generator
from typing import Iterator
import os

async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        print(f"action:{__name__}")

        response = await __event_call__(
            {
                "type": "input",
                "data": {
                    "title": "write a message",
                    "message": "here write a message to append",
                    "placeholder": "enter your message",
                },
            }
        )
        print(response)
        
class Pipe:
    class Valves(BaseModel):
        RANDOM_CONFIG_OPTION: str = Field(default="")

    def __init__(self):
        self.type = "pipe"
        self.id = "blah"
        self.name = "Testing"
        self.valves = self.Valves(
            **{"RANDOM_CONFIG_OPTION": os.getenv("RANDOM_CONFIG_OPTION", "")}
        )
        pass

    def get_provider_models(self):
        return [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
            {"id": "model_id_3", "name": "model_3"},
        ]

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
      # Logic goes here
      return "Hi"