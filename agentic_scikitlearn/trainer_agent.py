import pandas as pd
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent
from agentic_scikitlearn.workflows.agent_workflows import (
    FindModelWorkflow,
    ExecuteModelWorkflow,
)


class TrainerAgent:
    def __init__(
        self,
        model: str,
        dataset_path: str,
        column_descriptions: dict,
        models_path: str,
        temperature: int = 0.3,
    ):
        self.model = model
        self.dataset_path = dataset_path
        self.column_descriptions = column_descriptions
        self.models_path = models_path
        self.temperature = temperature
        self.dataset = pd.read_csv(self.dataset_path)

    async def auto_train(self):
        workflow = FindModelWorkflow(
            timeout=1000,
        )
        handler = workflow.run(
            model=self.model,
            column_descriptions=self.column_descriptions,
            temperature=self.temperature,
        )
        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                print("We have generated the following models:\n")
                print(event.result)
                response = input(event.prefix)
                handler.ctx.send_event(HumanResponseEvent(response=response))
        response = await handler
        print("Agent complete! Here's your final result :")
        print(str(response))

        workflow = ExecuteModelWorkflow(
            timeout=1000,
        )
        handler = workflow.run(
            model_description=response,
            dataset=self.dataset,
            models_name=self.models_path,
        )
        response = await handler
        print("Agent complete! Here's your final result :")
        print(str(response))
