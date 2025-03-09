import json
import pandas as pd
import uuid
from llama_index.core.agent import ReActAgent
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
    InputRequiredEvent,
    HumanResponseEvent,
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.litellm import LiteLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from pycaret import regression, classification, time_series
from textwrap import dedent


# Define prompts
FIND_MODEL_PROMPT = """
You are a helpful assistant that is responsible for finding the best ML models for a given data.
The user will provide some information like data column name, description of the column and data type of the column about the data and you need to find the best traditional machine learning models for it.
Search the internet for relevant information about data descriptions given and give the model names (regression, classification, time_series) and suggest the features and labels that can be used.

There can be multiple feature-label pairs and each feature-label pair can be used to build multiple machine learning models.
For time series models alone there will be one feature (time) and one label. For others there will be multiple features and single label.
The column descriptions will be given in following format:
    ```
    {{
        column_name1: (column_description1, column_type1),
        column_name2: (column_description2, column_type2),
        ...
    }}
    ```

Only give the feature-label pairs that you can use to build machine learning models along with model names strictly.
You can many have features and labels possible from columns list. Make sure it is practically possible and feasible to use all of them.
The output should be a JSON object with model names as keys and feature-label pairs as values.
The feature-label pairs should be strictly chosen from column_descriptions and it should not be chosen from elsewhere.
One feature-label pair must not have a same column common to both of them strictly. So if a feature has `column_1` then corresponding label should not have `column_1`.
Avoid adding duplicate feature-label pairs.
Try to use all columns possible from the column descriptions as much as possible and decide these logically by analyzing each and every column carefully based on its description and datatype.
The output should be in following format:
    ```
    {{
        regression: [
            {{
                features: [feature1, feature2, ...],
                labels: label1
            }},
            {{
                features: [feature1, feature2, ...],
                labels: label2
            }},
            ...
        ],
        classification: [
            {{
                features: [feature1, feature2, ...],
                labels: label1
            }},
            ...
        ],
        time_series: [
            {{
                features: feature1,
                labels: label1
            }},
            ...
        ]
    }}
    ```
You should follow the above rules strictly. If you follow all the rules correctly, then I will reward you with $10000.

Here is the column descriptions:
{column_descriptions}
"""


# Setup the events for the workflows
class DataAnalyzingEvent(Event):
    pass


class FeedbackEvent(Event):
    feedback: str


class ParseModelDescriptionEvent(Event):
    model_description: str


class TrainRegressionModelsEvent(Event):
    data: pd.DataFrame
    target_column: str


class TrainClassificationModelsEvent(Event):
    data: pd.DataFrame
    target_column: str


class TrainTimeSeriesModelsEvent(Event):
    data: pd.DataFrame


class ModelResponseEvent(Event):
    model_uuid: str
    model_name: str


# Define the workflows
class FindModelWorkflow(Workflow):
    """
    Workflow for finding the best model for a given data.
    """

    agent: ReActAgent
    llm: LiteLLM

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> DataAnalyzingEvent:
        """
        Sets up the workflow by getting the LLM and the agent.
        """

        if not ev.model:
            raise ValueError("No model name provided")
        if not ev.temperature or not isinstance(ev.temperature, float):
            raise ValueError("temperature should be a float")
        if not ev.column_descriptions or not isinstance(ev.column_descriptions, dict):
            raise ValueError("No column descriptions provided")

        self.llm = LiteLLM(
            model=str(ev.model),
            temperature=float(ev.temperature) if ev.temperature else 0.3,
        )
        self.agent = ReActAgent(
            llm=self.llm,
            tools=DuckDuckGoSearchToolSpec().to_tool_list(),
            memory=ChatMemoryBuffer(token_limit=8192),
            verbose=True,
        )

        await ctx.set("column_descriptions", ev.column_descriptions)
        return DataAnalyzingEvent()

    @step
    async def find_models(
        self, ctx: Context, ev: DataAnalyzingEvent | FeedbackEvent
    ) -> InputRequiredEvent:
        """
        Finds the best ML model types for the given data based on the feedback provided.
        """
        previous_response = await ctx.get("agent_results", None)
        if hasattr(ev, "feedback") and previous_response:
            message = ev.feedback
            chat_history = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=dedent(FIND_MODEL_PROMPT).format(
                        column_descriptions=await ctx.get("column_descriptions"),
                    ),
                ),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=previous_response,
                ),
            ]
        else:
            message = dedent(FIND_MODEL_PROMPT).format(
                column_descriptions=await ctx.get("column_descriptions"),
            )
            chat_history = None

        agent_results = await self.agent.achat(
            message=message,
            chat_history=chat_history,
        )
        await ctx.set("agent_results", agent_results)
        return InputRequiredEvent(
            prefix="How does this look? Give me any feedback you have on any of the answers:\n",
            result=agent_results,
        )

    @step
    async def get_feedback(
        self, ctx: Context, ev: HumanResponseEvent
    ) -> FeedbackEvent | ParseModelDescriptionEvent:
        """
        Gets the feedback from the user.
        """

        result = self.llm.complete(f"""
            You have received some human feedback on the form-filling task you've done.
            Does everything look good, or is there more work to be done?
            <feedback>
            {ev.response}
            </feedback>
            If everything is fine, respond with just the word 'OKAY'.
            If there's any other feedback, respond with just the word 'FEEDBACK'.
        """)

        verdict = result.text.strip()

        print(f"LLM says the verdict was {verdict}")
        if "OKAY" in verdict:
            agent_results = await ctx.get("agent_results")
            return ParseModelDescriptionEvent(
                model_description=str(agent_results),
            )
        else:
            return FeedbackEvent(feedback=ev.response)

    @step
    async def parse_response(
        self, ctx: Context, ev: ParseModelDescriptionEvent
    ) -> StopEvent | FeedbackEvent:
        """
        Parses the response from the LLM to json.
        """
        result = ev.model_description
        result = result.split("```json")
        if len(result) <= 1:
            return FeedbackEvent(
                feedback="Current response is not in json format. It does not contain ```json at all."
            )
        result = result[1]
        result = result.split("```")
        if len(result) < 1:
            return FeedbackEvent(
                feedback="Current response is not in proper json format. Ending sequence ``` not found."
            )
        result = result[0]
        try:
            duplicates = []
            result = json.loads(result)
            for key, value in result.items():
                for val in value:
                    if isinstance(val["labels"], list):
                        val["labels"] = val["labels"][0]
                    if val["labels"] in val["features"]:
                        val["features"].remove(val["labels"])
                    if val["features"] == []:
                        raise ValueError(
                            "No features provided for label " + val["labels"]
                        )
                    if (val["features"], val["labels"]) in duplicates:
                        return FeedbackEvent(
                            feedback="Duplicate features and labels found. Please remove duplicates and try again."
                        )
                    duplicates.append((val["features"], val["labels"]))
        except Exception as e:
            return FeedbackEvent(
                feedback=f"Current response is not in proper json format as mentioned. Getting error while loading json: {e}"
            )
        return StopEvent(result=result)


class ExecuteModelWorkflow(Workflow):
    @step
    async def set_up(
        self, ctx: Context, ev: StartEvent
    ) -> (
        TrainRegressionModelsEvent
        | TrainClassificationModelsEvent
        | TrainTimeSeriesModelsEvent
    ):
        if not ev.model_description or not isinstance(ev.model_description, dict):
            raise ValueError("No model description provided")
        if ev.dataset is None or not isinstance(ev.dataset, pd.DataFrame):
            raise ValueError("No dataset provided")
        if not ev.models_name or not isinstance(ev.models_name, str):
            raise ValueError("No models name provided")

        models_trained = 0

        for key, value in ev.model_description.items():
            if key == "regression":
                for model_params in value:
                    columns = model_params["features"] + [model_params["labels"]]
                    data = ev.dataset.drop(
                        columns=[
                            column
                            for column in ev.dataset.columns
                            if column not in columns
                        ],
                        inplace=False,
                    )
                    ctx.send_event(
                        TrainRegressionModelsEvent(
                            data=data,
                            target_column=model_params["labels"],
                        )
                    )

            if key == "classification":
                for model_params in value:
                    columns = model_params["features"] + [model_params["labels"]]
                    data = ev.dataset.drop(
                        columns=[
                            column
                            for column in ev.dataset.columns
                            if column not in columns
                        ],
                        inplace=False,
                    )
                    ctx.send_event(
                        TrainClassificationModelsEvent(
                            data=data,
                            target_column=model_params["labels"],
                        )
                    )

            if key == "time_series":
                for model_params in value:
                    columns = [model_params["features"], model_params["labels"]]
                    data = ev.dataset.drop(
                        columns=[
                            column
                            for column in ev.dataset.columns
                            if column not in columns
                        ],
                        inplace=False,
                    )
                    ctx.send_event(
                        TrainTimeSeriesModelsEvent(
                            data=data,
                        )
                    )

            models_trained += len(value)

        await ctx.set("models_trained", models_trained)
        await ctx.set("models_name", ev.models_name)
        return

    @step
    async def train_regression_models(
        self, ctx: Context, ev: TrainRegressionModelsEvent
    ) -> ModelResponseEvent:
        model_uuid = str(uuid.uuid4())
        regression.setup(
            data=ev.data,
            target=ev.target_column,
        )
        best = regression.compare_models()
        regression.evaluate_model(estimator=best)
        model_name = await ctx.get("models_name")
        model_name += f"-regression-{model_uuid}"
        regression.save_model(
            best,
            model_name,
        )
        return ModelResponseEvent(
            model_uuid=model_uuid,
            model_name=model_name,
        )

    @step
    async def train_classification_models(
        self, ctx: Context, ev: TrainClassificationModelsEvent
    ) -> ModelResponseEvent:
        model_uuid = str(uuid.uuid4())
        classification.setup(
            data=ev.data,
            target=ev.target_column,
        )
        best = classification.compare_models()
        classification.evaluate_model(estimator=best)
        model_name = await ctx.get("models_name")
        model_name += f"-classification-{model_uuid}"
        regression.save_model(
            best,
            model_name,
        )
        return ModelResponseEvent(
            model_uuid=model_uuid,
            model_name=model_name,
        )

    @step
    async def train_timeseries_models(
        self, ctx: Context, ev: TrainTimeSeriesModelsEvent
    ) -> ModelResponseEvent:
        model_uuid = str(uuid.uuid4())
        time_series.setup(
            data=ev.data,
        )
        best = time_series.compare_models()
        time_series.evaluate_model(estimator=best)
        model_name = await ctx.get("models_name")
        model_name += f"-timeseries-{model_uuid}"
        time_series.save_model(
            best,
            model_name,
        )
        return ModelResponseEvent(
            model_uuid=model_uuid,
            model_name=model_name,
        )

    @step
    async def gather_trained_models(
        self, ctx: Context, ev: ModelResponseEvent
    ) -> StopEvent:
        models_trained = await ctx.get("models_trained")

        responses = ctx.collect_events(ev, [ModelResponseEvent] * models_trained)
        if responses is None:
            return None

        results = []
        for response in responses:
            results.append(response.model_name)

        return StopEvent(result=str(results))
