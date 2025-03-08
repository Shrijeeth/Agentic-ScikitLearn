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
from textwrap import dedent


# Define prompts
FIND_MODEL_PROMPT = """
You are a helpful assistant that is responsible for finding the best ML models for a given data.
The user will provide some information like data column name, description of the column and data type of the column about the data and you need to find the best traditional machine learning models for it.
Search the internet for relevant information regarding the models provided by scikit learn library and give the model names and suggest the features and labels that can be used.
There can be multiple feature-label pairs and each feature-label pair can be used to build multiple machine learning models.
The column descriptions will be given in following format:
    ```
    {{
        column_name1: (column_description1, column_type1),
        column_name2: (column_description2, column_type2),
        ...
    }}
    ```
Only give the feature-label pairs that you can use to build machine learning models along with model names in Scikit-Learn strictly.
You can many features and many labels possible from a column. Make sure it is practically possible and feasible to use all of them.
The output should be a JSON object with model names as keys and feature-label pairs as values.
The feature-label pairs should be strictly chosen from column_descriptions and it should not be chosen from elsewhere.
Try to use all columns possible from the column descriptions as much as possible.
The output should be in following format:
    ```
    {{
        model_name1: [
            {{
                features: [feature1, feature2, ...],
                labels: [label1, label2, ...]
            }},
            {{
                features: [feature1, feature2, ...],
                labels: [label1, label2, ...]
            }},
            ...
        ],
        model_name2: [
            {{
                features: [feature1, feature2, ...],
                labels: [label1, label2, ...]
            }},
            ...
        ],
        ...
    }}
    ```

Here is the column descriptions:
{column_descriptions}
"""


# Setup the events for the workflow
class DataAnalyzingEvent(Event):
    pass


class FeedbackEvent(Event):
    feedback: str


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
        Finds the best ML models for the given data based on the feedback provided.
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
            prefix="How does this look? Give me any feedback you have on any of the answers.",
            result=agent_results,
        )

    @step
    async def get_feedback(
        self, ctx: Context, ev: HumanResponseEvent
    ) -> FeedbackEvent | StopEvent:
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
            return StopEvent(result=await ctx.get("agent_results"))
        else:
            return FeedbackEvent(feedback=ev.response)
