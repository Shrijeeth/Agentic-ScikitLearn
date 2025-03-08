import asyncio
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent
from agentic_scikitlearn.workflows.agent_workflows import FindModelWorkflow


async def main():
    workflow = FindModelWorkflow(
        timeout=1000,
    )
    handler = workflow.run(
        model="gemini/gemini-2.0-flash-lite-preview-02-05",
        column_descriptions={
            "Date": ("The transaction date of the chocolate sale.", "datetime"),
            "Product Name": ("Name of the chocolate product sold.", "str"),
            "Category": ("Type of chocolate (Dark, Milk, White).", "str"),
            "Quantity Sold": (
                "Number of chocolate units sold in the transaction.",
                "int",
            ),
            "Revenue": ("Total revenue generated from the sale.", "float"),
            "Customer Segment": ("Type of customer (Retail, Wholesale).", "str"),
            "Location": (
                "Sales region or store location where the transaction took place.",
                "str",
            ),
        },
        temperature=0.1,
    )
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            print("We have generated the following models:\n")
            print(event.result)
            response = input(event.prefix)
            handler.ctx.send_event(HumanResponseEvent(response=response))
    response = await handler
    print("Agent complete! Here's your final result:")
    print(str(response))


if __name__ == "__main__":
    asyncio.run(main())
