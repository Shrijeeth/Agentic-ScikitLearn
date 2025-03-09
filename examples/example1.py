import asyncio
from agentic_scikitlearn.trainer_agent import TrainerAgent


async def main():
    agent = TrainerAgent(
        model="gemini/gemini-2.0-flash-lite-preview-02-05",
        dataset_path="/Users/shrijeet/personal/Agentic-ScikitLearn/examples/data/Salary_dataset.csv",
        column_descriptions={
            "YearsExperience": ("Years of experience.", "float"),
            "Salary": ("Salary of the employee.", "float"),
        },
        models_path="sample_test",
    )
    await agent.auto_train()


if __name__ == "__main__":
    asyncio.run(main())
