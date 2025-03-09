# Agentic-ScikitLearn

Agentic-ScikitLearn is an innovative machine learning workflow automation library that leverages AI agents to streamline the process of model training, feature selection, and model discovery using ScikitLearn.

## Key Features

- 🤖 **AI-Powered Model Discovery**: Automatically identify the most suitable machine learning models for your dataset
- 🔍 **Intelligent Feature Selection**: Dynamically analyze dataset characteristics to recommend optimal features
- 🚀 **Multi-Model Training**: Support for regression, classification, and time series models
- 🧠 **LLM-Driven Workflow**: Utilizes advanced language models to guide the machine learning process

## Installation

```bash
pip install agentic-scikitlearn
```

## Quick Start

```python
import asyncio
from agentic_scikitlearn.trainer_agent import TrainerAgent

async def main():
    agent = TrainerAgent(
        model="gemini/gemini-2.0-flash-lite-preview-02-05",
        dataset_path="path/to/your/dataset.csv",
        column_descriptions={
            "column1": ("Description of column1", "data_type"),
            "column2": ("Description of column2", "data_type"),
        },
        models_path="output_models_directory"
    )
    await agent.auto_train()

if __name__ == "__main__":
    asyncio.run(main())
```

## Workflow

1. **Data Analysis**: AI agent analyzes your dataset
2. **Model Discovery**: Recommends optimal machine learning models
3. **Feature Selection**: Intelligently selects best features
4. **Model Training**: Automatically trains and evaluates models

## Supported Model Types

- Regression Models
- Classification Models
- Time Series Models

## Dependencies

- Python 3.8+
- Scikit-Learn
- LlamaIndex
- PyCaret
- LiteLLM

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Scikit-Learn
- LlamaIndex
- PyCaret
- LiteLLM

## Contact

For support, please open an issue on our GitHub repository or contact the maintainers.
