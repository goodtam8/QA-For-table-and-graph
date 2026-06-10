import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Use your local file path
CONTEXT_PATH = Path(
    "/Users/goodtam8/Documents/Programming/QA-For-table-and-graph/hsbc_output/hsbc.md"
)


def load_context() -> str:
    return CONTEXT_PATH.read_text(encoding="utf-8")


def answerdirectly(question: str) -> str:
    context = load_context()

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful QA assistant. "
                    "Answer the user's question using only the provided HSBC Markdown document. "
                    "If the answer is not in the document, say that you cannot find it in the provided document."
                ),
            },
            {
                "role": "user",
                "content": f"""
Here is the HSBC Markdown document:

```markdown
{context}
```

User question:
{question}
""",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    question = input("Ask a question about the HSBC document: ")
    answer = answerdirectly(question)
    print(answer)