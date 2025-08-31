import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

@traceable  # LangSmith decorator to log runs
def answer_basic_question(query: str):
    response = llm.invoke(query)
    return response.content

if __name__ == "__main__":
    query = "What are the top skills for AI engineers in 2025?"
    print(answer_basic_question(query))
