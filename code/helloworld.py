import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("🦜🔗 Hello LangChain!\n")

    # Create a chat model instance
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Make your first AI call!
    response = model.invoke("Explain AI in simple terms?")

    print("🤖 AI Response:", response.content)
    print("\n✅ Success! You just made your first LangChain call!")


if __name__ == "__main__":
    main()
