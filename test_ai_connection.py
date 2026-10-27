
from ai_assistant import ask_expense_assistant

if __name__ == "__main__":
    summary = """
    Total monthly expenses: Rs 33,000
    Food: Rs 12,000
    Transport: Rs 8,000
    Shopping: Rs 7,000
    Bills: Rs 6,000
    """

    print("Testing local Ollama AI...")
    answer = ask_expense_assistant(summary)

    print("\nAI Assistant Response:")
    print(answer)
