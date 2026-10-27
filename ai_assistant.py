import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b-instruct"


def ask_expense_assistant(expense_summary):
    """
    Ask the local Ollama model to explain expense information.
    Python remains responsible for all financial calculations.
    """

    prompt = f"""
You are a helpful personal expense assistant.

Explain the following expense analysis in simple English.
Give practical, realistic ways to reduce unnecessary spending.
Use only the supplied information. Do not invent numbers.
Do not claim that an expense is suspicious unless the analysis says so.
Keep the answer concise and easy to understand.

Expense analysis:
{expense_summary}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 180
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        return result.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return (
            "Ollama is not running. Start Ollama and try again."
        )

    except requests.exceptions.Timeout:
        return (
            "The local AI took too long to respond. Please try again."
        )

    except requests.exceptions.RequestException as error:
        return f"Could not contact the local AI: {error}"