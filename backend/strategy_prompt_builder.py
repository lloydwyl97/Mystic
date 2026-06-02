import os

import openai


def generate_strategy_code(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY environment variable not set"
        raise OSError(msg)

    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a crypto trading strategy coder."},
            {"role": "user", "content": f"Generate a Python strategy: {prompt}"},
        ],
    )

    # Handle different response shapes (dict-like or attribute access)
    try:
        return response["choices"][0]["message"]["content"]
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        try:
            return response.choices[0].message.content
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return str(response)
