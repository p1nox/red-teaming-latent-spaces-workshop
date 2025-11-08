from IPython.display import display, Markdown


def display_chat_response(content):
    """
    Display LLM response properly, including "chain-of-thought" reasoning phase.
    Works with both OpenAI and Ollama models.
    """
    if "<think>" not in content:
        return display(Markdown(f"ANSWER: {content}"))

    parts = content.split("</think>")
    reasoning_text = parts[0].replace("<think>", "").strip()
    final_output = parts[1].strip()
    reasoning_array = reasoning_text.split("\n")
    formatted_content = f"THINK: {reasoning_text}\n\nANSWER: {final_output}"
    return display(Markdown(formatted_content))
