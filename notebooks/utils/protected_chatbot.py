import os
import ollama
from openai import OpenAI
from utils.formatting import display_chat_response
from dotenv import load_dotenv
from guardrails import Guard
from guardrails.hub import DetectJailbreak, UnusualPrompt

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class OllamaChatBot:
    """Chatbot implementation using Ollama models."""

    def __init__(self, model: str, system_prompt: str = None, guard: Guard = None):
        self.model = model
        self.system_prompt = system_prompt
        self.guard = guard
        self.messages = []
        self.init_conversation()

    def init_conversation(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def ask(self, user_input: str, display_only=False) -> str:
        if self.guard is not None:
            try:
                print(f" > Validating message prompt first")
                self.guard.validate(user_input)
                print(f" > Message prompt passed guardrail validation")
            except Exception as e:
                error_msg = f"⚠️ Guardrail blocked this message: {str(e)}"
                if display_only:
                    return display_chat_response(error_msg)
                return error_msg

        self.messages.append({"role": "user", "content": user_input})
        response = ollama.chat(model=self.model, messages=self.messages)
        self.messages.append(
            {"role": "assistant", "content": response["message"]["content"]}
        )
        content = response["message"]["content"]
        if display_only:
            return display_chat_response(content)
        return content

    def clear_history(self):
        self.init_conversation()


class OpenAIChatBot:
    """Chatbot implementation using OpenAI models."""

    def __init__(self, model: str, system_prompt: str = None, guard: Guard = None):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        self.model = model
        self.system_prompt = system_prompt
        self.guard = guard
        self.messages = []
        self.init_conversation()

    def init_conversation(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def ask(self, user_input: str, display_only=False) -> str:
        if self.guard is not None:
            try:
                print(f" > Validating message prompt first")
                self.guard.validate(user_input)
                print(f" > Message prompt passed guardrail validation")
            except Exception as e:
                error_msg = f"⚠️ Guardrail blocked this message: {str(e)}"
                if display_only:
                    return display_chat_response(error_msg)
                return error_msg

        self.messages.append({"role": "user", "content": user_input})
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        response_content = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_content})
        if display_only:
            return display_chat_response(response_content)
        return response_content

    def clear_history(self):
        self.init_conversation()


class ChatBot:
    """
    Unified chatbot class that can use either Ollama or OpenAI as the backend provider.

    Args:
        model: The model name to use (e.g., "gemma3:4b" for Ollama or "gpt-4" for OpenAI)
        system_prompt: Optional system prompt to initialize the conversation
        provider: The backend provider to use, either "ollama" or "openai" (default: "ollama")

    Example:
        # Using Ollama
        chatbot = ChatBot(model="gemma3:4b", system_prompt="You are a helpful assistant", provider="ollama")

        # Using OpenAI
        chatbot = ChatBot(model="gpt-4", system_prompt="You are a helpful assistant", provider="openai")
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = None,
        provider: str = "ollama",
        protected: bool = False,
    ):
        self.provider = provider.lower()
        self.protected = protected

        guard = (
            Guard().use(DetectJailbreak()).use(UnusualPrompt, on_fail="exception")
            if protected
            else None
        )

        if self.provider == "ollama":
            self._bot = OllamaChatBot(
                model=model, system_prompt=system_prompt, guard=guard
            )
        elif self.provider == "openai":
            self._bot = OpenAIChatBot(
                model=model, system_prompt=system_prompt, guard=guard
            )
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Must be 'ollama' or 'openai'"
            )

    def ask(self, user_input: str, display_only=False) -> str:
        """
        Send a message to the chatbot and get a response.

        Args:
            user_input: The user's message
            display_only: If True, formats the output for display (default: False)

        Returns:
            The chatbot's response
        """
        return self._bot.ask(user_input, display_only=display_only)

    def clear_history(self):
        """Clear the conversation history."""
        self._bot.clear_history()

    def init_conversation(self):
        """Initialize or reset the conversation."""
        self._bot.init_conversation()
