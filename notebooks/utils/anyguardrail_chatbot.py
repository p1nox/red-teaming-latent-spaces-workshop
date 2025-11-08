import os
import ollama
from openai import OpenAI
from utils.formatting import display_chat_response
from dotenv import load_dotenv
from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class OllamaChatBot:
    """Chatbot implementation using Ollama models with AnyGuardrail protection."""

    def __init__(
        self, model: str, system_prompt: str = None, guardrail: AnyGuardrail = None
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.guardrail = guardrail
        self.messages = []
        self.init_conversation()

    def init_conversation(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def ask(self, user_input: str, display_only=False) -> str:
        if self.guardrail is not None:
            print(f" > Validating message with AnyGuardrail")
            result: GuardrailOutput = self.guardrail.validate(user_input)
            if not result.valid:
                error_msg = f"⚠️ Guardrail blocked this message: {result.explanation}"
                print(f" > Message blocked by guardrail")
                if display_only:
                    return display_chat_response(error_msg)
                return error_msg
            print(f" > Message passed guardrail validation")

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
    """Chatbot implementation using OpenAI models with AnyGuardrail protection."""

    def __init__(
        self, model: str, system_prompt: str = None, guardrail: AnyGuardrail = None
    ):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        self.model = model
        self.system_prompt = system_prompt
        self.guardrail = guardrail
        self.messages = []
        self.init_conversation()

    def init_conversation(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def ask(self, user_input: str, display_only=False) -> str:
        if self.guardrail is not None:
            print(f" > Validating message with AnyGuardrail")
            result: GuardrailOutput = self.guardrail.validate(user_input)
            if not result.valid:
                error_msg = f"⚠️ Guardrail blocked this message: {result.explanation}"
                print(f" > Message blocked by guardrail")
                if display_only:
                    return display_chat_response(error_msg)
                return error_msg
            print(f" > Message passed guardrail validation")

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


class AnyGuardrailChatBot:
    """
    Unified chatbot class that can use either Ollama or OpenAI as the backend provider,
    protected by AnyGuardrail.

    Args:
        model: The model name to use (e.g., "gemma3:4b" for Ollama or "gpt-4" for OpenAI)
        system_prompt: Optional system prompt to initialize the conversation
        provider: The backend provider to use, either "ollama" or "openai" (default: "ollama")
        protected: Whether to enable AnyGuardrail protection (default: False)
        guardrail_name: The guardrail to use (default: GuardrailName.DEEPSET)

    Example:
        # Using Ollama with protection
        chatbot = AnyGuardrailChatBot(
            model="gemma3:4b",
            system_prompt="You are a helpful assistant",
            provider="ollama",
            protected=True
        )

        # Using OpenAI with protection and custom guardrail
        chatbot = AnyGuardrailChatBot(
            model="gpt-4",
            system_prompt="You are a helpful assistant",
            provider="openai",
            protected=True,
            guardrail_name=GuardrailName.DEEPSET
        )
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = None,
        provider: str = "ollama",
        protected: bool = False,
        guardrail_name: GuardrailName = GuardrailName.DEEPSET,
    ):
        self.provider = provider.lower()
        self.protected = protected

        guardrail = AnyGuardrail.create(guardrail_name) if protected else None

        if self.provider == "ollama":
            self._bot = OllamaChatBot(
                model=model, system_prompt=system_prompt, guardrail=guardrail
            )
        elif self.provider == "openai":
            self._bot = OpenAIChatBot(
                model=model, system_prompt=system_prompt, guardrail=guardrail
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
