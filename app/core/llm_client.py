# llm_client.py
# This module is the single point of contact between our project
# and the Groq LLM API.
#
# Every part of the project that needs to talk to an AI model
# goes through this file. Nobody calls Groq directly.
# This is the professional pattern called a "wrapper" or "adapter".

from groq import Groq
from app.utils.config import settings
from app.utils.logger import get_logger

# Create a logger specifically for this module
# __name__ will be "app.core.llm_client" automatically
logger = get_logger(__name__)


class LLMClient:
    """
    Wrapper around the Groq API.

    Handles:
    - Initialization with API key
    - Single turn conversations (one question, one answer)
    - Multi turn conversations (full chat history)
    - Error handling so crashes are informative
    - Logging so every call is recorded
    """

    def __init__(self):
        """
        Initialize the LLM client.
        This runs once when the object is created.
        """

        # Validate settings first — fail fast if API key missing
        settings.validate()

        # Create the Groq client using our API key
        self.client = Groq(api_key=settings.GROQ_API_KEY)

        # Store the model name for use in all API calls
        self.model = settings.GROQ_MODEL

        logger.info(f"LLMClient ready | model={self.model}")

    def chat(
        self,
        user_message: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send one message and get one response back.
        This is the simplest way to talk to the LLM.

        Args:
            user_message:  The question or input from the user
            system_prompt: Instructions that define how the AI behaves.
                          This is where we inject domain-specific rules.
            temperature:  Controls creativity vs consistency.
                          0.0 = always gives same answer (good for facts)
                          1.0 = creative and varied (good for writing)
                          We use 0.1 for document Q&A — consistent answers
            max_tokens:   Maximum length of the response.
                          1 token is roughly 4 characters or 0.75 words.

        Returns:
            The AI's response as a plain string
        """

        logger.info(
            f"Sending message to LLM | "
            f"model={self.model} | "
            f"temperature={temperature} | "
            f"message_length={len(user_message)}"
        )

        try:
            # Make the API call to Groq
            # messages is a list of dicts with role and content
            # role "system" = instructions for the AI
            # role "user"   = the human's message
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract the text response from the API response object
            answer = response.choices[0].message.content

            # Log token usage — important for monitoring costs
            # in production systems
            usage = response.usage
            logger.info(
                f"LLM response received | "
                f"prompt_tokens={usage.prompt_tokens} | "
                f"completion_tokens={usage.completion_tokens} | "
                f"total_tokens={usage.total_tokens}"
            )

            return answer

        except Exception as e:
            # Never let a raw error crash silently
            # Always log what went wrong with full context
            logger.error(f"LLM call failed | error={type(e).__name__} | {e}")
            raise RuntimeError(f"LLM call failed: {e}") from e

    def chat_with_history(
        self,
        messages: list,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a full conversation history and get the next response.
        Used for multi-turn chat where the AI needs to remember
        what was said earlier in the conversation.

        Args:
            messages: List of previous messages in this format:
                     [
                       {"role": "user", "content": "Hello"},
                       {"role": "assistant", "content": "Hi there!"},
                       {"role": "user", "content": "How are you?"}
                     ]
            system_prompt: Instructions for the AI's behavior

        Returns:
            The AI's next response as a plain string
        """

        logger.info(
            f"Sending conversation to LLM | "
            f"turns={len(messages)}"
        )

        try:
            # Combine system prompt with conversation history
            full_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content

            logger.info(
                f"Multi-turn response received | "
                f"total_tokens={response.usage.total_tokens}"
            )

            return answer

        except Exception as e:
            logger.error(f"Multi-turn LLM call failed | error={e}")
            raise RuntimeError(f"LLM call failed: {e}") from e


# Create one single instance to be shared across the entire project
# This means Groq client is initialized exactly once at startup
llm_client = LLMClient()