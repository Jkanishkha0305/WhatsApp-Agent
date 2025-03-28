import re

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from ai_companion.modules.image.image_to_text import ImageToText
from ai_companion.modules.image.text_to_image import TextToImage
from ai_companion.modules.speech import TextToSpeech
from ai_companion.settings import settings


def get_chat_model(temperature: float = 0.7):
    """"
    Initializes and returns a ChatGroq model for text-based AI interactions.

    Parameters:
        temperature (float): Controls randomness of responses (default: 0.7).

    Returns:
        ChatGroq: A configured ChatGroq object for text generation.
    """
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.TEXT_MODEL_NAME,
        temperature=temperature,
    )


def get_text_to_speech_module():
    """
    Initializes and returns a text-to-speech module.

    Returns:
        TextToSpeech: A module for converting text to spoken audio.
    """
    return TextToSpeech()


def get_text_to_image_module():
    """
    Initializes and returns a text-to-image module.

    Returns:
        TextToImage: A module for generating images from text prompts.
    """
    return TextToImage()


def get_image_to_text_module():
    """
    Initializes and returns an image-to-text module.

    Returns:
        ImageToText: A module for extracting text descriptions from images.
    """
    return ImageToText()


def remove_asterisk_content(text: str) -> str:
    """
    Removes text enclosed within asterisks (*) from the given string.

    Parameters:
        text (str): Input string that may contain asterisk-enclosed content.

    Returns:
        str: The cleaned string with asterisk-marked content removed.
    """
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    """
    A custom parser that removes text enclosed in asterisks from model-generated responses.
    """
    def parse(self, text):
        """
        Cleans the output by removing any asterisk-enclosed content before returning it.

        Parameters:
            text (str): Input string from the model response.

        Returns:
            str: A cleaned string with asterisk-marked content removed.
        """
        return remove_asterisk_content(super().parse(text))