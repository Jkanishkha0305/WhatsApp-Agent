from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from ai_companion.core.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT
from ai_companion.graph.utils.helpers import AsteriskRemovalParser, get_chat_model


class RouterResponse(BaseModel):
    """
    Defines the structured response format for the router model.

    Attributes:
        response_type (str): The type of response to generate. 
            Must be one of: 'conversation', 'image', or 'audio'.
    """
    response_type: str = Field(
        description="The response type to give to the user. It must be one of: 'conversation', 'image' or 'audio'"
    )


def get_router_chain():
    """
    Creates a router chain that selects the appropriate response type.

    Uses a structured output model to determine whether the response should be 
    conversational text, an image, or audio.

    Returns:
        A LangChain pipeline combining a system prompt with a structured output model.
    """
    model = get_chat_model(temperature=0.3).with_structured_output(RouterResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )

    return prompt | model


def get_character_response_chain(summary: str = ""):
    """
    Creates a character response chain that generates a conversational reply.

    This chain uses a system prompt and optionally incorporates a summary of prior 
    interactions for continuity.

    Parameters:
        summary (str, optional): A summary of past interactions to maintain conversation flow. 
                                 Defaults to an empty string.

    Returns:
        A LangChain pipeline that processes user messages and generates responses 
        while removing unwanted asterisk-enclosed content.
    """
    model = get_chat_model()
    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"\n\nSummary of conversation earlier between Ava and the user: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model | AsteriskRemovalParser()