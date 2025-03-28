from langgraph.graph import END
from typing_extensions import Literal

from ai_companion.graph.state import AICompanionState
from ai_companion.settings import settings


def should_summarize_conversation(
    state: AICompanionState,
) -> Literal["summarize_conversation_node", "__end__"]:
    '''
    This function checks the number of messages in the conversation. 
    If the count exceeds a predefined threshold (set in settings), 
    it triggers a summarization process by returning "summarize_conversation_node". 
    Otherwise, it ends the workflow.
    '''
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END


def select_workflow(
    state: AICompanionState,
) -> Literal["conversation_node", "image_node", "audio_node"]:
    '''
    This function determines which workflow to follow based on the current state. 
    If the workflow type is "image", it routes to an image-processing node. 
    If it’s "audio", it goes to an audio-processing node. 
    Otherwise, it defaults to a standard conversation node.
    '''
    workflow = state["workflow"]

    if workflow == "image":
        return "image_node"

    elif workflow == "audio":
        return "audio_node"

    else:
        return "conversation_node"