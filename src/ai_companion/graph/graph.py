from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from ai_companion.graph.edges import (
    select_workflow,
    should_summarize_conversation,
)
from ai_companion.graph.nodes import (
    audio_node,
    context_injection_node,
    conversation_node,
    image_node,
    memory_extraction_node,
    memory_injection_node,
    router_node,
    summarize_conversation_node,
)
from ai_companion.graph.state import AICompanionState


@lru_cache(maxsize=1)
def create_workflow_graph():
    """
    Creates a workflow graph for processing user interactions in an AI companion system. 
    The function initializes a StateGraph using AICompanionState and defines various 
    nodes representing different stages of processing, such as memory extraction, 
    context injection, and conversation handling.

    The graph starts by extracting memory from the user's message. It then determines 
    the type of response required through a routing mechanism. Context and memory 
    are injected into the response, after which the workflow proceeds to the appropriate 
    response node based on whether the input is conversational, image-based, or audio-based.

    After generating a response, the workflow checks if the conversation should be 
    summarized. If required, the summarization node is triggered, after which the workflow 
    ends. The function is cached to ensure that the graph is created only once, improving 
    efficiency.

    The final compiled graph is used in LangGraph Studio for structured AI-driven 
    conversations.
    """
    
    graph_builder = StateGraph(AICompanionState)

    # Add all nodes
    graph_builder.add_node("memory_extraction_node", memory_extraction_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("context_injection_node", context_injection_node)
    graph_builder.add_node("memory_injection_node", memory_injection_node)
    graph_builder.add_node("conversation_node", conversation_node)
    graph_builder.add_node("image_node", image_node)
    graph_builder.add_node("audio_node", audio_node)
    graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)

    # Define the flow
    # First extract memories from user message
    graph_builder.add_edge(START, "memory_extraction_node")

    # Then determine response type
    graph_builder.add_edge("memory_extraction_node", "router_node")

    # Then inject both context and memories
    graph_builder.add_edge("router_node", "context_injection_node")
    graph_builder.add_edge("context_injection_node", "memory_injection_node")

    # Then proceed to appropriate response node
    graph_builder.add_conditional_edges("memory_injection_node", select_workflow)

    # Check for summarization after any response
    graph_builder.add_conditional_edges("conversation_node", should_summarize_conversation)
    graph_builder.add_conditional_edges("image_node", should_summarize_conversation)
    graph_builder.add_conditional_edges("audio_node", should_summarize_conversation)
    graph_builder.add_edge("summarize_conversation_node", END)

    return graph_builder


# Compiled without a checkpointer. Used for LangGraph Studio
graph = create_workflow_graph().compile()