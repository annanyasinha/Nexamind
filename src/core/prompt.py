"""
Central prompt templates and prompt builder utilities for RAG platform.
"""

from typing import List, Dict, Optional, Tuple

RAG_SYSTEM_INSTRUCTION = """You are a helpful RAG AI Assistant. Answer the user query using the provided document context and conversation history. Maintain conversational context if the user asks follow-up questions."""

RAG_RESPONSE_PROMPT_TEMPLATE = """{system_instruction}

{history_text}Retrieved Document Context:
{context}

Current User Query: {query}

Answer/Summary:"""

NO_CONTEXT_FOUND_MESSAGE = "No relevant documents found in the vector store."


def format_chat_history(chat_history: Optional[List[Dict[str, str]]] = None, max_turns: int = 6) -> Tuple[str, str]:
    """
    Formats recent chat history turns for inclusion into the prompt context and returns
    a tuple of (formatted_history_text, augmented_retrieval_query_suffix).
    """
    if not chat_history:
        return "", ""
    
    recent_turns = []
    for msg in chat_history[-max_turns:]:
        role = "User" if msg.get("role") in ["user", "human"] else "Assistant"
        content = msg.get("content") or msg.get("summary") or msg.get("query") or ""
        if content:
            recent_turns.append(f"{role}: {content}")
    
    history_text = ""
    last_query_augment = ""
    if recent_turns:
        history_text = "Prior Conversation History:\n" + "\n".join(recent_turns) + "\n\n"
        last_queries = [
            m.get("query") or m.get("content") 
            for m in chat_history 
            if m.get("role") in ["user", "human"] or "query" in m
        ]
        if last_queries:
            last_query_augment = str(last_queries[-1])
            
    return history_text, last_query_augment


def build_rag_prompt(
    query: str, 
    context: str, 
    history_text: str = "", 
    system_instruction: str = None
) -> str:
    """
    Constructs a formatted RAG prompt combining system instructions, conversation history,
    retrieved document context chunks, and the user query.
    """
    instruction = system_instruction or RAG_SYSTEM_INSTRUCTION
    return RAG_RESPONSE_PROMPT_TEMPLATE.format(
        system_instruction=instruction,
        history_text=history_text,
        context=context,
        query=query
    )
