from core.prompt import (
    build_rag_prompt, 
    format_chat_history, 
    RAG_SYSTEM_INSTRUCTION, 
    NO_CONTEXT_FOUND_MESSAGE
)


def test_build_rag_prompt():
    query = "What is RAG?"
    context = "RAG stands for Retrieval-Augmented Generation."
    history = "User: Hi\nAssistant: Hello!"
    
    prompt = build_rag_prompt(query=query, context=context, history_text=history)
    assert RAG_SYSTEM_INSTRUCTION in prompt
    assert query in prompt
    assert context in prompt
    assert history in prompt


def test_format_chat_history():
    chat_history = [
        {"role": "user", "query": "Where did Shubham study?"},
        {"role": "assistant", "summary": "He studied Computer Science."}
    ]
    history_text, last_q = format_chat_history(chat_history)
    assert "User: Where did Shubham study?" in history_text
    assert "Assistant: He studied Computer Science." in history_text
    assert last_q == "Where did Shubham study?"
