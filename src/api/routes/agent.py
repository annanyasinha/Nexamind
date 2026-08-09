from fastapi import APIRouter, HTTPException
from api.schemas import AgentQueryRequest, AgentQueryResponse
from api.deps import get_nexamind_agent
from core.session_manager import session_manager
from utils.logger import logger

router = APIRouter(tags=["AI Agent"])


@router.post("/agent/query", response_model=AgentQueryResponse)
def execute_agent_query(req: AgentQueryRequest):
    """
    Executes the NexaMind AI Agent on the query.
    Dynamically selects Document RAG, YouTube RAG, and Web Search tools,
    executes selected tools, and synthesizes an LLM response.
    """
    try:
        agent = get_nexamind_agent()
        
        # Load session history if session_id is provided
        session_id = req.session_id
        history = req.chat_history or []
        if session_id:
            session = session_manager.get_session(session_id)
            if session:
                history = session.get_history()

        res = agent.run(
            query=req.query,
            chat_history=history,
            enabled_tools=req.enabled_tools
        )

        # Update session memory if session_id is present
        if session_id:
            session_manager.add_message(session_id, role="user", content=req.query)
            session_manager.add_message(
                session_id, 
                role="assistant", 
                content=res["answer"],
                sources=res.get("sources", [])
            )

        return AgentQueryResponse(
            query=res["query"],
            answer=res["answer"],
            session_id=session_id,
            steps=res["steps"],
            sources=res["sources"],
            execution_time_ms=res["execution_time_ms"]
        )

    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")
