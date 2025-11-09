from __future__ import annotations
import os, time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from aadhaar_mask import mask_aadhaar
import history_db
import json
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# -----------------------------
# Setup Semantic Kernel
# -----------------------------
kernel = sk.Kernel()
load_dotenv()
# Add Azure OpenAI Chat Service
chat_service = AzureChatCompletion(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
)
kernel.add_service(chat_service)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_K = 3
MAX_HISTORY_TO_SUMMARIZE = 20

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    k: int = DEFAULT_K


class ChatResponse(BaseModel):
    reply: str
    citations: List[Dict[str, str]]
    retrieval: Dict[str, Any]
    session_id: str
    used_clarification: bool = False
    confidence_score: Optional[float] = None


def call_ask(query: str, k: int) -> Dict[str, Any]:
    """Call the /ask endpoint logic directly instead of HTTP request"""
    try:
        # Import inside the function to avoid circular import
        from app import QueryRequest, retrieve, compose_answer
        req = QueryRequest(query=query, k=k)
        
        # Call retrieve and compose directly
        start = time.time()
        chunks, grounding_score = retrieve(req.query, req.k)
        answer = " ".join([c["full_snippet"] for c in chunks])
        latency = int((time.time() - start) * 1000)
        
        citations = [
            {"doc": c["doc"], "snippet": c["citation_preview"]} 
            for c in chunks
        ]
        
        return {
            "answer": answer,
            "citations": citations,
            "retrieval": {
                "k": req.k, 
                "latency_ms": latency,
                "grounding_score": round(grounding_score, 4)
            }
        }
    except Exception as e:
        print(f"Error in call_ask: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"/ask error: {str(e)}")


async def check_needs_clarification_with_llm(query: str, grounding: str, conversation_history: str) -> Dict[str, Any]:
    """
    Use LLM to determine if clarification is needed
    Returns: {"needs_clarification": bool, "clarification_question": str, "confidence": float}
    """
    prompt = """
You are an insurance claims assistant. Analyze the user's query and available information to determine if clarification is needed.

{{$conversation_history}}

Current User Query: {{$query}}

Available Information (Grounding):
{{$grounding}}

Task: Determine if you have enough information to answer the query confidently.

Analysis Criteria:
1. Is the query specific enough? (e.g., mentions policy number, specific dates, claim ID)
2. Is there relevant information in the grounding?
3. Are there multiple possible interpretations?
4. Does the conversation history provide additional context?

Response Format (JSON):
{
  "needs_clarification": true/false,
  "reason": "brief explanation",
  "clarification_question": "specific question to ask user (if needed)",
  "confidence": 0.0-1.0
}

If needs_clarification is false, set clarification_question to empty string.
Be specific in clarification questions - reference policy numbers, document names, or specific details.

Response:"""

    try:
        # Create semantic function for clarification check
        clarification_function = kernel.add_function(
            function_name="check_clarification",
            plugin_name="clarification_checker",
            prompt=prompt,
            description="Check if clarification is needed"
        )
        
        result = await kernel.invoke(
            clarification_function,
            query=query,
            conversation_history=conversation_history,
            grounding=grounding if grounding else "No relevant information found."
        )
        
        # Parse the LLM response
        response_text = str(result).strip()
        
        # Extract JSON from response (handles cases where LLM adds extra text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            response_text = response_text[json_start:json_end]
        
        clarification_data = json.loads(response_text)
        
        return {
            "needs_clarification": clarification_data.get("needs_clarification", False),
            "clarification_question": clarification_data.get("clarification_question", ""),
            "confidence": clarification_data.get("confidence", 0.5),
            "reason": clarification_data.get("reason", "")
        }
        
    except Exception as e:
        print(f"Error in clarification check: {e}")
        # Fallback to basic rule-based check
        if not grounding or len(grounding) < 50:
            return {
                "needs_clarification": True,
                "clarification_question": "I don't have enough information to answer your question. Could you provide more details or specify which policy (101 or 201) you're referring to?",
                "confidence": 0.3,
                "reason": "Insufficient grounding information"
            }
        return {
            "needs_clarification": False,
            "clarification_question": "",
            "confidence": 0.7,
            "reason": "Fallback - sufficient information available"
        }


async def generate_response_with_llm(query: str, grounding: str, conversation_history: str) -> str:
    """
    Use Semantic Kernel to generate a well-formatted response with conversation memory
    """
    prompt = """
You are a professional insurance claims assistant. Generate a helpful, accurate response.

{{$conversation_history}}

Current User Query: {{$query}}

Available Information (Grounding):
{{$grounding}}

Instructions for Response:
1. Use conversation history to maintain context and remember previous interactions
2. Answer based ONLY on the grounding information provided
3. Be professional, clear, and concise (2-4 sentences)
4. If answering about policies, mention the policy number
5. If answering about procedures, be specific with steps or timeframes
6. If information is incomplete in grounding, acknowledge what you know and what's missing
7. Use bullet points for multiple items
8. Include relevant numbers, dates, or amounts from the grounding

Format:
- Start with a direct answer
- Provide supporting details
- End with an offer to help further if appropriate

Response:"""

    try:
        # Create semantic function for response generation
        response_function = kernel.add_function(
            function_name="generate_response",
            plugin_name="response_generator",
            prompt=prompt,
            description="Generate formatted response"
        )
        
        result = await kernel.invoke(
            response_function,
            query=query,
            conversation_history=conversation_history,
            grounding=grounding
        )
        
        return str(result).strip()
        
    except Exception as e:
        print(f"Error in response generation: {e}")
        # Fallback to simple response
        return grounding[:500] if grounding else "I apologize, but I'm having trouble generating a response at the moment."


def build_conversation_context(session_id: str, max_turns: int = 5) -> str:
    """Build formatted conversation context from history"""
    history = history_db.get_recent(session_id, limit=max_turns)
    
    if not history:
        return "This is the start of the conversation.\n"
    
    context = "Previous conversation:\n"
    for turn in history[-max_turns:]:
        role = turn.get('role', 'unknown')
        message = turn.get('text', turn.get('message', ''))
        context += f"{role.capitalize()}: {message}\n"
    context += "\n"
    
    return context


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main conversational endpoint with full Semantic Kernel integration
    """
    session = req.session_id

    try:
        # Sanitize and validate input
        user_msg = mask_aadhaar(req.message.strip())
        
        if not user_msg:
            raise HTTPException(status_code=400, detail="Empty message")
        
        # Store user message in history
        history_db.add_turn(session, "user", user_msg)

        # Build conversation context
        conversation_context = build_conversation_context(session, max_turns=5)

        # Retrieve relevant information
        start = time.time()
        ask_resp = call_ask(user_msg, req.k)
        latency = int((time.time() - start) * 1000)
        
        answer = ask_resp.get("answer", "")
        citations = ask_resp.get("citations", [])

        # Use LLM to check if clarification is needed
        clarification_check = await check_needs_clarification_with_llm(
            user_msg, 
            answer, 
            conversation_context
        )

        if clarification_check["needs_clarification"]:
            # LLM determined clarification is needed
            reply = clarification_check["clarification_question"]
            retrieval = {"k": req.k, "latency_ms": latency}
            used_clar = True
            confidence = clarification_check["confidence"]
            
            print(f"Clarification needed: {clarification_check['reason']}")
        else:
            # Generate well-formatted response using LLM
            raw_reply = await generate_response_with_llm(
                user_msg, 
                answer, 
                conversation_context
            )
            reply = mask_aadhaar(raw_reply)
            retrieval = ask_resp.get("retrieval", {"k": req.k, "latency_ms": latency})
            used_clar = False
            confidence = clarification_check["confidence"]

        # Store assistant response in history
        history_db.add_turn(session, "assistant", reply)

        return ChatResponse(
            reply=reply,
            citations=citations,
            retrieval=retrieval,
            session_id=session,
            used_clarification=used_clar,
            confidence_score=confidence
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@router.get("/history/{session_id}")
def get_history(session_id: str):
    """Get conversation history for a session"""
    return {
        "session_id": session_id,
        "history": history_db.get_recent(session_id, limit=MAX_HISTORY_TO_SUMMARIZE)
    }


@router.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Clear conversation history for a session"""
    history_db.clear(session_id)
    return {
        "status": "success",
        "message": f"History cleared for session: {session_id}"
    }


@router.get("/sessions")
def list_sessions():
    """List all active sessions"""
    # Implement if history_db supports it
    return {"message": "Session listing not yet implemented"}