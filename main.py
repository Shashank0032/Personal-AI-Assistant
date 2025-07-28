import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# --- 1. SETUP ---
# Load environment variables from .env file
load_dotenv()

# Ensure required API keys are set
if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    raise ValueError(
        "Required API keys not found. "
        "Please create a .env file and set your GEMINI_API_KEY and TAVILY_API_KEY."
    )

# Import the LangGraph application from your existing SupervisorAgent.py file
try:
    from SupervisorAgent import app as supervisor_agent_app
except ImportError:
    raise ImportError(
        "Could not import 'app' from SupervisorAgent.py. "
        "Please ensure the file exists and is in the same directory."
    )


# Initialize FastAPI application
app = FastAPI(
    title="Personal AI Assistant API",
    description="An API to interact with a multi-agent supervisor built with LangGraph.",
    version="1.0.0",
)

# --- 2. Mount Static Files ---
# Create a 'static' directory and place your style.css and script.js inside it.
# This line tells FastAPI to serve files from the 'static' directory at the '/static' URL path.
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- 3. Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    query: str
    history: list[dict]


# --- 4. FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_frontend(request: Request):
    """Serves the main chat interface."""
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1>"
                    "<p>Please make sure the index.html file is in the same directory as main.py</p>",
            status_code=404
        )


@app.post("/invoke")
async def invoke_agent(chat_request: ChatRequest):
    """
    Endpoint to process a query with the SupervisorAgent.
    It takes a query and chat history, and streams the agent's final response.
    """
    query = chat_request.query
    history = chat_request.history

    # Reconstruct the message history for the LangGraph agent
    messages: list[BaseMessage] = []
    for msg in history:
        if msg.get("type") == "human":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("type") == "ai":
            messages.append(AIMessage(content=msg.get("content", "")))

    messages.append(HumanMessage(content=query))
    inputs = {"messages": messages}

    async def event_stream():
        """
        Streams the final answer from the LangGraph agent execution.
        """
        final_answer = ""
        try:
            async for event in supervisor_agent_app.astream(inputs):
                for key, value in event.items():
                    if key == "supervisor" and not value["messages"][-1].additional_kwargs:
                        final_answer = value["messages"][-1].content
                        yield f"data: {json.dumps({'content': final_answer})}\n\n"
                        return

        except Exception as e:
            print(f"Error during agent invocation: {e}")
            error_message = f"An error occurred while processing your request: {e}"
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


