import os
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch

if not os.environ.get("TAVILY_API_KEY"):
    raise ValueError("Please set the TAVILY_API_KEY environment variable.")

web_search_tool = TavilySearch(max_results=3, name="tavily_search")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

def create_search_agent_app():
    tools = [web_search_tool]
    tool_node = ToolNode(tools)
    SYSTEM_PROMPT = """🔍 You are a **powerful Research AI** designed to answer user questions with accuracy, clarity, and reliable sources.

---

## ✅ Behavior Guidelines

### 1. SEARCH FIRST
- Your **first step must always** be to use the `tavily_search` tool to gather relevant, up-to-date information.

### 2. SYNTHESIZE AND ANSWER
- Use the search results to write a **concise, informative, and well-structured answer**.
- Combine facts from multiple sources where appropriate.
- Do **not fabricate** any information. Only use what’s in the search results.

### 3. CITE SOURCES
- Always end your response with a `Sources:` section.
- Include only the **URLs used** in the final answer.
- Format:
"""
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        google_api_key=os.environ["GEMINI_API_KEY"],
    ).bind(functions=[convert_to_openai_function(t) for t in tools])

    def call_model(state):
        response = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    def should_continue(state):
        return "function_call" in state["messages"][-1].additional_kwargs

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {True: "action", False: END})
    workflow.add_edge("action", "agent")
    return workflow.compile()

@tool
def run_search_agent(query: str) -> str:
    """Use this tool to search the web for information."""
    app = create_search_agent_app()
    final_answer = ""
    for event in app.stream({"messages": [HumanMessage(content=query)]}):
        for value in event.values():
            if isinstance(value['messages'][-1], HumanMessage):
                continue
            if not value['messages'][-1].additional_kwargs:
                final_answer = value['messages'][-1].content
    return final_answer
