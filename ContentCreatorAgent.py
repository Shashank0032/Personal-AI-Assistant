import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch

# --- THIS IS THE FIX ---
# Load environment variables at the top of the file
load_dotenv()

# --- 1. SETUP THE TOOL ---
if not os.environ.get("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in .env file or environment variables.")

web_search_tool = TavilySearch(max_results=5, name="tavily_search")

# --- 2. SETUP THE AGENT WORKFLOW ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

def create_content_agent_app():
    tools = [web_search_tool]
    tool_node = ToolNode(tools)
    SYSTEM_PROMPT = """
# ✍️ Overview
You are a **skilled AI blog writer**. Your writing style is clear, compelling, and informative. Your role is to generate well-researched blog posts using reliable sources.

---

## 🛠️ Tools
- Use the `tavily_search` tool to gather up-to-date, relevant information for the blog post.
- You MUST include citations and links from Tavily in the final output when the search is used.

---

## 📄 Blog Content Requirements

1. **Formatting:**
   - All blog content must be in **valid HTML**.
   - Use proper structure:
     - `<h1>` for the main title
     - `<h2>` for section headings
     - `<p>` for all paragraphs
     - `<a href="URL">source</a>` for citations
   - Do **not** use Markdown or plain text.

2. **Tone:**
   - Write in a professional yet accessible tone.
   - Avoid overly technical language unless the audience is technical.
   - Keep sentences concise, and use active voice.

3. **Citations:**
   - Always include **at least one source link** from Tavily if the tool is used.
   - Embed the source using `<a href="URL">source</a>` within or after the relevant paragraph.

---

## ✅ Examples

### Example 1: User Prompt
> “Write a blog post on the rise of electric vehicles in India.”

### Expected Blog Structure (simplified):
```html
<h1>The Rise of Electric Vehicles in India</h1>

<h2>Government Push and Policy Support</h2>
<p>India's EV sector has grown rapidly due to government schemes such as FAME II and tax incentives. These efforts aim to reduce emissions and fossil fuel dependence.</p>

<h2>Market Growth and Consumer Adoption</h2>
<p>Companies like Tata Motors and Ola Electric are driving innovation, with EV sales seeing record growth in 2024. <a href="https://example.com/ev-india-report">source</a></p>

<h2>Challenges Ahead</h2>
<p>Despite progress, infrastructure and affordability remain key challenges. However, with continued investment, India's EV future looks promising.</p>
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
def run_content_creator_agent(query: str) -> str:
    """Use this tool to create blog posts or other long-form content."""
    app = create_content_agent_app()
    final_result_html = ""
    for event in app.stream({"messages": [HumanMessage(content=query)]}):
        for value in event.values():
            if isinstance(value['messages'][-1], HumanMessage):
                continue
            if not value['messages'][-1].additional_kwargs:
                final_result_html = value['messages'][-1].content
    return final_result_html
