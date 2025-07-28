import os
from typing import TypedDict, Annotated, Sequence, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Scope allows for reading and writing contacts
SCOPES = ["https://www.googleapis.com/auth/contacts"]
TOKEN_FILE = "token_people.json"
CREDENTIALS_FILE = "client_secret.json"

def get_people_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token_file:
            token_file.write(creds.to_json())
    return build("people", "v1", credentials=creds)

@tool
def get_contacts(query: str) -> list:
    """Searches for contacts by name to find their email or phone number."""
    service = get_people_service()
    try:
        connections = service.people().connections().list(
            resourceName="people/me", pageSize=1000, personFields="names,phoneNumbers,emailAddresses"
        ).execute()
        
        all_contacts = connections.get("connections", [])
        found_contacts = []
        lower_query = query.lower()
        
        for person in all_contacts:
            match = False
            if any(lower_query in n.get("displayName", "").lower() for n in person.get("names", [])):
                match = True
            
            if match:
                name = person.get("names", [{}])[0].get("displayName", "N/A")
                phones = [p.get("value") for p in person.get("phoneNumbers", [])]
                emails = [e.get("value") for e in person.get("emailAddresses", [])]
                found_contacts.append({"name": name, "phones": phones, "emails": emails})

        return found_contacts if found_contacts else "No contacts found matching that query."
    except HttpError as e:
        return f"An error occurred: {e}"

@tool
def add_or_update_contact(name: str, phone: Optional[str] = None, email: Optional[str] = None):
    """Creates a new contact or updates an existing one with a phone number or email."""
    service = get_people_service()
    try:
        # This is a simplified creation logic. A full implementation would search and update.
        new_contact = {
            "names": [{"givenName": name}],
            "phoneNumbers": [{"value": phone}] if phone else [],
            "emailAddresses": [{"value": email}] if email else [],
        }
        created_person = service.people().createContact(body=new_contact).execute()
        return f"Successfully created/updated contact: {created_person.get('names')[0].get('displayName')}"
    except HttpError as e:
        return f"An error occurred: {e}"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

def create_contact_agent_app():
    # Ensure both tools are available to the agent
    tools = [get_contacts, add_or_update_contact]
    tool_node = ToolNode(tools)
    
    # --- THIS IS THE ENHANCED PROMPT ---
    SYSTEM_PROMPT = """
    📇 You are an expert Contacts Assistant.

🎯 Your ONLY job is to use the provided tools to **get, add, or update contacts**, and then **summarize the result in a clean, single sentence**.

🚫 Do NOT add any extra commentary, context, or formatting — only provide the final summary after the tool runs.

---

## 🧾 `get_contacts` Examples:

### ✅ When contact is found:
- **Tool Output:** `[{'name': 'Shashank Heroor', 'phones': [], 'emails': ['shashankheroor2@gmail.com']}]`
- **Your Final Summary:** `Contact found for Shashank Heroor. Email: shashankheroor2@gmail.com`

- **Tool Output:** `[{'name': 'Ananya Singh', 'phones': ['+918888888888'], 'emails': []}]`
- **Summary:** `Contact found for Ananya Singh. Phone: +918888888888`

- **Tool Output:** `[{'name': 'Rahul Verma', 'phones': ['+911234567890'], 'emails': ['rahul@example.com']}]`
- **Summary:** `Contact found for Rahul Verma. Phone: +911234567890. Email: rahul@example.com`

### ❌ When contact is not found:
- **Tool Output:** `[]`
- **Summary:** `No contact found for that query.`

---

## ✏️ `add_or_update_contact` Examples:

### ✅ When creating a new contact:
- **Tool Output:** `'Successfully created/updated contact: Jane Doe'`
- **Your Final Summary:** `Successfully created contact for Jane Doe.`

### 🔁 When updating an existing contact:
- **Tool Output:** `'Successfully created/updated contact: Kiran Rao'`
- **Your Final Summary:** `Successfully updated contact for Kiran Rao.`

### ⚠️ If the contact name already exists but fields are new:
- **Tool Output:** `'Successfully created/updated contact: Dr. Meera Patel'`
- **Your Final Summary:** `Successfully updated contact for Dr. Meera Patel.`

---

✅ Keep your response short and exact.  
✅ Always mention the contact name in the final summary.  
✅ If phone and email are both available, include both.  
✅ If neither is available in `get_contacts`, still return: `Contact found for [name].`
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
def run_contact_agent(query: str) -> str:
    """Use this tool to manage contacts (get, add, update)."""
    app = create_contact_agent_app()
    result = app.invoke({"messages": [HumanMessage(content=query)]})
    return result['messages'][-1].content
