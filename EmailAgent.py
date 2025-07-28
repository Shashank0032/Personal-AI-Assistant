import os
import base64
from typing import TypedDict, Annotated, Sequence
from datetime import datetime, timedelta

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
from email.mime.text import MIMEText

# --- 1. CONFIGURATION AND AUTHENTICATION ---

# This scope allows for reading, composing, sending, and modifying labels.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
TOKEN_FILE = "token_gmail.json"
CREDENTIALS_FILE = "client_secret.json"

def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

# --- 2. DEFINE ALL EMAIL TOOLS ---

@tool
def send_email(to: str, subject: str, body: str):
    """Sends a new email."""
    service = get_gmail_service()
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    create_message = {'raw': encoded_message}
    sent_message = service.users().messages().send(userId="me", body=create_message).execute()
    return f"Email sent to {to}."

@tool
def reply_to_email(message_id: str, body: str):
    """Replies to a specific email identified by its message ID."""
    service = get_gmail_service()
    original_message = service.users().messages().get(userId='me', id=message_id).execute()
    thread_id = original_message['threadId']
    
    # Get headers from original message to properly form the reply
    headers = {h['name']: h['value'] for h in original_message['payload']['headers']}
    to_addr = headers.get('Reply-To') or headers.get('From')
    subject = headers.get('Subject', '')
    if not subject.lower().startswith('re:'):
        subject = f"Re: {subject}"

    message = MIMEText(body)
    message['to'] = to_addr
    message['subject'] = subject
    message['In-Reply-To'] = headers['Message-ID']
    message['References'] = headers.get('References', '') + f" {headers['Message-ID']}"

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    reply_body = {'raw': encoded_message, 'threadId': thread_id}
    sent_message = service.users().messages().send(userId='me', body=reply_body).execute()
    return f"Replied to message in thread {thread_id}."

@tool
def add_label_to_email(message_id: str, label_name: str):
    """Adds a label to a specific email."""
    service = get_gmail_service()
    # First, get the list of all labels to find the ID of the one we want.
    labels_response = service.users().labels().list(userId='me').execute()
    labels = labels_response.get('labels', [])
    label_id = next((l['id'] for l in labels if l['name'].lower() == label_name.lower()), None)
    
    if not label_id:
        return f"Error: Label '{label_name}' not found."
        
    modify_request = {'addLabelIds': [label_id], 'removeLabelIds': []}
    service.users().messages().modify(userId='me', id=message_id, body=modify_request).execute()
    return f"Label '{label_name}' added to message {message_id}."

@tool
def create_draft(to: str, subject: str, body: str):
    """Creates a draft email."""
    service = get_gmail_service()
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    draft_body = {'message': {'raw': encoded_message}}
    draft = service.users().drafts().create(userId='me', body=draft_body).execute()
    return f"Draft created for {to} with subject '{subject}'."

@tool
def get_emails(query: str):
    """Searches for emails using a query (e.g., 'from:jane@example.com is:unread')."""
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", q=query).execute()
    messages = results.get("messages", [])
    if not messages:
        return "No emails found for that query."
    
    emails = []
    for msg in messages[:5]: # Limit to 5 results for brevity
        msg_data = service.users().messages().get(userId="me", id=msg['id']).execute()
        headers = {h['name']: h['value'] for h in msg_data['payload']['headers']}
        emails.append({
            "id": msg['id'],
            "threadId": msg['threadId'],
            "subject": headers.get('Subject', ''),
            "from": headers.get('From', ''),
            "snippet": msg_data['snippet']
        })
    return emails

@tool
def mark_as_unread(message_id: str):
    """Marks a specific email as unread by its ID."""
    service = get_gmail_service()
    service.users().messages().modify(
        userId="me", id=message_id, body={'addLabelIds': ['UNREAD'], 'removeLabelIds': ['READ']}
    ).execute()
    return f"Message {message_id} marked as unread."

# --- 3. SETUP THE AGENT WORKFLOW ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

def create_email_agent_app():
    # Add all the new tools to this list
    tools = [send_email, reply_to_email, add_label_to_email, create_draft, get_emails, mark_as_unread]
    tool_node = ToolNode(tools)
    SYSTEM_PROMPT = """
You are an Email Assistant responsible for writing concise, professional plain-text emails.

## Guidelines:
- Write in clear, human-like language.
- Do not use any HTML or markup — just plain text.
- Include greetings and sign-offs when appropriate.
- Preserve tone and context from the user input.

---

Email Management Tools:

1. Send Email
   Use this tool to send new emails.
   Example:
   User: "Email the marketing team that the campaign starts Monday."
   Email Body:
   Hi team,
   Just a reminder that the campaign officially starts on Monday. Please ensure all assets are scheduled in advance.
   Shashank

2. Create Draft
   Use this when the user wants to write an email but not send it yet.
   Example:
   User: "Draft an apology email to the client for the delay."
   Draft Body:
   Hi [Client Name],
   I sincerely apologize for the delay in our response. We are actively working to resolve the issue and will follow up shortly.
   Shashank

3. Get Emails
   Use this to retrieve the user's recent or specific emails.
   Example:
   User: "Show me the latest email from Samantha."
   Action: Call Get Emails.

4. Get Labels
   Use this to retrieve all available labels.
   Example:
   User: "Label this as urgent."
   Action: Call Get Labels to find the label ID for 'Urgent'.

5. Label Email
   Use this to apply a label to an email.
   Requirements:
   - Call Get Emails to find the message ID.
   - Call Get Labels to find the label ID.
   Example:
   User: "Flag the last message from HR as 'Review Later'."
   Action:
     - Get Emails → Get message ID.
     - Get Labels → Get label ID for 'Review Later'.
     - Label Email → Use both IDs.

6. Email Reply
   Use this to reply to a specific email.
   Requirements:
   - Call Get Emails first to retrieve the message ID.
   Example:
   User: "Reply to Mike’s email and say I’ll call him tomorrow."
   Reply Body:
   Hi Mike,
   Thanks for the note. I’ll give you a call tomorrow to discuss further.
   Shashank

7. Mark Unread
   Use this to mark a message as unread.
   Requirements:
   - Call Get Emails first to get the message ID.
   Example:
   User: "Mark John’s last email as unread."
   Action:
     - Get Emails → Get message ID.
     - Mark Unread → Use message ID.

---

General Rules:
- All email messages must be written in a professional, friendly tone.
- Do not include <html> or <body> wrappers.

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
def run_email_agent(query: str) -> str:
    """Use this tool to manage emails (send, reply, get, label, draft, mark unread)."""
    app = create_email_agent_app()
    result = app.invoke({"messages": [HumanMessage(content=query)]})
    # Return the direct output of the last tool call or the final summary
    for key, value in result.items():
        if key == 'action':
            return str(value['messages'][-1].content)
    return str(result['messages'][-1].content)
