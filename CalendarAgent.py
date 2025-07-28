import os
from datetime import datetime, timedelta
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

# --- 1. CONFIGURATION AND AUTHENTICATION ---

# This scope allows for reading, creating, updating, and deleting events.
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_FILE = "token_calendar.json"
CREDENTIALS_FILE = "client_secret.json"
LOCAL_TIMEZONE = "Asia/Kolkata" 

def get_calendar_service():
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
    return build("calendar", "v3", credentials=creds)

# --- 2. DEFINE CALENDAR TOOLS ---

@tool
def get_calendar_events(day: Optional[str] = "today"):
    """
    Gets a list of events from the calendar for a specific day.
    Args:
        day (str, optional): The day to get events for. Can be "today", "tomorrow", or a date in "YYYY-MM-DD" format. Defaults to "today".
    """
    service = get_calendar_service()
    try:
        now_utc = datetime.utcnow()
        if day is None or day.lower() == 'today':
            target_date = now_utc.date()
        elif day.lower() == 'tomorrow':
            target_date = now_utc.date() + timedelta(days=1)
        else:
            target_date = datetime.strptime(day, "%Y-%m-%d").date()

        start_of_day_utc = datetime.combine(target_date, datetime.min.time())
        end_of_day_utc = datetime.combine(target_date, datetime.max.time())
        
        start_time_iso = start_of_day_utc.isoformat() + "Z"
        end_time_iso = end_of_day_utc.isoformat() + "Z"

        events_result = service.events().list(
            calendarId='primary', timeMin=start_time_iso, timeMax=end_time_iso,
            maxResults=20, singleEvents=True, orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])

        if not events:
            return f"No upcoming events found for {day}."

        return [{"id": event['id'], "summary": event['summary'], "start": event['start'].get('dateTime', event['start'].get('date'))} for event in events]
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def create_calendar_event(summary: str, start_time: str, end_time: str, description: Optional[str] = None, attendees: Optional[list[str]] = None):
    """Creates a new event on the primary Google Calendar."""
    service = get_calendar_service()
    event = {
        'summary': summary,
        'description': description,
        'start': {'dateTime': start_time, 'timeZone': LOCAL_TIMEZONE},
        'end': {'dateTime': end_time, 'timeZone': LOCAL_TIMEZONE},
        'attendees': [{'email': email} for email in attendees] if attendees else [],
    }
    try:
        created_event = service.events().insert(calendarId='primary', body=event, sendUpdates="all").execute()
        return f"Event created successfully. Summary: '{summary}'. Link: {created_event.get('htmlLink')}"
    except HttpError as error:
        return f"An error occurred: {error}"

@tool
def update_calendar_event(event_id: str, summary: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    """Updates an existing calendar event by its ID."""
    service = get_calendar_service()
    try:
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        if summary: event['summary'] = summary
        if start_time: event['start']['dateTime'] = start_time
        if end_time: event['end']['dateTime'] = end_time
        
        updated_event = service.events().update(calendarId='primary', eventId=event['id'], body=event).execute()
        return f"Event updated successfully. Link: {updated_event.get('htmlLink')}"
    except HttpError as error:
        return f"An error occurred: {error}"

@tool
def delete_calendar_event(event_id: str):
    """Deletes a calendar event by its ID."""
    service = get_calendar_service()
    try:
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        return f"Event with ID {event_id} deleted successfully."
    except HttpError as error:
        return f"An error occurred: {error}"

# --- 3. SETUP THE AGENT WORKFLOW ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

def create_calendar_agent_app():
    tools = [get_calendar_events, create_calendar_event, update_calendar_event, delete_calendar_event]
    tool_node = ToolNode(tools)
    
    SYSTEM_PROMPT = f"""
# 📅 Overview
You are a **calendar assistant**. Your responsibilities include creating, retrieving, updating, and deleting events in the user's calendar.

All responses must be clear, concise, and based on the user's instructions.

---

## 🛠️ Calendar Management Tools

### ✅ Create Event
Use this tool to create calendar events — solo or with participants.

**Assumptions:**
- If the user doesn’t specify duration, default to **1 hour**.
- If time isn’t mentioned, ask the user for clarification before creating.

**Examples:**
- User: “Schedule a meeting with Priya tomorrow at 3 PM.”
  → Action: `Create Event` with:
    - Title: “Meeting with Priya”
    - Date: Tomorrow
    - Time: 3 PM
    - Duration: 1 hour
    - Participants: Priya

- User: “Block some time for focused work on Thursday afternoon.”
  → Action: `Create Event` with:
    - Title: “Focus time”
    - Date: Upcoming Thursday
    - Time: 2 PM (default)
    - Duration: 1 hour
    - Participants: None

---

### 📂 Get Events
Use this to retrieve the user’s upcoming events, daily schedule, or specific event details.

**Examples:**
- User: “What’s on my calendar today?”
  → Action: `Get Events` with today’s date.

- User: “Do I have any meetings next week?”
  → Action: `Get Events` from next Monday to Sunday.

- User: “List all events with Ankit this month.”
  → Action: `Get Events` with filter by participant “Ankit”.

---

### 🗑️ Delete Event
Use this to delete a calendar event.

**Important:**
You must first use `Get Events` to find the **event ID**.

**Examples:**
- User: “Cancel my 1-on-1 with Shashank on Friday.”
  → Step 1: `Get Events` to find matching event.
  → Step 2: `Delete Event` using the event ID.

- User: “Delete the focus session I added for Wednesday.”
  → Step 1: Get the event list for Wednesday.
  → Step 2: Identify the focus session.
  → Step 3: `Delete Event` with event ID.

---

### ✏️ Update Event
Use this to modify an existing event’s details — time, date, duration, title, or participants.

**Important:** Use `Get Events` first to retrieve the event and its ID.

**Examples:**
- User: “Reschedule my call with Alex from 4 PM to 5 PM.”
  → Step 1: `Get Events` to find the call with Alex.
  → Step 2: `Update Event` with new time = 5 PM.

- User: “Change tomorrow’s design review to 90 minutes.”
  → Step 1: Get tomorrow’s events.
  → Step 2: Identify “Design review”.
  → Step 3: Update duration to 90 minutes.

---

## 🕒 Final Notes
- Assume event duration is **1 hour** if not specified.
- If a participant is mentioned, include them in the invite.
- Be proactive: if details are missing (date/time), ask for clarification before proceeding.
- Respect the current date/time: **{datetime.now().strftime('%Y-%m-%d')}**

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
def run_calendar_agent(query: str) -> str:
    """Use this tool to manage calendar events (get, create, update, delete)."""
    app = create_calendar_agent_app()
    result = app.invoke({"messages": [HumanMessage(content=query)]})
    return result['messages'][-1].content
