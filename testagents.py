import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Import the runnable functions from your agent files
from EmailAgent import run_email_agent
from ContactAgent import run_contact_agent
from CalendarAgent import run_calendar_agent
from ContentCreatorAgent import run_content_creator_agent
from SearchAgent import run_search_agent

def test_all_agents():
    """
    A simple script to test each agent's functionality one by one.
    """
    print("--- 📧 Testing Email Agent ---")
    # Note: This will actually send an email. Use a test address.
    email_query = "Send an email to test@example.com with the subject 'Agent Test' and body 'This is a test of the email agent.'"
    email_result = run_email_agent.invoke({"query": email_query})
    print(f"Email Agent Result: {email_result}\n")

    print("--- 👤 Testing Contact Agent ---")
    contact_query = "Find the contact information for 'Test User'"
    contact_result = run_contact_agent.invoke({"query": contact_query})
    print(f"Contact Agent Result: {contact_result}\n")

    print("--- 🗓️ Testing Calendar Agent ---")
    calendar_query = "What are my events for the next 2 days?"
    calendar_result = run_calendar_agent.invoke({"query": calendar_query})
    print(f"Calendar Agent Result: {calendar_result}\n")

    print("--- ✍️ Testing Content Creator Agent ---")
    content_query = "Write a short blog post about the future of AI assistants."
    content_result = run_content_creator_agent.invoke({"query": content_query})
    print(f"Content Creator Result (HTML):\n{content_result}\n")

    print("--- 🔎 Testing Search Agent ---")
    search_query = "What is LangGraph?"
    search_result = run_search_agent.invoke({"query": search_query})
    print(f"Search Agent Result:\n{search_result}\n")


if __name__ == "__main__":
    # Before running, ensure you have:
    # 1. A .env file with your GEMINI_API_KEY and TAVILY_API_KEY
    # 2. A client_secret.json file for Google OAuth
    # 3. Run the script once to generate token files for each service (gmail, people, calendar)
    
    # Check for necessary API keys
    if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        print("❌ ERROR: Make sure GEMINI_API_KEY and TAVILY_API_KEY are set in your .env file.")
    else:
        test_all_agents()
