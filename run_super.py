import os
from dotenv import load_dotenv
from SupervisorAgent import app
from langchain_core.messages import HumanMessage

# --- 1. SETUP ---
# Load API keys from your .env file
load_dotenv()

# Check if necessary API keys are set before running
if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    raise ValueError(
        "Required API keys not found. "
        "Please create a .env file and set your GEMINI_API_KEY and TAVILY_API_KEY."
    )

# --- 2. DEFINE AND RUN QUERY ---

# The specific query you want to test.
query = "send an email to Shashank aksking if he gonna come to office today" 

# Package the input for the agent.
inputs = {"messages": [HumanMessage(content=query)]}

print(f"\n🤖 Supervisor is thinking about: '{query}'...")
try:
    # Stream the agent's response for the single query.
    final_answer = ""
    for event in app.stream(inputs):
        for key, value in event.items():
            # Print the output of each node as it runs for debugging
            print(f"--- Node '{key}' Output ---")
            print(value)
            print("--------------------")
            
            # Check if this is the final response from the supervisor
            if key == 'supervisor' and not value['messages'][-1].additional_kwargs:
                final_answer = value['messages'][-1].content
    
    # Print the final, consolidated answer if one was generated
    if final_answer:
            print("\n--- ✅ Final Response ---")
            print(final_answer)
            print("----------------------")

except Exception as e:
    print(f"\nAn error occurred during execution: {e}")
