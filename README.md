# Personal AI Assistant API

This project implements a multi-agent personal AI assistant built using LangGraph and FastAPI. It provides a web-based interface for interacting with the AI assistant, which can leverage various specialized agents for tasks such as calendar management, contact handling, content creation, email operations, and web searching.

## Features

*   **Multi-Agent Architecture**: Utilizes LangGraph to orchestrate multiple AI agents for specialized tasks.
*   **FastAPI Backend**: Provides a robust and scalable API for handling requests and streaming responses.
*   **Web Interface**: Includes a simple `index.html` frontend for easy interaction.
*   **Modular Design**: Agents are separated into individual Python files for better organization and maintainability.

## Setup

### Prerequisites

Before you begin, ensure you have the following installed:

*   Python 3.9+
*   pip (Python package installer)

### Environment Variables

This project requires API keys for Google Gemini and TAVILY. Create a `.env` file in the root directory of the project and add the following:

```
GEMINI_API_KEY="your_gemini_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
```

Replace `"your_gemini_api_key_here"` and `"your_tavily_api_key_here"` with your actual API keys.

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To start the FastAPI application, navigate to the project's root directory and run:

```bash
uvicorn main:app --reload
```

This will start the server, typically accessible at `http://127.0.0.1:8000`.

## Project Structure

Here's a brief overview of the main files in this project:

*   `main.py`: The main FastAPI application entry point, handling web requests and integrating with the supervisor agent.
*   `SupervisorAgent.py`: Contains the core LangGraph application that orchestrates the various specialized agents.
*   `CalendarAgent.py`: Agent responsible for calendar-related operations.
*   `ContactAgent.py`: Agent responsible for managing contacts.
*   `ContentCreatorAgent.py`: Agent responsible for generating content.
*   `EmailAgent.py`: Agent responsible for email operations.
*   `SearchAgent.py`: Agent responsible for performing web searches.
*   `requirements.txt`: Lists all Python dependencies required for the project.
*   `index.html`: The frontend HTML file for the chat interface.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Usage

Once the application is running, open your web browser and navigate to `http://127.0.0.1:8000`. You can then interact with the AI assistant through the provided chat interface. The assistant will route your queries to the appropriate specialized agent based on the context of your request.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is open-source and available under the [MIT License](LICENSE).

