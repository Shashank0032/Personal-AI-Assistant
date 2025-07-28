// DOM element references
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const chatWindow = document.getElementById('chat-window');

// In-memory store for the conversation history
const chatHistory = [{
    type: 'ai',
    content: "Hello! I'm your personal assistant. How can I help you orchestrate a task today?"
}];

/**
 * Appends a message to the chat window.
 * @param {string} type - 'human' or 'ai'.
 * @param {string} content - The HTML content of the message.
 * @returns {HTMLElement} The message element that was just added.
 */
function appendMessage(type, content) {
    const messageWrapper = document.createElement('div');
    messageWrapper.classList.add('message', `${type}-message`, 'flex', 'gap-3');

    let messageHTML = '';
    if (type === 'ai') {
        messageHTML = `
            <div class="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center text-white font-bold">A</div>
            <div class="bg-gray-200 text-gray-800 p-3 rounded-lg max-w-lg">
                ${content}
            </div>
        `;
    } else { // human
        messageWrapper.classList.add('justify-end');
        messageHTML = `
             <div class="bg-indigo-600 text-white p-3 rounded-lg max-w-lg">
                ${content}
            </div>
            <div class="flex-shrink-0 w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-white font-bold">U</div>
        `;
    }

    messageWrapper.innerHTML = messageHTML;
    chatWindow.appendChild(messageWrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Auto-scroll to the bottom
    return messageWrapper;
}

// Handle form submission
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userInput = chatInput.value.trim();
    if (!userInput) return;

    // Disable form and clear input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Display user's message and add to history
    appendMessage('human', `<p>${userInput}</p>`);
    chatHistory.push({ type: 'human', content: userInput });

    // Add a "thinking" indicator for the AI response
    const thinkingMessage = appendMessage('ai', '<p class="thinking">Thinking</p>');
    const aiContentContainer = thinkingMessage.querySelector('.bg-gray-200 > p');


    try {
        // Send the request to the backend
        const response = await fetch('/invoke', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Send history *before* the current user message for context
            body: JSON.stringify({ query: userInput, history: chatHistory.slice(0, -1) })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        // Process the streaming response from the server
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalContent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep any incomplete line for the next chunk

            for (const line of lines) {
                if (line.startsWith('data:')) {
                    try {
                        const data = JSON.parse(line.substring(5));
                        if (data.error) {
                            finalContent = `<p class="text-red-600"><strong>Error:</strong> ${data.error}</p>`;
                        } else {
                            finalContent = `<p>${data.content}</p>`;
                        }
                        // Update the thinking indicator with the final content
                        aiContentContainer.innerHTML = finalContent;
                    } catch (jsonError) {
                        console.error("Failed to parse JSON from stream:", line);
                        aiContentContainer.innerHTML = `<p class="text-red-600"><strong>Error:</strong> Received invalid data from server.</p>`;
                    }
                }
            }
        }
        
        // Add the final AI response to history
        // We need to parse the finalContent to get the raw text for the history object
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = finalContent;
        chatHistory.push({ type: 'ai', content: tempDiv.textContent || "" });


    } catch (error) {
        console.error('Fetch error:', error);
        const errorMessage = 'Sorry, something went wrong while connecting to the agent.';
        aiContentContainer.innerHTML = `<p class="text-red-600"><strong>Error:</strong> ${errorMessage}</p>`;
        chatHistory.push({ type: 'ai', content: errorMessage });
    } finally {
        // Re-enable the form
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
});

