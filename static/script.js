document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const queryInput = document.getElementById('queryInput');
    const chatHistory = document.getElementById('chatHistory');

    const intentColors = {
        'complaint': 'var(--intent-complaint)',
        'order_status': 'var(--intent-order)',
        'refund_request': 'var(--intent-refund)',
        'product_query': 'var(--intent-product)',
        'general': 'var(--intent-general)',
        'fallback_to_human': '#EF4444' // red for fallback
    };

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (!text) return;

        // Add user message
        addUserMessage(text);
        queryInput.value = '';

        // Add loading indicator
        const loadingId = addLoadingIndicator();

        try {
            // Send request to FastAPI backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            
            // Remove loading indicator
            document.getElementById(loadingId).remove();

            if (data.error) {
                addAiMessage("Sorry, the model failed to process this request. Make sure the backend loaded correctly.");
            } else {
                addAnalysisMessage(data);
            }

        } catch (error) {
            document.getElementById(loadingId).remove();
            addAiMessage("Error connecting to the backend server. Is the FastAPI app running?");
        }
    });

    function addUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-message';
        msgDiv.innerHTML = `
            <div class="avatar user-avatar">U</div>
            <div class="bubble">${escapeHtml(text)}</div>
        `;
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function addAiMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-message';
        msgDiv.innerHTML = `
            <div class="avatar ai-avatar">AI</div>
            <div class="bubble">${escapeHtml(text)}</div>
        `;
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function addAnalysisMessage(data) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-message';
        
        const intentId = data.predicted_intent;
        const color = intentColors[intentId] || 'var(--text-secondary)';
        const confPercent = (data.confidence * 100).toFixed(1);
        
        let responseText = "Here is my classification for your query:";
        if (intentId === 'fallback_to_human') {
            responseText = "I'm not confident enough to answer this. Routing to a human agent.";
        }

        msgDiv.innerHTML = `
            <div class="avatar ai-avatar">AI</div>
            <div class="bubble">
                <p>${responseText}</p>
                <div class="analysis-card">
                    <div class="analysis-row">
                        <span class="analysis-label">Predicted Intent</span>
                        <span class="intent-tag" style="background: ${color}">${data.readable_intent}</span>
                    </div>
                    <div class="analysis-row">
                        <span class="analysis-label">Confidence Score</span>
                        <span class="analysis-value">${confPercent}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width: 0%; background: ${color}"></div>
                    </div>
                </div>
            </div>
        `;
        
        chatHistory.appendChild(msgDiv);
        scrollToBottom();

        // Animate the confidence bar
        setTimeout(() => {
            const bar = msgDiv.querySelector('.conf-bar-fill');
            if(bar) bar.style.width = `${confPercent}%`;
        }, 50);
    }

    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-message';
        msgDiv.id = id;
        msgDiv.innerHTML = `
            <div class="avatar ai-avatar">AI</div>
            <div class="bubble">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
});
