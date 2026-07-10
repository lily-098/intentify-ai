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
                    'Content-Type': 'application/json',
                    'X-API-Key': 'intentify-secret-key'
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
                    ${data.log_id ? `
                    <div class="feedback-actions" data-logid="${data.log_id}">
                        <button class="thumb-btn up">👍</button>
                        <button class="thumb-btn down">👎</button>
                    </div>` : ''}
                </div>
            </div>
        `;
        
        chatHistory.appendChild(msgDiv);
        scrollToBottom();

        if (data.log_id) {
            const upBtn = msgDiv.querySelector('.thumb-btn.up');
            const downBtn = msgDiv.querySelector('.thumb-btn.down');
            
            upBtn.addEventListener('click', () => sendFeedback(data.log_id, 1, msgDiv));
            downBtn.addEventListener('click', () => sendFeedback(data.log_id, -1, msgDiv));
        }

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

    async function sendFeedback(logId, feedbackValue, msgDiv) {
        try {
            await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': 'intentify-secret-key'
                },
                body: JSON.stringify({ log_id: logId, feedback: feedbackValue })
            });
            const actions = msgDiv.querySelector('.feedback-actions');
            actions.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
        } catch(e) {
            console.error(e);
        }
    }

    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            const firstMsg = chatHistory.firstElementChild;
            chatHistory.innerHTML = '';
            if (firstMsg) {
                chatHistory.appendChild(firstMsg);
            }
        });
    }

    const toggleViewBtn = document.getElementById('toggleViewBtn');
    const chatView = document.getElementById('chatView');
    const dashboardView = document.getElementById('dashboardView');
    const headerTitle = document.getElementById('headerTitle');
    let intentChartInstance = null;
    let confChartInstance = null;

    if (toggleViewBtn) {
        toggleViewBtn.addEventListener('click', () => {
            if (chatView.style.display !== 'none') {
                // Switch to Dashboard
                chatView.style.display = 'none';
                dashboardView.style.display = 'flex';
                toggleViewBtn.innerHTML = '💬 Chat';
                headerTitle.innerHTML = 'Analytics Dashboard';
                if(clearBtn) clearBtn.style.display = 'none';
                loadDashboardData();
            } else {
                // Switch to Chat
                dashboardView.style.display = 'none';
                chatView.style.display = 'flex';
                toggleViewBtn.innerHTML = '📊 Dashboard';
                headerTitle.innerHTML = 'AI Model Active';
                if(clearBtn) clearBtn.style.display = 'inline-flex';
            }
        });
    }

    async function loadDashboardData() {
        try {
            const response = await fetch('/api/logs');
            const logs = await response.json();

            const intentCounts = {};
            const confRanges = { '< 60% (Human)': 0, '60-80%': 0, '> 80%': 0 };

            logs.forEach(log => {
                const intent = log.predicted_intent;
                intentCounts[intent] = (intentCounts[intent] || 0) + 1;
                if (log.confidence < 0.6) confRanges['< 60% (Human)']++;
                else if (log.confidence < 0.8) confRanges['60-80%']++;
                else confRanges['> 80%']++;
            });

            if (intentChartInstance) intentChartInstance.destroy();
            if (confChartInstance) confChartInstance.destroy();

            intentChartInstance = new Chart(document.getElementById('intentChart'), {
                type: 'doughnut',
                data: {
                    labels: Object.keys(intentCounts),
                    datasets: [{
                        data: Object.values(intentCounts),
                        backgroundColor: ['#FF453A', '#32D74B', '#FF9F0A', '#BF5AF2', '#5E5CE6', '#EF4444'],
                        borderWidth: 0
                    }]
                },
                options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#FAFAFA' } } }, cutout: '70%' }
            });

            confChartInstance = new Chart(document.getElementById('confChart'), {
                type: 'bar',
                data: {
                    labels: Object.keys(confRanges),
                    datasets: [{
                        label: 'Queries',
                        data: Object.values(confRanges),
                        backgroundColor: '#8B5CF6',
                        borderRadius: 6
                    }]
                },
                options: { 
                    responsive: true, 
                    scales: {
                        y: { ticks: { color: '#FAFAFA' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        x: { ticks: { color: '#FAFAFA' }, grid: { display: false } }
                    },
                    plugins: { legend: { display: false } } 
                }
            });
        } catch (error) {
            console.error("Error loading dashboard data", error);
        }
    }
});
