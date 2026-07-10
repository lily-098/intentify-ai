document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/api/logs');
        const logs = await response.json();

        const intentCounts = {};
        const confRanges = {
            '< 60% (Human)': 0,
            '60-80%': 0,
            '> 80%': 0
        };

        logs.forEach(log => {
            const intent = log.predicted_intent;
            intentCounts[intent] = (intentCounts[intent] || 0) + 1;

            if (log.confidence < 0.6) confRanges['< 60% (Human)']++;
            else if (log.confidence < 0.8) confRanges['60-80%']++;
            else confRanges['> 80%']++;
        });

        new Chart(document.getElementById('intentChart'), {
            type: 'doughnut',
            data: {
                labels: Object.keys(intentCounts),
                datasets: [{
                    data: Object.values(intentCounts),
                    backgroundColor: ['#FF453A', '#32D74B', '#FF9F0A', '#BF5AF2', '#5E5CE6', '#EF4444']
                }]
            },
            options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: 'white' } } } }
        });

        new Chart(document.getElementById('confChart'), {
            type: 'bar',
            data: {
                labels: Object.keys(confRanges),
                datasets: [{
                    label: 'Number of Queries',
                    data: Object.values(confRanges),
                    backgroundColor: '#8B5CF6'
                }]
            },
            options: { 
                responsive: true, 
                scales: {
                    y: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    x: { ticks: { color: 'white' }, grid: { display: false } }
                },
                plugins: { legend: { display: false } } 
            }
        });

    } catch (error) {
        console.error("Error loading dashboard data", error);
    }
});
