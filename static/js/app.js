/**
 * StockSignal AI - JavaScript Application
 * Stock Sentiment Analyzer
 */

// DOM Elements
const companyInput = document.getElementById('companyInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingState = document.getElementById('loadingState');
const loadingText = document.getElementById('loadingText');
const resultsSection = document.getElementById('resultsSection');
const quickButtons = document.querySelectorAll('.quick-btn');

// Loading messages for realistic feel
const loadingMessages = [
    'Connecting to Reddit API...',
    'Fetching posts from r/wallstreetbets...',
    'Fetching posts from r/stocks...',
    'Analyzing sentiment patterns...',
    'Processing natural language...',
    'Calculating sentiment scores...',
    'Generating investment insights...',
    'Preparing your report...'
];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Analyze button click
    analyzeBtn.addEventListener('click', handleAnalyze);
    
    // Enter key press
    companyInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleAnalyze();
    });
    
    // Quick select buttons
    quickButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            companyInput.value = btn.dataset.company;
            handleAnalyze();
        });
    });
    
    // Smooth scroll for nav links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});

// Handle analyze action
async function handleAnalyze() {
    const company = companyInput.value.trim();
    
    if (!company) {
        shakeInput();
        return;
    }
    
    // Show loading state
    showLoading();
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ company })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Connection failed. Please try again.');
    }
}

// Show loading state with animated messages
function showLoading() {
    resultsSection.classList.add('hidden');
    loadingState.classList.remove('hidden');
    analyzeBtn.disabled = true;
    
    let messageIndex = 0;
    const messageInterval = setInterval(() => {
        if (messageIndex < loadingMessages.length) {
            loadingText.textContent = loadingMessages[messageIndex];
            messageIndex++;
        }
    }, 400);
    
    // Store interval ID for cleanup
    loadingState.dataset.intervalId = messageInterval;
}

// Hide loading state
function hideLoading() {
    loadingState.classList.add('hidden');
    analyzeBtn.disabled = false;
    
    // Clear message interval
    const intervalId = loadingState.dataset.intervalId;
    if (intervalId) {
        clearInterval(parseInt(intervalId));
    }
}

// Display results
function displayResults(data) {
    hideLoading();
    resultsSection.classList.remove('hidden');
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
    
    // Update recommendation card
    document.getElementById('companyBadge').textContent = data.company;
    const recAction = document.getElementById('recAction');
    recAction.textContent = data.recommendation.action;
    recAction.style.color = data.recommendation.color;
    document.getElementById('recDescription').textContent = data.recommendation.description;
    
    // Confidence bar animation
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = '0%';
    setTimeout(() => {
        confidenceFill.style.width = data.recommendation.confidence + '%';
    }, 100);
    document.getElementById('confidenceText').textContent = `Confidence: ${data.recommendation.confidence}%`;
    
    // Update metrics with animation
    animateValue('positiveValue', 0, data.positive_pct, '%');
    animateValue('neutralValue', 0, data.neutral_pct, '%');
    animateValue('negativeValue', 0, data.negative_pct, '%');
    animateValue('avgSentiment', 0, data.avg_sentiment, '', 3);
    
    // Update bars
    setTimeout(() => {
        document.getElementById('positiveBar').style.width = data.positive_pct + '%';
        document.getElementById('neutralBar').style.width = data.neutral_pct + '%';
        document.getElementById('negativeBar').style.width = data.negative_pct + '%';
    }, 100);
    
    document.getElementById('posBarValue').textContent = data.positive_pct.toFixed(1) + '%';
    document.getElementById('neutBarValue').textContent = data.neutral_pct.toFixed(1) + '%';
    document.getElementById('negBarValue').textContent = data.negative_pct.toFixed(1) + '%';
    
    // Update posts table
    document.getElementById('postsCount').textContent = data.posts_analyzed + ' posts analyzed';
    const postsBody = document.getElementById('postsBody');
    postsBody.innerHTML = '';
    
    data.posts.forEach(post => {
        const row = document.createElement('tr');
        const linkHtml = post.url 
            ? `<a href="${post.url}" target="_blank" class="reddit-link">View on Reddit 🔗</a>`
            : '<span class="no-link">Sample Data</span>';
        row.innerHTML = `
            <td>${escapeHtml(post.title)}</td>
            <td>⬆️ ${post.score.toLocaleString()}</td>
            <td>💬 ${post.comments.toLocaleString()}</td>
            <td><span class="sentiment-badge ${post.label.toLowerCase()}">${post.label}</span></td>
            <td>${linkHtml}</td>
        `;
        postsBody.appendChild(row);
    });
    
    // Update data notice
    const noticeText = document.getElementById('noticeText');
    if (data.is_sample) {
        noticeText.textContent = 'Using sample data (Reddit API rate limited on cloud servers)';
    } else {
        noticeText.textContent = `Live data from Reddit • Updated: ${data.timestamp}`;
    }
}

// Animate numeric values
function animateValue(elementId, start, end, suffix = '', decimals = 1) {
    const element = document.getElementById(elementId);
    const duration = 1000;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeProgress = 1 - Math.pow(1 - progress, 3); // Ease out cubic
        
        const current = start + (end - start) * easeProgress;
        element.textContent = current.toFixed(decimals) + suffix;
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// Show error message
function showError(message) {
    hideLoading();
    
    // Create error toast
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.innerHTML = `
        <span>❌</span>
        <span>${message}</span>
    `;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #ff6b6b, #ff4444);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 500;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
        z-index: 1000;
        animation: slideUp 0.3s ease-out;
    `;
    
    document.body.appendChild(toast);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideDown 0.3s ease-out forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Shake input on empty submit
function shakeInput() {
    companyInput.style.animation = 'shake 0.5s ease-out';
    companyInput.style.borderColor = '#ff6b6b';
    
    setTimeout(() => {
        companyInput.style.animation = '';
        companyInput.style.borderColor = '';
    }, 500);
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add shake animation dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-10px); }
        40% { transform: translateX(10px); }
        60% { transform: translateX(-10px); }
        80% { transform: translateX(10px); }
    }
    
    @keyframes slideUp {
        from { transform: translate(-50%, 100px); opacity: 0; }
        to { transform: translate(-50%, 0); opacity: 1; }
    }
    
    @keyframes slideDown {
        from { transform: translate(-50%, 0); opacity: 1; }
        to { transform: translate(-50%, 100px); opacity: 0; }
    }
`;
document.head.appendChild(style);
