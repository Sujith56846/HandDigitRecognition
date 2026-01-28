// ===== CANVAS DRAWING FUNCTIONALITY =====
class DigitDrawing {
    constructor() {
        this.canvas = document.getElementById('drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        
        this.setupCanvas();
        this.bindEvents();
        this.setupAutoPrediction();
    }
    
    setupCanvas() {
        // Set canvas properties for drawing
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.lineWidth = 12;
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.fillStyle = '#000000';
        
        // Fill canvas with black background
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    bindEvents() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));
        
        // Button events
        document.getElementById('clearBtn').addEventListener('click', this.clearCanvas.bind(this));
        document.getElementById('predictBtn').addEventListener('click', this.predictDigit.bind(this));
    }
    
    setupAutoPrediction() {
        // Auto-predict after user stops drawing for 1 second
        this.predictionTimeout = null;
        this.hasDrawn = false;
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    getTouchPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.touches[0].clientX - rect.left,
            y: e.touches[0].clientY - rect.top
        };
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        const pos = this.getMousePos(e);
        this.lastX = pos.x;
        this.lastY = pos.y;
        
        // Hide canvas overlay
        document.getElementById('canvasOverlay').classList.add('hidden');
        this.hasDrawn = true;
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        e.preventDefault();
        const pos = this.getMousePos(e);
        
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();
        
        this.lastX = pos.x;
        this.lastY = pos.y;
        
        // Reset auto-prediction timer
        this.resetAutoPrediction();
    }
    
    stopDrawing() {
        if (this.isDrawing && this.hasDrawn) {
            this.isDrawing = false;
            // Trigger auto-prediction after a short delay
            this.resetAutoPrediction();
        }
    }
    
    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                         e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        this.canvas.dispatchEvent(mouseEvent);
    }
    
    resetAutoPrediction() {
        // Clear existing timeout
        if (this.predictionTimeout) {
            clearTimeout(this.predictionTimeout);
        }
        
        // Set new timeout for auto-prediction
        this.predictionTimeout = setTimeout(() => {
            if (this.hasDrawn && !this.isDrawing) {
                this.predictDigit();
            }
        }, 1500); // 1.5 seconds delay
    }
    
    clearCanvas() {
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Show canvas overlay
        document.getElementById('canvasOverlay').classList.remove('hidden');
        
        // Clear prediction
        this.clearPrediction();
        
        // Reset state
        this.hasDrawn = false;
        if (this.predictionTimeout) {
            clearTimeout(this.predictionTimeout);
        }
        
        this.showToast('Canvas cleared!', 'success');
    }
    
    clearPrediction() {
        const predictionContent = document.getElementById('predictionContent');
        predictionContent.innerHTML = `
            <div class="prediction-placeholder">
                <i class="fas fa-draw-polygon"></i>
                <p>Draw a digit to see the AI prediction</p>
            </div>
        `;
    }
    
    async predictDigit() {
        if (!this.hasDrawn) {
            this.showToast('Please draw a digit first!', 'error');
            return;
        }
        
        try {
            // Show loading
            this.showLoading(true);
            
            // Get canvas data as base64 image
            const imageData = this.canvas.toDataURL('image/png');
            
            // Send to Flask backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            // Display prediction
            this.displayPrediction(result.prediction, result.confidence, result.model_type);
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showToast('Prediction failed: ' + error.message, 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    displayPrediction(digit, confidence, modelType) {
        const predictionContent = document.getElementById('predictionContent');
        
        predictionContent.innerHTML = `
            <div class="prediction-result">
                <div class="prediction-digit">${digit}</div>
                <div class="confidence-score">Confidence: ${confidence}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                </div>
                <small style="color: #666; margin-top: 10px; display: block;">
                    Predicted by ${modelType} model
                </small>
            </div>
        `;
        
        this.showToast(`Predicted digit: ${digit} (${confidence}% confidence)`, 'success');
    }
    
    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (show) {
            overlay.classList.add('show');
        } else {
            overlay.classList.remove('show');
        }
    }
    
    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        container.appendChild(toast);
        
        // Animate in
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => container.removeChild(toast), 300);
        }, 3000);
    }
}

// ===== UTILITY FUNCTIONS =====
function checkModelStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            if (!data.model_loaded) {
                document.body.innerHTML = `
                    <div style="display: flex; justify-content: center; align-items: center; min-height: 100vh; text-align: center; color: white;">
                        <div>
                            <h1><i class="fas fa-exclamation-triangle"></i> Model Not Found</h1>
                            <p>Please run 'train_model.py' first to train the AI models.</p>
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Health check failed:', error);
        });
}

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', (e) => {
    if (e.key === 'c' || e.key === 'C') {
        if (window.digitDrawing) {
            window.digitDrawing.clearCanvas();
        }
    } else if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        if (window.digitDrawing) {
            window.digitDrawing.predictDigit();
        }
    }
});

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the drawing functionality
    window.digitDrawing = new DigitDrawing();
    
    // Check model status
    checkModelStatus();
    
    // Add some interactive effects
    addInteractiveEffects();
    
    console.log('🎨 Hand Digit Recognition Web App Initialized!');
    console.log('📖 Instructions:');
    console.log('   - Draw digits with mouse/finger');
    console.log('   - Press C to clear canvas');
    console.log('   - Press Space/Enter to predict');
});

function addInteractiveEffects() {
    // Add hover effects to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', () => {
            btn.style.transform = 'translateY(-2px)';
        });
        btn.addEventListener('mouseleave', () => {
            btn.style.transform = 'translateY(0)';
        });
    });
    
    // Add click effect to canvas
    const canvas = document.getElementById('drawingCanvas');
    canvas.addEventListener('click', () => {
        canvas.style.transform = 'scale(1.02)';
        setTimeout(() => {
            canvas.style.transform = 'scale(1)';
        }, 150);
    });
}

// ===== PWA SUPPORT (OPTIONAL) =====
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/sw.js')
        .then(() => console.log('Service Worker registered'))
        .catch(err => console.log('Service Worker registration failed'));
}