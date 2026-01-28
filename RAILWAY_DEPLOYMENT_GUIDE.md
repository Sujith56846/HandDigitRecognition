# 🚂 Complete Guide: Deploy Flask App to Railway

## 📋 Table of Contents
1. [Why Railway?](#why-railway)
2. [Prerequisites](#prerequisites)
3. [Project Setup](#project-setup)
4. [Deployment Methods](#deployment-methods)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Why Railway?

### ✅ Advantages
- **No size limits** - Perfect for PyTorch/ML apps
- **Free $5 credit/month** - Enough for small projects
- **Easy deployment** - Git push to deploy
- **Auto HTTPS** - Free SSL certificates
- **Simple pricing** - Pay for what you use
- **Great for ML** - Handles large dependencies

### 📊 Railway vs Others

| Feature | Railway | Vercel | Render | Heroku |
|---------|---------|--------|--------|--------|
| **PyTorch Support** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Free Tier** | $5 credit | Limited | 750hrs | Limited |
| **Deployment** | Git/CLI | Git/CLI | Git | Git |
| **Setup Time** | 5 min | 5 min | 10 min | 15 min |
| **Best For** | ML/Full apps | APIs | Full apps | Full apps |

---

## Prerequisites

### What You Need
- ✅ Railway account (free)
- ✅ GitHub account (recommended)
- ✅ Your Flask app files
- ✅ Git installed

### Create Railway Account
1. Go to https://railway.app/
2. Click "Login" → Sign in with GitHub
3. Authorize Railway
4. You get **$5 free credit** per month!

---

## Project Setup

### Required Files

Railway needs these files in your project root:

```
HandDigitRecognition/
├── app_flask.py              # Your Flask app
├── requirements.txt          # Python dependencies
├── railway.json             # Railway config (optional)
├── Procfile                 # How to run your app
├── runtime.txt              # Python version (optional)
├── templates/               # HTML templates
├── static/                  # CSS/JS files
└── *.pth                    # Model files
```

I'll create all necessary files for you.

---

## Configuration Files

### 1. Procfile

Create this file to tell Railway how to run your app:

```
web: python app_flask.py
```

**Or with Gunicorn (recommended for production):**
```
web: gunicorn app_flask:app --bind 0.0.0.0:$PORT
```

### 2. runtime.txt (Optional)

Specify Python version:
```
python-3.11.7
```

### 3. requirements.txt

Update your existing requirements.txt:
```txt
Flask==3.0.0
gunicorn==21.2.0
torch==2.2.0
torchvision==0.17.0
numpy==1.26.0
opencv-python-headless==4.9.0.80
Pillow==10.2.0
```

### 4. Update app_flask.py

Make sure your Flask app reads the PORT from environment:

```python
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

---

## Deployment Methods

### 🚀 METHOD 1: Deploy via GitHub (Recommended)

#### **Step 1: Push to GitHub**

```powershell
# Initialize git (if not already done)
cd D:\MajorProject\HandDigitRecognition
git init

# Create .gitignore
# (I'll provide this file)

# Add all files
git add .
git commit -m "Initial commit for Railway deployment"

# Create GitHub repo and push
# Go to https://github.com/new
# Create repo: hand-digit-recognition
# Then:
git remote add origin https://github.com/YOUR_USERNAME/hand-digit-recognition.git
git branch -M main
git push -u origin main
```

#### **Step 2: Deploy on Railway**

1. **Go to Railway Dashboard:**
   - Visit https://railway.app/dashboard
   - Click "New Project"

2. **Deploy from GitHub:**
   - Click "Deploy from GitHub repo"
   - If first time: Click "Configure GitHub App"
   - Authorize Railway to access your repos
   - Select your repository: `hand-digit-recognition`

3. **Configure Deployment:**
   - Railway auto-detects it's a Python app
   - Click "Deploy Now"
   - Railway automatically:
     - Installs dependencies
     - Runs your app
     - Assigns a URL

4. **Wait for Build:**
   - First build: 5-10 minutes (installing PyTorch)
   - Watch build logs in real-time
   - Status changes to "Active" when ready

5. **Get Your URL:**
   - Click "Settings" tab
   - Scroll to "Domains"
   - Click "Generate Domain"
   - Your app is live at: `https://your-app.up.railway.app`

#### **Step 3: Auto-Deploy**

From now on:
```powershell
# Make changes
git add .
git commit -m "Update app"
git push

# Railway automatically redeploys! 🎉
```

---

### 🖥️ METHOD 2: Deploy via Railway CLI

#### **Step 1: Install Railway CLI**

```powershell
# Using npm
npm install -g @railway/cli

# Or download from https://railway.app/cli
```

#### **Step 2: Login**

```powershell
railway login
```

Browser opens → Login with GitHub → Return to terminal

#### **Step 3: Initialize Project**

```powershell
cd D:\MajorProject\HandDigitRecognition

# Link to Railway
railway init

# Follow prompts:
# ? Enter project name: hand-digit-recognition
# ? Environment: production
```

#### **Step 4: Deploy**

```powershell
# Deploy your app
railway up

# Watch logs
railway logs

# Open in browser
railway open
```

#### **Step 5: Get URL**

```powershell
# Generate public URL
railway domain
```

---

## Step-by-Step Setup

### 📝 Complete Setup Process

#### **1. Update app_flask.py**

Ensure your app reads the PORT from Railway:

```python
if __name__ == '__main__':
    import os
    # Railway provides PORT via environment variable
    port = int(os.environ.get('PORT', 5000))
    
    # Load the model on startup
    if load_model():
        print(f"\n🚀 Hand Digit Recognition Web App")
        print(f"📊 Model Type: {model_type}")
        print(f"🌐 Starting Flask server on port {port}...")
        
        # Run the Flask app
        # Use 0.0.0.0 to accept external connections
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Failed to load model.")
```

#### **2. Create Procfile**

Create a file named `Procfile` (no extension) in project root:

```
web: gunicorn app_flask:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

**What this does:**
- `web:` - Tells Railway this is a web service
- `gunicorn` - Production WSGI server
- `app_flask:app` - Import app from app_flask.py
- `--bind 0.0.0.0:$PORT` - Listen on Railway's port
- `--workers 1` - One worker (saves memory)
- `--timeout 120` - 2 minute timeout for model loading

#### **3. Update requirements.txt**

Add Gunicorn:

```txt
Flask==3.0.0
gunicorn==21.2.0
torch==2.2.0
torchvision==0.17.0
numpy==1.26.0
opencv-python-headless==4.9.0.80
Pillow==10.2.0
```

#### **4. Create .gitignore**

```txt
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual Environment
venv/
ENV/
env/
myenv/

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Local data
data/
build/
dist/

# Railway
.railway/
```

#### **5. Optional: Create railway.json**

For advanced configuration:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "gunicorn app_flask:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

---

## Testing Before Deployment

### Test Locally with Gunicorn

```powershell
# Install gunicorn
pip install gunicorn

# Test locally
gunicorn app_flask:app --bind 0.0.0.0:5000 --workers 1

# Visit: http://localhost:5000
```

If this works, Railway will work!

---

## Monitoring & Management

### View Logs

**In Railway Dashboard:**
1. Click your project
2. Click "Deployments" tab
3. Click latest deployment
4. View real-time logs

**Using CLI:**
```powershell
railway logs
```

### Environment Variables

**Add secrets/config:**

**In Dashboard:**
1. Project → Variables tab
2. Click "New Variable"
3. Add KEY=VALUE
4. Redeploy

**Using CLI:**
```powershell
railway variables set SECRET_KEY=your_secret_value
```

**In code:**
```python
import os
secret = os.environ.get('SECRET_KEY')
```

### Restart Service

**In Dashboard:**
- Click "Deployments"
- Click "⋯" → "Restart"

**Using CLI:**
```powershell
railway restart
```

### View Metrics

**In Dashboard:**
- Click "Metrics" tab
- See CPU, Memory, Network usage
- Monitor costs

---

## Troubleshooting

### ❌ Issue 1: "Build Failed"

**Symptoms:**
```
Error: Failed to install requirements
```

**Solutions:**

1. **Check requirements.txt syntax:**
   - No typos in package names
   - Valid versions
   - One package per line

2. **Use compatible versions:**
   ```txt
   torch==2.2.0  # Not 2.0.0
   ```

3. **Check build logs:**
   - Look for specific error messages
   - Fix missing dependencies

### ❌ Issue 2: "Application Failed to Start"

**Symptoms:**
```
Error: Web process failed to bind to $PORT
```

**Solutions:**

1. **Check Procfile:**
   ```
   web: gunicorn app_flask:app --bind 0.0.0.0:$PORT
   ```

2. **Check app_flask.py:**
   ```python
   port = int(os.environ.get('PORT', 5000))
   app.run(host='0.0.0.0', port=port)
   ```

3. **Use health check endpoint:**
   ```python
   @app.route('/health')
   def health():
       return jsonify({'status': 'healthy'})
   ```

### ❌ Issue 3: "Model Not Found"

**Symptoms:**
```
FileNotFoundError: digit_recognition_cnn_model.pth
```

**Solutions:**

1. **Verify files are in repo:**
   ```powershell
   git ls-files | grep .pth
   ```

2. **Check .gitignore doesn't exclude .pth files:**
   - Remove `*.pth` from .gitignore if present

3. **Use absolute paths:**
   ```python
   import os
   BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   model_path = os.path.join(BASE_DIR, 'digit_recognition_cnn_model.pth')
   ```

### ❌ Issue 4: "Out of Memory"

**Symptoms:**
```
Error: Process killed (OOM)
```

**Solutions:**

1. **Use fewer workers:**
   ```
   web: gunicorn app_flask:app --workers 1
   ```

2. **Optimize model loading:**
   ```python
   # Load model once, not per request
   model = None
   def load_model():
       global model
       if model is None:
           # Load here
   ```

3. **Upgrade Railway plan:**
   - Free tier: 512MB-1GB RAM
   - Paid: More memory available

### ❌ Issue 5: "Deployment Timeout"

**Symptoms:**
```
Error: Build timeout after 10 minutes
```

**Solutions:**

1. **Increase timeout in Procfile:**
   ```
   web: gunicorn app_flask:app --timeout 300
   ```

2. **Optimize dependencies:**
   - Only include needed packages
   - Use pre-built wheels

### ❌ Issue 6: "Static Files Not Loading"

**Symptoms:**
- CSS/JS not loading
- 404 errors for static files

**Solutions:**

1. **Check Flask configuration:**
   ```python
   app = Flask(__name__)  # Auto-detects static folder
   ```

2. **Verify folder structure:**
   ```
   static/
     style.css
     script.js
   ```

3. **Use url_for in templates:**
   ```html
   <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
   ```

---

## Costs & Pricing

### Free Tier
- **$5 credit/month** (enough for small apps)
- **500 hours** of usage
- **Usage-based** - Pay for what you use

### Example Costs
- **Idle app:** ~$0-1/month
- **Light usage:** ~$2-3/month  
- **Medium usage:** ~$5-10/month

### Cost Optimization

1. **Use sleep mode:**
   - App sleeps when inactive
   - Wakes up on request

2. **Optimize resources:**
   - Reduce workers
   - Efficient code

3. **Monitor usage:**
   - Check Metrics tab
   - Set spending limits

---

## Production Best Practices

### 1. Use Gunicorn
```
web: gunicorn app_flask:app --workers 1 --timeout 120
```

### 2. Add Health Checks
```python
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': model_type})
```

### 3. Error Handling
```python
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
```

### 4. Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
app.logger.info('Model loaded successfully')
```

### 5. Environment Variables
```python
import os
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
app.run(debug=DEBUG)
```

---

## Custom Domain

### Add Your Domain

1. **In Railway Dashboard:**
   - Settings → Domains
   - Click "Custom Domain"
   - Enter your domain: `myapp.com`

2. **Update DNS:**
   - Add CNAME record
   - Point to Railway URL
   - Wait for DNS propagation

3. **SSL Certificate:**
   - Railway auto-generates
   - HTTPS enabled automatically

---

## Commands Cheat Sheet

```powershell
# Login
railway login

# Initialize project
railway init

# Deploy
railway up

# View logs
railway logs

# Open in browser
railway open

# Set environment variable
railway variables set KEY=VALUE

# Check status
railway status

# Link to existing project
railway link

# Disconnect
railway unlink

# Restart
railway restart

# Delete project
railway delete
```

---

## Comparison: Railway vs Render

| Feature | Railway | Render |
|---------|---------|--------|
| **Free Tier** | $5/month credit | 750 hours |
| **Setup** | Easier | Slightly complex |
| **Speed** | Fast | Medium |
| **Dashboard** | Modern | Traditional |
| **CLI** | Excellent | Good |
| **Best For** | Quick deploys | Longer projects |

**For this project:** Both work great! Railway is slightly easier.

---

## Complete Deployment Checklist

- [ ] Create Railway account
- [ ] Update `app_flask.py` with PORT handling
- [ ] Create `Procfile`
- [ ] Update `requirements.txt` (add gunicorn)
- [ ] Create `.gitignore`
- [ ] Test locally with gunicorn
- [ ] Push to GitHub
- [ ] Connect GitHub to Railway
- [ ] Deploy and wait for build
- [ ] Generate domain
- [ ] Test deployed app
- [ ] Monitor logs and metrics

---

## Next Steps After Deployment

### 1. Test Your App
- Visit your Railway URL
- Test digit recognition
- Check all features work

### 2. Monitor
- Watch logs for errors
- Check metrics
- Monitor costs

### 3. Share
- Add to portfolio
- Share on social media
- Add to resume

### 4. Iterate
- Fix bugs
- Add features
- Push updates (auto-deploys!)

---

## Resources

- 📚 **Railway Docs:** https://docs.railway.app/
- 💬 **Railway Discord:** https://discord.gg/railway
- 🐦 **Railway Twitter:** https://twitter.com/Railway
- 📖 **Gunicorn Docs:** https://docs.gunicorn.org/

---

## 🎉 You're Ready!

Railway is perfect for your PyTorch Flask app. No size limits, easy deployment, and generous free tier.

**Start deploying now! 🚂**
