# 🎨 Complete Guide: Deploy Flask App to Render

## 📋 Table of Contents
1. [Why Render?](#why-render)
2. [Prerequisites](#prerequisites)
3. [Project Setup](#project-setup)
4. [Deployment Process](#deployment-process)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Why Render?

### ✅ Advantages
- **750 free hours/month** - Generous free tier
- **No size limits** - Perfect for PyTorch apps
- **Auto-deploy from Git** - Push to deploy
- **Free SSL** - HTTPS included
- **Simple pricing** - Clear and predictable
- **Great uptime** - Reliable hosting

### 📊 Render vs Others

| Feature | Render | Railway | Vercel | Heroku |
|---------|--------|---------|--------|--------|
| **PyTorch Support** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Free Tier** | 750hrs | $5 credit | Limited | Limited |
| **Auto HTTPS** | ✅ Yes | ✅ Yes | ✅ Yes | Add-on |
| **Setup Time** | 10 min | 5 min | 5 min | 15 min |
| **Best For** | Full apps | Quick deploys | APIs | Enterprise |

---

## Prerequisites

### What You Need
- ✅ Render account (free)
- ✅ GitHub/GitLab/Bitbucket account
- ✅ Your Flask app files
- ✅ Git installed

### Create Render Account
1. Go to https://render.com/
2. Click "Get Started" → Sign up with GitHub
3. Authorize Render
4. Complete profile setup
5. **750 free hours/month!**

---

## Project Setup

### Required Files

Render needs these files:

```
HandDigitRecognition/
├── app_flask.py              # Your Flask app
├── requirements.txt          # Python dependencies  
├── render.yaml              # Render config (optional)
├── build.sh                 # Build commands (optional)
├── templates/               # HTML templates
├── static/                  # CSS/JS files
└── *.pth                    # Model files
```

---

## Configuration Files

### 1. requirements.txt

Update your requirements.txt:

```txt
Flask==3.0.0
gunicorn==21.2.0
torch==2.2.0
torchvision==0.17.0
numpy==1.26.0
opencv-python-headless==4.9.0.80
Pillow==10.2.0
```

### 2. Update app_flask.py

Ensure your app reads PORT from environment:

```python
if __name__ == '__main__':
    import os
    # Render provides PORT via environment variable
    port = int(os.environ.get('PORT', 10000))
    
    # Load the model on startup
    if load_model():
        print(f"\n🚀 Hand Digit Recognition Web App")
        print(f"📊 Model Type: {model_type}")
        print(f"🌐 Starting Flask server on port {port}...")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Failed to load model.")
```

### 3. render.yaml (Optional)

For infrastructure as code:

```yaml
services:
  - type: web
    name: hand-digit-recognition
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app_flask:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.7
```

---

## Deployment Process

### 🚀 METHOD 1: Deploy via Dashboard (Recommended)

#### **Step 1: Push to GitHub**

```powershell
# Initialize git (if not already done)
cd D:\MajorProject\HandDigitRecognition
git init

# Create .gitignore (I'll provide this)

# Add all files
git add .
git commit -m "Initial commit for Render deployment"

# Create GitHub repo
# Go to https://github.com/new
# Create repo: hand-digit-recognition

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/hand-digit-recognition.git
git branch -M main
git push -u origin main
```

#### **Step 2: Create Web Service on Render**

1. **Go to Render Dashboard:**
   - Visit https://dashboard.render.com/
   - Click "New +" button (top right)
   - Select "Web Service"

2. **Connect Repository:**
   - If first time: Click "Connect account" → Authorize GitHub
   - Find your repository: `hand-digit-recognition`
   - Click "Connect"

3. **Configure Service:**

   | Field | Value |
   |-------|-------|
   | **Name** | `hand-digit-recognition` |
   | **Region** | Choose closest to you (e.g., Oregon USA) |
   | **Branch** | `main` |
   | **Root Directory** | (leave blank) |
   | **Environment** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn app_flask:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120` |

4. **Select Plan:**
   - Choose **"Free"** plan
   - 750 hours/month
   - 512MB RAM
   - Click "Create Web Service"

5. **Wait for Deployment:**
   - First build: 5-15 minutes (installing PyTorch)
   - Watch build logs in real-time
   - Green "Live" badge when ready

6. **Get Your URL:**
   - Your app will be at: `https://hand-digit-recognition.onrender.com`
   - Click the URL to test!

#### **Step 3: Auto-Deploy Setup**

Auto-deploy is enabled by default!

```powershell
# Make changes to your code
git add .
git commit -m "Update feature"
git push

# Render automatically rebuilds and redeploys! 🎉
```

---

### 🖥️ METHOD 2: Deploy via render.yaml (Infrastructure as Code)

#### **Step 1: Create render.yaml**

Create this file in your project root:

```yaml
services:
  - type: web
    name: hand-digit-recognition
    env: python
    region: oregon
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app_flask:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.7
      - key: FLASK_ENV
        value: production
```

#### **Step 2: Deploy**

1. Push render.yaml to GitHub
2. In Render Dashboard:
   - Click "New +" → "Blueprint"
   - Connect your repository
   - Select `render.yaml`
   - Click "Apply"

Render creates everything from your YAML file!

---

## Advanced Configuration

### Environment Variables

**In Render Dashboard:**
1. Go to your service
2. Click "Environment" tab
3. Click "Add Environment Variable"
4. Add KEY and VALUE
5. Click "Save Changes" (triggers redeploy)

**Access in code:**
```python
import os
secret_key = os.environ.get('SECRET_KEY', 'default-value')
```

### Custom Start Command

For more control:

```bash
gunicorn app_flask:app \
  --bind 0.0.0.0:$PORT \
  --workers 1 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Build Command

If you need custom build steps:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

### Health Checks

Render automatically pings your app. Add a health endpoint:

```python
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_type
    }), 200
```

Then in Render:
- Environment tab
- Health Check Path: `/health`

---

## Free Tier Details

### What You Get
- **750 hours/month** - Enough for one always-on service
- **512MB RAM** - Sufficient for small ML models
- **Auto-scaling** - Spins down after 15min inactivity
- **Free SSL** - HTTPS included
- **Shared CPU** - Good for light traffic

### Limitations
- **Spin down** - Free services sleep after 15min inactivity
- **Cold starts** - First request after sleep takes ~30-60 seconds
- **Lower priority** - Paid services get priority
- **Build time** - Limited to 10-15 minutes

### Keeping Service Awake

**Option 1: Upgrade to Paid ($7/month)**
- No spin down
- Faster builds
- More RAM

**Option 2: Use a Keep-Alive Service**
- Ping your app every 10 minutes
- Use services like UptimeRobot (free)
- Not recommended for free tier (wastes resources)

---

## Monitoring & Management

### View Logs

**In Dashboard:**
1. Click your service
2. "Logs" tab shows real-time logs
3. Search and filter logs
4. Download logs

### View Metrics

**In Dashboard:**
1. "Metrics" tab
2. See:
   - Request rate
   - Response time
   - Memory usage
   - CPU usage

### Manual Deploy

**In Dashboard:**
1. "Manual Deploy" button
2. Select branch
3. Click "Deploy latest commit"

### Restart Service

**In Dashboard:**
1. Settings tab
2. Scroll to "Restart Service"
3. Click "Restart"

---

## Troubleshooting

### ❌ Issue 1: "Build Failed"

**Symptoms:**
```
Error: Failed to install torch==2.2.0
```

**Solutions:**

1. **Check requirements.txt:**
   ```txt
   # Use compatible versions
   torch==2.2.0
   torchvision==0.17.0
   ```

2. **Increase build timeout:**
   - Free tier: 15 min max
   - If timeout, optimize dependencies
   - Or upgrade to paid plan

3. **Use smaller packages:**
   ```txt
   # Use CPU-only PyTorch if possible
   torch==2.2.0+cpu
   ```

### ❌ Issue 2: "Service Not Starting"

**Symptoms:**
```
Error: Web service failed to bind to port
```

**Solutions:**

1. **Check Start Command:**
   ```bash
   gunicorn app_flask:app --bind 0.0.0.0:$PORT
   ```

2. **Check app_flask.py:**
   ```python
   port = int(os.environ.get('PORT', 10000))
   app.run(host='0.0.0.0', port=port)
   ```

3. **Check Logs:**
   - Look for Python errors
   - Check model loading errors

### ❌ Issue 3: "Out of Memory"

**Symptoms:**
```
Error: Container killed (OOM)
```

**Solutions:**

1. **Reduce workers:**
   ```bash
   gunicorn app_flask:app --workers 1
   ```

2. **Optimize model:**
   ```python
   # Load model once
   if model is None:
       model = load_model()
   ```

3. **Upgrade plan:**
   - Free: 512MB RAM
   - Starter: 2GB RAM ($7/month)

### ❌ Issue 4: "Cold Start Slow"

**Symptoms:**
- First request takes 30-60 seconds
- Subsequent requests are fast

**This is normal for free tier!**

**Solutions:**

1. **Accept it** - Free tier spins down
2. **Upgrade to paid** - No spin down
3. **Use keep-alive** - Ping every 10min (not recommended)
4. **Show loading message** - Inform users

### ❌ Issue 5: "Model Not Found"

**Symptoms:**
```
FileNotFoundError: digit_recognition_cnn_model.pth
```

**Solutions:**

1. **Check file is in repo:**
   ```powershell
   git ls-files | findstr .pth
   ```

2. **Check .gitignore:**
   - Don't exclude `*.pth` files

3. **Use absolute paths:**
   ```python
   import os
   BASE_DIR = os.path.dirname(__file__)
   model_path = os.path.join(BASE_DIR, 'digit_recognition_cnn_model.pth')
   ```

### ❌ Issue 6: "Static Files 404"

**Symptoms:**
- CSS/JS not loading
- 404 errors

**Solutions:**

1. **Check Flask setup:**
   ```python
   app = Flask(__name__)  # Auto-detects static/
   ```

2. **Use url_for:**
   ```html
   <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
   ```

3. **Check folder structure:**
   ```
   static/
     style.css
     script.js
   ```

---

## Custom Domain

### Add Your Domain

1. **In Render Dashboard:**
   - Click your service
   - "Settings" tab
   - Scroll to "Custom Domains"
   - Click "Add Custom Domain"
   - Enter: `myapp.com`

2. **Update DNS:**
   - Add CNAME record
   - Point to: `hand-digit-recognition.onrender.com`
   - Wait for DNS propagation (up to 24 hours)

3. **SSL Certificate:**
   - Render auto-generates
   - HTTPS enabled automatically
   - Certificate auto-renews

---

## Costs & Pricing

### Free Tier
- **750 hours/month** - ~31 days of uptime
- **512MB RAM**
- **Shared CPU**
- **Spins down after 15min**
- **Perfect for:** Demos, portfolios, testing

### Starter Plan ($7/month)
- **No spin down** - Always on
- **2GB RAM**
- **Shared CPU**
- **Faster builds**
- **Perfect for:** Small production apps

### Pro Plan ($25/month)
- **4GB RAM**
- **Dedicated CPU**
- **Even faster builds**
- **Perfect for:** Production apps

---

## Production Best Practices

### 1. Use Gunicorn
```bash
gunicorn app_flask:app --workers 1 --timeout 120 --access-logfile -
```

### 2. Add Health Checks
```python
@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200
```

### 3. Error Handling
```python
@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500
```

### 4. Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
app.logger.info('Application started')
```

### 5. Security Headers
```python
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

---

## Complete Deployment Checklist

- [ ] Create Render account
- [ ] Update `app_flask.py` with PORT handling
- [ ] Update `requirements.txt` (add gunicorn)
- [ ] Add health check endpoint
- [ ] Create `.gitignore`
- [ ] Push to GitHub
- [ ] Create Web Service on Render
- [ ] Configure Build/Start commands
- [ ] Select Free plan
- [ ] Wait for deployment
- [ ] Test deployed app
- [ ] Monitor logs
- [ ] (Optional) Add custom domain

---

## Commands & Tips

### Git Commands
```powershell
# Initialize
git init
git add .
git commit -m "Deploy to Render"

# Push
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main

# Update
git add .
git commit -m "Update"
git push  # Auto-deploys on Render!
```

### Testing Locally
```powershell
# Test with Gunicorn
pip install gunicorn
gunicorn app_flask:app --bind 0.0.0.0:5000

# Visit http://localhost:5000
```

---

## Resources

- 📚 **Render Docs:** https://render.com/docs
- 💬 **Render Community:** https://community.render.com/
- 📖 **Python Quickstart:** https://render.com/docs/deploy-flask
- 🎓 **Gunicorn Docs:** https://docs.gunicorn.org/

---

## 🎉 You're Ready!

Render is an excellent choice for your PyTorch Flask app:
- ✅ Generous free tier (750 hours)
- ✅ No size limits
- ✅ Easy deployment
- ✅ Auto HTTPS

**Start deploying now! 🎨**
