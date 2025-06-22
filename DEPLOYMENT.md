# 🚀 Deployment Guide

This guide shows you how to set up automatic deployment for your Streamlit app so that it updates automatically whenever you push changes to GitHub.

## Method 1: Streamlit Community Cloud (Recommended - Free & Easy)

### Step 1: Prepare Your Repository
1. Make sure your code is in a GitHub repository
2. Ensure `app.py` is in the root directory
3. Ensure `requirements.txt` contains all dependencies

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch (usually `main`), and main file (`app.py`)
5. Click "Deploy!"

### Step 3: Automatic Updates
✅ **That's it!** Every time you push to GitHub, Streamlit Cloud automatically redeploys your app.

**Benefits:**
- ✅ Completely free
- ✅ Automatic SSL certificate
- ✅ Custom domain support
- ✅ Automatic deployments on Git push
- ✅ No configuration required

---

## Method 2: Railway (Alternative - Also Free)

### Step 1: Connect GitHub
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "Deploy from GitHub repo"
4. Select your repository

### Step 2: Configure (if needed)
Railway auto-detects Streamlit apps, but you can use the included `railway.json` for custom configuration.

### Step 3: Enjoy Auto-Deployment
✅ Automatic deployments on every GitHub push!

---

## Method 3: Heroku (Advanced)

### Step 1: Heroku Setup
```bash
# Install Heroku CLI
# Create new Heroku app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main
```

### Step 2: Enable Auto-Deploy
1. Go to your Heroku dashboard
2. Navigate to your app → Deploy tab
3. Connect to GitHub
4. Enable "Automatic deploys" from main branch

### Step 3: Environment Variables (if needed)
```bash
heroku config:set STREAMLIT_SERVER_PORT=$PORT
```

---

## Method 4: Docker + Any Cloud Provider

### Build and Run Locally:
```bash
# Build Docker image
docker build -t digit-generator .

# Run container
docker run -p 8501:8501 digit-generator
```

### Deploy to Cloud:
- **Google Cloud Run**: `gcloud run deploy`
- **AWS App Runner**: Connect GitHub repository
- **Azure Container Instances**: Deploy from container registry

---

## Method 5: GitHub Actions + Custom Deployment

The included `.github/workflows/deploy.yml` provides:
- ✅ Automatic testing on every push
- ✅ Deployment triggers
- ✅ Can be customized for any hosting provider

---

## 🔧 Troubleshooting

### Common Issues:

**1. App won't start:**
- Check that `requirements.txt` includes all dependencies
- Ensure `app.py` is in the root directory

**2. Model file missing:**
- Your app handles this gracefully by using an untrained model
- For production, train and include `generator_model.pth`

**3. Port issues:**
- Streamlit Cloud: No action needed
- Heroku/Railway: Uses `$PORT` environment variable (already configured)

### Performance Tips:
- Use `@st.cache_resource` for model loading (already implemented)
- Keep your repository under 1GB for faster deployments
- Consider using Git LFS for large model files

---

## 📱 Quick Commands

```bash
# Test locally
streamlit run app.py

# Check deployment
git add .
git commit -m "Update app"
git push origin main
# Your app will auto-update in 2-3 minutes!
```

---

**Choose Streamlit Community Cloud for the easiest setup with zero configuration!**
