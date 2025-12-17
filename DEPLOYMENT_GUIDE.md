# GitHub Repository Setup Instructions

## Step 1: Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and log in
2. Click the **"+"** icon in the top-right corner and select **"New repository"**
3. Set the repository name to: **Alert-AID**
4. Add a description: 
   ```
   AI-Powered Disaster Prediction Dashboard with 90%+ ML accuracy, real-time weather monitoring, AQI tracking, and Indian emergency response system
   ```
5. Select **Public** repository
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click **"Create repository"**

## Step 2: Push Your Code to GitHub

After creating the repository, GitHub will show you commands to push an existing repository. Run these commands in your terminal:

```powershell
# Navigate to your project directory
cd "c:\Visual Studio Code\alert-aid"

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/Alert-AID.git

# Rename branch to main (if needed)
git branch -M main

# Push your code to GitHub
git push -u origin main
```

**Alternative: Using SSH (if you have SSH keys set up)**
```powershell
git remote add origin git@github.com:YOUR_USERNAME/Alert-AID.git
git branch -M main
git push -u origin main
```

## Step 3: Verify on GitHub

1. Go to your repository: `https://github.com/YOUR_USERNAME/Alert-AID`
2. Verify that all files are uploaded
3. Check that the README.md displays correctly

## Step 4: Deploy to Vercel

### Option A: Using Vercel Dashboard (Recommended)
 
1. Go to [vercel.com](https://vercel.com) and log in (or sign up with GitHub)
2. Click **"Add New Project"** or **"Import Project"**
3. Select **"Import Git Repository"**
4. Find and select your **Alert-AID** repository
5. Configure the project:
   - **Framework Preset**: Create React App
   - **Root Directory**: `./` (leave as default)
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `build` (auto-detected)
   - **Install Command**: `npm install` (auto-detected)
6. Add Environment Variable:
   - Name: `REACT_APP_API_URL`
   - Value: Your backend URL (e.g., `https://your-backend.railway.app`)
   - Or temporarily use: `http://localhost:8000` for testing
7. Click **"Deploy"**

Vercel will:
- Install dependencies
- Build your project
- Deploy to a URL like: `https://alert-aid-xyz123.vercel.app`

### Option B: Using Vercel CLI

```powershell
# Install Vercel CLI globally
npm install -g vercel

# Navigate to your project
cd "c:\Visual Studio Code\alert-aid"

# Login to Vercel
vercel login

# Deploy (follow the prompts)
vercel

# For production deployment
vercel --prod
```

## Step 5: Deploy Backend to Railway

1. Go to [railway.app](https://railway.app) and log in with GitHub
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your **Alert-AID** repository
5. **IMPORTANT**: Railway will auto-detect the project type:
   - It should detect Python (from `requirements.txt` in root and `nixpacks.toml`)
   - If it tries to build the frontend, manually configure:
     - **Settings** → **Build** → **Build Command**: `pip install -r backend/requirements.txt`
     - **Settings** → **Deploy** → **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variables in Railway Dashboard:
   - `OPENWEATHER_API_KEY`: `1801423b3942e324ab80f5b47afe0859`
   - `CORS_ORIGINS`: `https://your-vercel-app.vercel.app` (add your Vercel URL after Vercel deployment)
   - `PORT`: Railway will auto-assign this, but you can set `8000` as default
7. Click **"Deploy"**

Railway will:
- Detect Python project via `nixpacks.toml` and `Procfile`
- Install dependencies from `requirements.txt`
- Run the backend on the assigned port
- Provide a public URL like: `https://alert-aid-production.up.railway.app`

8. **Copy the Railway URL** and update your Vercel environment variable:
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add or update `REACT_APP_API_URL` to your Railway URL
   - Redeploy your Vercel project (or it will auto-redeploy)

## Step 6: Final Verification

1. **Visit your Vercel URL**: `https://your-app.vercel.app`
2. **Check that**:
   - Location detection works
   - Weather data loads
   - Air quality widget shows data
   - ML metrics are available
   - Emergency response panel displays correctly
3. **Test Features**:
   - Manual location search
   - Download reports (PDF/CSV)
   - Emergency numbers display
   - 7-day forecast

## Troubleshooting

### Issue: Frontend can't connect to backend
**Solution**: 
- Check that `REACT_APP_API_URL` in Vercel points to your Railway URL
- Verify CORS_ORIGINS in Railway includes your Vercel URL
- Redeploy both services

### Issue: "Module not found" errors in Vercel
**Solution**:
- Check that all dependencies are in `package.json`
- Clear Vercel cache and redeploy

### Issue: Backend fails to start on Railway
**Solution**:
- Check Railway logs for errors
- Verify `OPENWEATHER_API_KEY` is set
- Ensure `requirements.txt` is in backend folder
- Check that `main.py` exists

### Issue: AQI not loading
**Solution**:
- Verify OpenWeatherMap API key is valid
- Check API quota hasn't been exceeded (1000 calls/day on free tier)
- Check browser console for errors

## Repository URL Format

After setup, your repository will be at:
```
https://github.com/YOUR_USERNAME/Alert-AID
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Quick Reference

**Local Development:**
- Frontend: `npm start` → http://localhost:3001
- Backend: `cd backend && python main.py` → http://localhost:8000

**Production URLs:**
- Frontend (Vercel): `https://alert-aid-[random].vercel.app`
- Backend (Railway): `https://alert-aid-production.up.railway.app`

## Need Help?

- Check README.md for detailed documentation
- Review GitHub Issues for common problems
- Verify environment variables are set correctly
- Check browser console and network tab for errors
- Review Railway/Vercel deployment logs

---

**Ready to deploy!** 🚀 Follow the steps above to get your Alert Aid dashboard live.
