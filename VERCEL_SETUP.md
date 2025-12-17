# 🚀 VERCEL ENVIRONMENT VARIABLE SETUP

## Quick Fix: Connect Frontend to Backend

Your Railway backend is deployed successfully! Now you need to tell Vercel where to find it.

## Step 1: Get Your Railway Backend URL

1. Go to your Railway dashboard
2. Click on your **Alert-AID** project
3. Copy the **Public URL** (should look like: `https://web-production-xxxx.up.railway.app`)

## Step 2: Add Environment Variable to Vercel

### Option A: Vercel Dashboard (Recommended)

1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Click on your **Alert-AID** project
3. Go to **Settings** → **Environment Variables**
4. Add a new variable:
   - **Name**: `REACT_APP_API_URL`
   - **Value**: Your Railway URL (e.g., `https://web-production-2f56.up.railway.app`)
   - **Environment**: Select all (Production, Preview, Development)
5. Click **Save**
6. Go to **Deployments** tab
7. Click the **three dots (...)** on the latest deployment
8. Click **Redeploy**

### Option B: Vercel CLI

```bash
# Install Vercel CLI if you haven't
npm install -g vercel

# Login
vercel login

# Add environment variable
vercel env add REACT_APP_API_URL

# When prompted, paste your Railway URL
# Select: Production, Preview, Development

# Redeploy
vercel --prod
```

## Step 3: Verify the Fix

After redeployment:

1. Visit your Vercel app: `https://your-app.vercel.app`
2. Go to **System Diagnostics** (navigation menu)
3. You should see:
   - ✅ **Alert Aid Backend**: Online (green)
   - ✅ **OpenWeatherMap**: Online (green)  
   - ✅ **IP Geolocation**: Online (green)
   - **System Status**: Should show higher percentage (90%+)

## Step 4: Update Railway CORS (Important!)

Your backend needs to allow requests from Vercel:

1. Go to Railway dashboard
2. Click your Alert-AID project
3. Go to **Variables** tab
4. Add or update:
   - **Name**: `CORS_ORIGINS`
   - **Value**: `https://your-vercel-app.vercel.app`
   - (Replace with your actual Vercel URL)
5. Railway will automatically redeploy

## Troubleshooting

### If backend still shows offline:

1. **Check Railway logs**:
   - Railway dashboard → Deployments → View Logs
   - Look for "Uvicorn running on..." message
   - Should show: `Application startup complete`

2. **Test Railway backend directly**:
   - Open: `https://your-railway-url.railway.app/health`
   - Should return: `{"status": "healthy"}`

3. **Check Vercel build logs**:
   - Vercel dashboard → Deployments → Build Logs
   - Verify `REACT_APP_API_URL` is set correctly

4. **Clear browser cache**:
   - Hard refresh: `Ctrl + Shift + R` (Windows) or `Cmd + Shift + R` (Mac)

## Example Configuration

**Railway Backend URL**: `https://web-production-2f56.up.railway.app`

**Vercel Environment Variables**:
```
REACT_APP_API_URL=https://web-production-2f56.up.railway.app
REACT_APP_OPENWEATHER_API_KEY=1801423b3942e324ab80f5b47afe0859
```

**Railway Environment Variables**:
```
OPENWEATHER_API_KEY=1801423b3942e324ab80f5b47afe0859
CORS_ORIGINS=https://your-vercel-app.vercel.app
PORT=8000
```

---

## ✅ Success Checklist

- [ ] Railway backend deployed and running
- [ ] Got Railway public URL
- [ ] Added `REACT_APP_API_URL` to Vercel
- [ ] Redeployed Vercel app
- [ ] Added Vercel URL to Railway `CORS_ORIGINS`
- [ ] System Diagnostics shows all services online
- [ ] Weather data loads correctly
- [ ] AQI widget shows data
- [ ] ML predictions work

---

**Need Help?** Check the Railway and Vercel deployment logs for specific error messages.
