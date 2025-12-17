# 🚨 URGENT FIX NEEDED - Environment Variable Configuration

## Current Issue
Your Vercel deployment is still pointing to `localhost` because the environment variable wasn't set correctly.

## ✅ IMMEDIATE ACTION REQUIRED

### Step 1: Get Your Railway Backend URL

Your Railway backend is deployed. Go to Railway and copy the URL:

**Example**: `https://web-production-2f56.up.railway.app`

### Step 2: Configure Vercel Environment Variable

**⚠️ IMPORTANT: You MUST do this in Vercel Dashboard**

1. Go to: https://vercel.com/dashboard
2. Click your **Alert-AID** project
3. Click **Settings** (left sidebar)
4. Click **Environment Variables**
5. Click **Add New**
6. Enter:
   ```
   Key: REACT_APP_API_URL
   Value: https://web-production-2f56.up.railway.app
   ```
   (Replace with YOUR actual Railway URL)
7. **Select ALL environments**: ✅ Production ✅ Preview ✅ Development
8. Click **Save**

### Step 3: Force Redeploy

**Option A: Via Dashboard (Recommended)**
1. Go to **Deployments** tab
2. Find the latest deployment
3. Click the **...** (three dots menu)
4. Click **Redeploy**
5. Click **Redeploy** again to confirm

**Option B: Push a dummy commit**
```bash
git commit --allow-empty -m "Trigger Vercel redeploy with environment variables"
git push origin main
```

### Step 4: Verify After Redeploy (2-3 minutes)

Visit your Vercel app and check **System Diagnostics**:
- ✅ Alert Aid Backend: Should show **Online** (green)
- ✅ OpenWeatherMap: Should show **Online** (green)
- ⚠️ IP Geolocation: May show offline (this is a rate limit issue, not critical)
- System should show **67%+ Operational** (without IP Geolocation)

### Step 5: Configure CORS in Railway

1. Go to Railway Dashboard
2. Click your Alert-AID project
3. Click **Variables** tab
4. Add or update:
   ```
   Key: CORS_ORIGINS
   Value: https://your-vercel-app.vercel.app,https://your-vercel-app-git-main.vercel.app
   ```
   (Replace with YOUR actual Vercel URLs - you may have multiple preview URLs)

---

## 🔍 Troubleshooting

### If Backend Still Shows Offline:

1. **Check Vercel Build Logs**:
   - Vercel Dashboard → Deployments → Click latest → View Build Logs
   - Search for `REACT_APP_API_URL`
   - Should show: `REACT_APP_API_URL=https://your-railway-url.railway.app`
   - If it shows `localhost`, the environment variable wasn't set correctly

2. **Check Railway Logs**:
   - Railway Dashboard → Deployments → View Logs
   - Should show: `Uvicorn running on http://0.0.0.0:8000`
   - Should show: `Application startup complete`
   - Test directly: `https://your-railway-url.railway.app/health`
   - Should return: `{"status":"healthy"}`

3. **Hard Refresh Browser**:
   - Windows: `Ctrl + Shift + R`
   - Mac: `Cmd + Shift + R`
   - Or open in Incognito/Private mode

4. **Check Network Tab**:
   - Open browser DevTools (F12)
   - Go to Network tab
   - Refresh page
   - Look for failed requests to `localhost` → If you see these, environment variable isn't set

### If AQI Widget Not Working:

The AQI service now calls `/api/weather/air-quality/` which requires:
- Backend to be online
- Environment variable set correctly
- Latest code deployed (just pushed fix)

Once backend is online, AQI should work automatically.

### IP Geolocation Offline:

This is using `ipapi.co` which has rate limits. This is **NOT CRITICAL** because:
- App uses GPS location as primary method
- Falls back to manual location selection
- Default location (Jaipur) is configured

---

## ✅ Success Checklist

- [ ] Got Railway backend URL from Railway dashboard
- [ ] Added `REACT_APP_API_URL` in Vercel environment variables
- [ ] Selected ALL environments (Production, Preview, Development)
- [ ] Clicked Save in Vercel
- [ ] Triggered redeploy (via dashboard or git push)
- [ ] Waited 2-3 minutes for deployment
- [ ] Checked System Diagnostics - Backend shows Online
- [ ] Added `CORS_ORIGINS` in Railway variables
- [ ] Weather widget loads data
- [ ] AQI widget shows air quality data
- [ ] ML predictions work

---

## 📝 Expected URLs

**Vercel Frontend**: `https://your-app-name.vercel.app`
**Railway Backend**: `https://web-production-xxxx.up.railway.app`

**Environment Variables**:

Vercel:
```env
REACT_APP_API_URL=https://web-production-xxxx.up.railway.app
REACT_APP_OPENWEATHER_API_KEY=1801423b3942e324ab80f5b47afe0859
```

Railway:
```env
OPENWEATHER_API_KEY=1801423b3942e324ab80f5b47afe0859
CORS_ORIGINS=https://your-app-name.vercel.app
PORT=8000
```

---

## 🆘 Still Not Working?

1. **Screenshot**:
   - Vercel Environment Variables page (showing REACT_APP_API_URL is set)
   - System Diagnostics page
   - Browser DevTools Network tab (showing the API calls)

2. **Check**:
   - Railway deployment logs
   - Vercel build logs
   - Browser console (F12) for errors

The most common issue is that **Vercel environment variable wasn't saved correctly**. Double-check it's there and redeploy!
