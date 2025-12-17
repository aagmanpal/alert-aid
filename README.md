# 🚨 Alert Aid - AI-Powered Disaster Prediction Dashboard

<div align="center">

**Professional disaster prediction and emergency management system powered by Machine Learning**

Alert Aid is a production-ready disaster prediction and alert system built with React, TypeScript, and advanced ML capabilities. Features an interactive 3D globe, real-time weather monitoring, and intelligent risk assessment.

</div>

## ✨ Features

- 🌍 **Interactive 3D Globe** - Real-time disaster visualization with risk heat maps
- ⭐ **Interactive Starfield Background** - Performance-optimized canvas animation with mouse interaction
- 📊 **ML-Powered Predictions** - Earthquake, flood, fire, and storm predictions
- 🌤️ **Real-time Weather** - Live weather data integration with 7-day forecasts

- 📍 **Smart Location Detection** - GPS-based location with manual override support

</div>- 🚑 **Emergency Response** - Communication hub and evacuation planning

- ♿ **Accessibility First** - WCAG 2.1 compliant with keyboard navigation

---- 🚀 **Production Ready** - Lazy loading, code splitting, and optimized builds



## 📋 Table of Contents## 🛠️ Technology Stack



- [Overview](#-overview)- **Frontend**: React 19.2.0, TypeScript 4.9.5

- [Features](#-features)- **3D Graphics**: Three.js, React Globe GL, @react-three/fiber

- [Tech Stack](#-tech-stack)- **Styling**: Styled Components 6.1.19

- [Installation](#-installation)- **Routing**: React Router DOM 7.9.4

- [Configuration](#-configuration)- **Charts**: Recharts 3.2.1

- [Usage](#-usage)- **Icons**: Lucide React

- [API Reference](#-api-reference)- **Testing**: Jest, React Testing Library

- [Deployment](#-deployment)- **Build**: Create React App 5.0.1

- [Contributing](#-contributing)

## 📋 Prerequisites

---

- Node.js 18.x or 20.x

## 🌟 Overview- npm 9.x or higher

- Modern browser with WebGL support

Alert Aid is a comprehensive disaster prediction and emergency management system that uses advanced Machine Learning models to predict natural disasters with **90%+ accuracy**. The platform provides real-time weather monitoring, air quality tracking, emergency response coordination, and evacuation planning—all in one intuitive dashboard.

## 🚀 Getting Started

### Why Alert Aid?

### Installation

- 🎯 **Highly Accurate ML Predictions** - 90%+ accuracy for flood, fire, storm, and earthquake predictions

- 🌍 **Real-time Monitoring** - Live weather data, 7-day forecasts, and air quality index tracking```bash

- 📍 **Smart Location Services** - GPS-based detection with manual search and triple API fallback# Clone the repository

- 🚨 **Emergency Response** - Indian emergency numbers (112, 100, 101, 102, 108) integrationgit clone <repository-url>

- 📊 **Comprehensive Reports** - Downloadable PDF/CSV reports with ML metricscd alert-aid

- 🎨 **Modern UI/UX** - Dark theme with responsive design and interactive 3D globe visualization

# Install dependencies

---npm install

```

## ✨ Features

### Development

### 🤖 Machine Learning Engine

- **4 Specialized ML Models**: Flood, Fire, Earthquake, and Storm prediction```bash

- **Advanced Training**: 25,000 samples per disaster type with 10+ features# Start the development server

- **High Accuracy**: 90%+ accuracy, F1 scores ranging from 89-94%npm start

- **Feature-Rich**: Includes precipitation, vegetation index, soil moisture, temperature changes

- **Model Persistence**: Joblib-based model storage with metadata tracking# Runs on http://localhost:3000

```

### 🌦️ Weather & Environmental Monitoring

- **Live Weather Data**: Real-time temperature, humidity, wind speed, pressure, UV index### Testing

- **7-Day Forecast**: Detailed daily forecasts with high/low temperatures and conditions

- **Air Quality Index (AQI)**: Real-time pollution monitoring with health advisories```bash

- **Pollutant Tracking**: PM2.5, PM10, NO2, O3, SO2, CO measurements# Run tests in watch mode

- **Risk Calculation**: Dynamic risk scoring based on weather + pollution factorsnpm test



### 📍 Location Services# Run tests with coverage

- **GPS Detection**: High-accuracy geolocation with `enableHighAccuracy: true`npm run test:ci

- **Manual Search**: City name search and coordinate input

- **Triple API Fallback**: OpenWeatherMap → Nominatim → BigDataCloud# Run tests once (CI mode)

- **Location Cache**: 30-minute cache for API optimizationCI=true npm test

```

### 🚨 Emergency Response

- **Indian Emergency Numbers**: 112, 100, 101, 102, 108### Building

- **SOS Alerts**: One-click emergency contact system

- **Evacuation Planning**: Route planning and safety zone mapping```bash

- **Resource Management**: Emergency supplies tracking and distribution# Create production build

- **Communication Hub**: Alert broadcasting and community coordinationnpm run build



### 📊 Data Export & Reporting# Build output will be in the ./build directory

- **PDF Reports**: Professional reports with ML performance metrics```

- **CSV Export**: Detailed data export with 12 ML metric columns

- **Live Data**: Real-time dashboard snapshots### Linting & Type Checking

- **ML Transparency**: Model accuracy, precision, recall, F1 scores included

```bash

### 🎨 User Experience# Run ESLint

- **Dark Theme**: Modern dark UI with gradient accentsnpm run lint

- **Responsive Design**: Mobile-first with breakpoints for all devices

- **Interactive Globe**: 3D Earth visualization with disaster risk heat mapping# Run TypeScript type check

- **Real-time Updates**: Auto-refresh with configurable intervals (5+ minutes)npm run type-check

- **Loading States**: Skeleton screens and smooth animations```

- **Error Handling**: Graceful fallbacks and user-friendly error messages

## 📁 Project Structure

---

```

## 🛠️ Tech Stackalert-aid/

├── public/              # Static assets

### Frontend├── src/

- **Framework**: React 19.2.0│   ├── components/      # React components

- **Language**: TypeScript 4.9.5│   │   ├── Dashboard/   # Main dashboard components

- **Styling**: Styled Components 6.1.19│   │   ├── Location/    # GPS and location services

- **Routing**: React Router DOM 7.9.4│   │   ├── Navigation/  # Navigation bar

- **State Management**: React Context API + Hooks│   │   ├── Starfield/   # Interactive starfield canvas

- **3D Graphics**: Three.js for globe visualization│   │   └── ...

- **Icons**: Lucide React│   ├── contexts/        # React contexts (Auth, Location, Notifications)

- **HTTP Client**: Axios│   ├── hooks/           # Custom React hooks

│   ├── pages/           # Page components

### Backend│   ├── services/        # API and external services

- **Framework**: FastAPI (Python)│   ├── styles/          # Styled components themes

- **Server**: Uvicorn (ASGI)│   ├── types/           # TypeScript type definitions

- **ML Library**: Scikit-learn│   ├── utils/           # Utility functions

- **Model Storage**: Joblib│   ├── __tests__/       # Unit tests

- **Data Processing**: NumPy, Pandas│   └── App.tsx          # Root component

- **Environment**: Python 3.9+├── backend/             # Python ML backend

│   ├── models/          # Trained ML models

### APIs & Services│   ├── routes/          # API routes

- **Weather Data**: OpenWeatherMap API│   └── services/        # Backend services

- **Air Quality**: OpenWeatherMap Air Pollution API└── .github/workflows/   # CI/CD pipelines

- **Geocoding**: OpenWeatherMap, Nominatim, BigDataCloud```

- **Reverse Geocoding**: Triple API fallback system

## 🔧 Configuration

---

### Environment Variables

## 🚀 Installation

Create `.env.local` for development:

### Prerequisites

```bash

- **Node.js** 16+ and npm# Development configuration

- **Python** 3.9+REACT_APP_OPENWEATHER_API_KEY=your_api_key

- **Git**REACT_APP_API_BASE_URL=http://localhost:8001

- **OpenWeatherMap API Key** (free tier available at [openweathermap.org](https://openweathermap.org/api))REACT_APP_ENVIRONMENT=development

```

### Step 1: Clone the Repository

Production configuration is in `.env.production`:

```bash

git clone https://github.com/yourusername/Alert-AID.git```bash

cd Alert-AID# See .env.production for full production config

```GENERATE_SOURCEMAP=false

REACT_APP_ENVIRONMENT=production

### Step 2: Frontend Setup```



```bash### Performance Optimization

# Install dependencies

npm install- **Lazy Loading**: GlobeRiskHero component uses React.lazy() and Suspense

- **Device-Optimized Rendering**: Starfield adjusts star count based on device (50/100/200)

# Start development server- **Reduced Motion**: Respects `prefers-reduced-motion` media query

npm start- **Code Splitting**: Automatic code splitting via React Router

```

## ♿ Accessibility Features

The frontend will run on `http://localhost:3001`

- Skip-to-content link for keyboard navigation

### Step 3: Backend Setup- Visible focus states on all interactive elements

- ARIA labels and attributes on components

```bash- Keyboard navigation support throughout

# Navigate to backend directory- Screen reader friendly

cd backend

## 🧪 Testing Strategy

# Create virtual environment (Windows PowerShell)

python -m venv venv- **Unit Tests**: Component and utility testing with Jest

.\venv\Scripts\Activate.ps1- **Integration Tests**: React Testing Library for user interactions

- **Coverage**: Aim for >80% code coverage

# Install dependencies- **CI/CD**: Automated testing on all pull requests

pip install -r requirements.txt

## 📦 Deployment

# Train ML models (first time only - takes 2-3 minutes)

python enhanced_main.py### Build for Production



# Start backend server```bash

python main.pynpm run build

``````



The backend will run on `http://localhost:8000`### Deploy to Vercel (Recommended)



### Step 4: Access the Application```bash

# Install Vercel CLI

- **Frontend**: http://localhost:3001npm i -g vercel

- **Backend API Docs**: http://localhost:8000/docs

# Deploy

---vercel



## ⚙️ Configuration# Deploy to production

vercel --prod

### Environment Variables```



#### Frontend (`.env`)### Deploy to Other Platforms

```env

REACT_APP_API_URL=http://localhost:8000The `build` folder can be deployed to any static hosting service:

```- Netlify

- AWS S3 + CloudFront

#### Backend (`backend/.env`)- Azure Static Web Apps

```env- GitHub Pages

# OpenWeatherMap API Key (REQUIRED)

OPENWEATHER_API_KEY=1801423b3942e324ab80f5b47afe0859## 🔐 Security



# Server Configuration### Known Vulnerabilities

HOST=0.0.0.0

PORT=8000As of the last audit, there are 9 vulnerabilities in development dependencies (transitive from `react-scripts`):

DEBUG=False- 3 moderate: postcss, webpack-dev-server

- 6 high: nth-check (via svgo chain)

# CORS Origins (comma-separated)

CORS_ORIGINS=http://localhost:3001,https://yourdomain.com**Note**: These are deep transitive dependencies in dev tooling only, not in production bundles. To resolve, upgrade to React 19 and migrate from `react-scripts` to Vite.

```

### Security Best Practices

### Getting an OpenWeatherMap API Key

- Source maps disabled in production (`.env.production`)

1. Visit [OpenWeatherMap](https://openweathermap.org/api)- API keys stored in environment variables

2. Sign up for a free account- HTTPS-only in production

3. Navigate to "API Keys" in your account dashboard- CSP headers enabled

4. Copy your API key- Input sanitization on all user inputs

5. Add it to `backend/.env` as `OPENWEATHER_API_KEY`

## 🐍 ML Backend

**Free tier includes**:

- 1,000 API calls/dayThe backend exposes endpoints to save and retrain ML models used by Alert Aid. Models are persisted to `backend/models/` and automatically loaded on startup if present.

- Current weather data

- 7-day forecast### Backend Endpoints

- Air quality data

- Geocoding- `POST /model/save` — Trigger saving current trained models to disk (background task)

- `POST /model/retrain` — Trigger retraining of models in background

---

Example (curl):

## 💻 Usage

```bash

### Key Featurescurl -X POST http://127.0.0.1:8001/model/save

curl -X POST http://127.0.0.1:8001/model/retrain

#### 1. Location Detection```

- Grant location permissions for automatic GPS detection

- Or click **Search** icon in navbar for manual location entryModel artifacts are saved as joblib files and metadata is stored in `metadata.json`.

- Search by city name or enter coordinates directly

## 🤝 Contributing

#### 2. View Dashboard

- **Risk Score**: Global disaster risk (0-10 scale)1. Fork the repository

- **Weather Widget**: Real-time weather with 7-day forecast2. Create a feature branch (`git checkout -b feature/amazing-feature`)

- **AQI Widget**: Air quality with health advisories3. Commit your changes (`git commit -m 'Add amazing feature'`)

- **Current Alerts**: Active disaster warnings4. Push to the branch (`git push origin feature/amazing-feature`)

- **ML Predictions**: Model accuracy metrics5. Open a Pull Request



#### 3. Emergency Response## 📄 License

- Click **Emergency** tab for SOS panel

- View Indian emergency numbers: 112, 100, 101, 102, 108This project is licensed under the MIT License.

- Access evacuation planning and resource management

## 🙏 Acknowledgments

#### 4. Download Reports

- Click **Download Report** button in dashboard- Create React App for the initial setup

- Exports PDF + CSV with:- OpenWeather API for weather data

  - Weather data- Three.js community for 3D graphics support

  - Risk assessments
  - ML model performance (accuracy, precision, F1 scores)

#### 5. Manual Refresh
- Click **Refresh** icon in navbar
- Auto-refresh configurable (minimum 5 minutes)

---

## 📡 API Reference

### Base URL
```
http://localhost:8000/api
```

### Weather Endpoints

#### Get Current Weather
```http
GET /weather/{lat}/{lon}
```

#### Get 7-Day Forecast
```http
GET /weather/forecast/{lat}/{lon}?days=7
```

#### Get Air Quality
```http
GET /weather/air-quality/{lat}/{lon}
```

**Response:**
```json
{
  "aqi": 2,
  "level": "Fair",
  "color": "#FFEB3B",
  "components": {
    "pm2_5": 12.5,
    "pm10": 25.0,
    "no2": 15.3,
    "o3": 45.2
  }
}
```

### ML Prediction Endpoints

#### Get ML Metrics
```http
GET /predict/ml-metrics
```

**Response:**
```json
{
  "flood": {
    "accuracy": 0.9432,
    "precision": 0.9321,
    "f1": 0.9388
  },
  "fire": {
    "accuracy": 0.9421,
    "f1": 0.9355
  },
  "storm": {
    "accuracy": 0.9275,
    "f1": 0.9221
  }
}
```

**Full API Documentation**: Visit http://localhost:8000/docs

---

## 🌐 Deployment

### Deploy Frontend to Vercel

1. **Push to GitHub** (see below)

2. **Import to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "Import Project"
   - Select your GitHub repository
   - Configure build settings:
     - Build Command: `npm run build`
     - Output Directory: `build`
     - Install Command: `npm install`

3. **Add Environment Variables**:
   - `REACT_APP_API_URL` = your backend URL

4. **Deploy**: Click "Deploy"

### Deploy Backend to Railway

1. **Push to GitHub**

2. **Deploy to Railway**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Choose `backend` directory as root

3. **Configure**:
   - Add `OPENWEATHER_API_KEY` environment variable
   - Add `CORS_ORIGINS` with your Vercel URL
   - Railway will auto-detect Python and install dependencies

4. **Deploy**: Railway will automatically deploy

### Alternative: Deploy Backend to Render

1. Push to GitHub
2. Create new "Web Service" on [render.com](https://render.com)
3. Connect your repository
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python main.py`
   - Add environment variables

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **OpenWeatherMap** for weather and air quality APIs
- **Nominatim (OpenStreetMap)** for geocoding fallback
- **BigDataCloud** for additional geocoding support
- **React Team** for the amazing framework
- **FastAPI Team** for the backend framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Alert-AID/issues)

---

<div align="center">

**Made with ❤️ for safer communities**

</div>
