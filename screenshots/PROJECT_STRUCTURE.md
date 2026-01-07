# Alert-AID Project Structure

```
Alert-AID/
├── 📁 src/                          # React Frontend Source
│   ├── 📁 components/               # UI Components
│   │   ├── 📁 Dashboard/            # Dashboard widgets
│   │   ├── 📁 Map/                  # Map components
│   │   ├── 📁 Emergency/            # Emergency features
│   │   ├── 📁 Safety/               # Safety checklists
│   │   ├── 📁 Notifications/        # Alert notifications
│   │   ├── 📁 Location/             # Location services
│   │   └── 📁 Starfield/            # Visual effects
│   ├── 📁 pages/                    # Page components
│   │   ├── HomePage.tsx             # Landing page
│   │   ├── DashboardPage.tsx        # Main dashboard
│   │   ├── AlertsPage.tsx           # Alerts view
│   │   ├── PredictionsPage.tsx      # ML predictions
│   │   ├── EvacuationPage.tsx       # Evacuation routes
│   │   ├── FloodForecastPageV2.tsx  # 🌊 Flood forecasting (NEW!)
│   │   └── SafetyPage.tsx           # Safety resources
│   ├── 📁 services/                 # API Services
│   │   ├── apiService.ts            # Main API client
│   │   ├── indiaFloodApi.ts         # India flood API
│   │   ├── advancedMLApi.ts         # ML services
│   │   └── disasterDataService.ts   # External data
│   ├── 📁 contexts/                 # React Contexts
│   │   ├── AuthContext.tsx          # Authentication
│   │   ├── LocationContext.tsx      # Geolocation
│   │   └── NotificationContext.tsx  # Notifications
│   ├── 📁 hooks/                    # Custom Hooks
│   │   ├── useDashboard.ts          # Dashboard data
│   │   ├── useDisasterData.ts       # Disaster data
│   │   └── useRealTimeData.tsx      # Real-time updates
│   ├── 📁 styles/                   # Styling
│   │   ├── production-ui-system.ts  # Design system
│   │   └── GlobalStyles.ts          # Global CSS
│   └── 📁 types/                    # TypeScript types
│
├── 📁 backend/                      # Python FastAPI Backend
│   ├── enhanced_main.py             # Main API server
│   ├── 📁 ml/                       # ML Models
│   │   ├── lstm_flood_model.py      # LSTM flood prediction
│   │   ├── rf_flood_classifier.py   # Random Forest classifier
│   │   ├── ensemble_predictor.py    # Ensemble ML
│   │   ├── anomaly_detector.py      # Anomaly detection
│   │   └── smart_alerts.py          # Smart alerting
│   ├── 📁 routes/                   # API Routes
│   │   ├── frontend_api.py          # Frontend API (/api/flood/india/v2)
│   │   ├── external_apis.py         # External APIs (GDACS, FIRMS)
│   │   ├── flood_forecast.py        # Flood forecasting
│   │   └── india_rivers.py          # India river data
│   ├── 📁 models/                   # Trained Models
│   │   ├── lstm_flood_*.pt          # PyTorch LSTM models
│   │   ├── rf_flood_*.joblib        # Random Forest models
│   │   └── *.joblib                 # Other ML models
│   └── 📁 data/                     # Training Data
│       └── cauvery_*.csv            # River data
│
├── 📁 public/                       # Static Assets
│   └── index.html                   # HTML template
│
├── 📁 screenshots/                  # App Screenshots
│   ├── 01_homepage_dashboard.png
│   ├── 02_dashboard_overview.png
│   ├── 03_predictions_page.png
│   └── ...more screenshots
│
├── 📁 api/                          # Serverless Functions
│   └── index.py                     # Vercel serverless
│
├── package.json                     # NPM dependencies
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation
├── HACKATHON_SUBMISSION.md          # Hackathon docs
└── netlify.toml / vercel.json       # Deployment configs
```

## 🚀 Key Features

| Feature | Location | Description |
|---------|----------|-------------|
| **Flood Forecasting** | `src/pages/FloodForecastPageV2.tsx` | AI-powered flood predictions |
| **ML Ensemble** | `backend/ml/` | Random Forest + LSTM models |
| **Real-time Alerts** | `backend/routes/external_apis.py` | GDACS, FIRMS, IMD integration |
| **Interactive Maps** | `src/components/Map/` | Leaflet-based visualizations |
| **Safety Checklists** | `src/pages/SafetyPage.tsx` | Emergency preparedness |

## 🤖 ML Models Performance

| Model | Accuracy | F1 Score | Use Case |
|-------|----------|----------|----------|
| Flood RF | 94.04% | 93.48% | Flood risk classification |
| Fire RF | 91.46% | 94.11% | Fire risk prediction |
| Storm RF | 93.70% | 92.80% | Storm prediction |
| LSTM Flood | 91.23% | 87.80% | Time-series forecasting |

## 📱 Screenshots Available

1. **Homepage** - `01_homepage_dashboard.png`
2. **Dashboard** - `02_dashboard_overview.png`
3. **Predictions** - `03_predictions_page.png`
4. **Alerts** - `04_alerts_page.png`
5. **Evacuation** - `05_evacuation_routes.png`
6. **Shelters** - `06_emergency_shelters.png`
7. **Safety** - `07_safety_checklist.png`
8. **AI Panel** - `09_ai_analysis_panel.png`
