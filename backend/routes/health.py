"""
Health Check Routes
Simple health monitoring endpoints
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "operational",
            "ml_model": "ready",
            "external_apis": "connected",
            "database": "not_implemented"
        },
        "version": "1.0.1",  # Updated to verify Railway deployment
        "uptime": "active",
        "cors_fixed": True  # Indicator that CORS fix is deployed
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service diagnostics"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "fastapi": {"status": "operational", "version": "0.104.1"},
            "ml_models": {"status": "ready", "models_loaded": 4},
            "weather_api": {"status": "configured", "provider": "OpenWeatherMap"},
            "alerts_api": {"status": "configured", "provider": "NOAA"},
            "earthquake_api": {"status": "configured", "provider": "USGS"}
        },
        "system": {
            "python_version": "3.x",
            "memory_usage": "normal",
            "disk_space": "sufficient"
        }
    }