"""
Alerts API Routes
Handles alert management and notification systems
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import random
import math
from typing import Dict, List, Any, Optional

router = APIRouter()

# In-memory alert storage (in production, use database)
alerts_storage = []
alert_id_counter = 1

@router.get("/alerts")
async def get_all_alerts(active_only: bool = False, severity: Optional[str] = None):
    """Get all alerts with optional filtering"""
    try:
        filtered_alerts = alerts_storage.copy()
        
        if active_only:
            current_time = datetime.now()
            filtered_alerts = [
                alert for alert in filtered_alerts 
                if datetime.fromisoformat(alert["expires_at"]) > current_time
            ]
        
        if severity:
            filtered_alerts = [
                alert for alert in filtered_alerts 
                if alert["severity"].lower() == severity.lower()
            ]
        
        return {
            "alerts": filtered_alerts,
            "total_count": len(filtered_alerts),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alerts retrieval error: {str(e)}")

@router.post("/alerts")
async def create_alert(alert_data: Dict[str, Any]):
    """Create a new alert"""
    try:
        global alert_id_counter
        
        # Validate required fields
        required_fields = ["type", "severity", "title", "description"]
        for field in required_fields:
            if field not in alert_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create alert
        new_alert = {
            "id": alert_id_counter,
            "type": alert_data["type"],
            "severity": alert_data["severity"],
            "title": alert_data["title"],
            "description": alert_data["description"],
            "location": alert_data.get("location", {}),
            "affected_areas": alert_data.get("affected_areas", []),
            "created_at": datetime.now().isoformat(),
            "expires_at": alert_data.get(
                "expires_at", 
                (datetime.now() + timedelta(hours=24)).isoformat()
            ),
            "status": "active",
            "source": alert_data.get("source", "alert_aid_system"),
            "emergency_contacts": alert_data.get("emergency_contacts", []),
            "recommended_actions": alert_data.get("recommended_actions", [])
        }
        
        alerts_storage.append(new_alert)
        alert_id_counter += 1
        
        return {
            "alert": new_alert,
            "message": "Alert created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert creation error: {str(e)}")

@router.get("/alerts/{alert_id}")
async def get_alert(alert_id: int):
    """Get specific alert by ID"""
    try:
        alert = next((a for a in alerts_storage if a["id"] == alert_id), None)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"alert": alert}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert retrieval error: {str(e)}")

@router.put("/alerts/{alert_id}")
async def update_alert(alert_id: int, update_data: Dict[str, Any]):
    """Update an existing alert"""
    try:
        alert_index = next((i for i, a in enumerate(alerts_storage) if a["id"] == alert_id), None)
        
        if alert_index is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Update alert fields
        alert = alerts_storage[alert_index]
        for key, value in update_data.items():
            if key != "id":  # Don't allow ID changes
                alert[key] = value
        
        alert["last_modified"] = datetime.now().isoformat()
        
        return {
            "alert": alert,
            "message": "Alert updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert update error: {str(e)}")

@router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: int):
    """Delete an alert"""
    try:
        global alerts_storage
        
        alert_index = next((i for i, a in enumerate(alerts_storage) if a["id"] == alert_id), None)
        
        if alert_index is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        deleted_alert = alerts_storage.pop(alert_index)
        
        return {
            "message": "Alert deleted successfully",
            "deleted_alert_id": deleted_alert["id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert deletion error: {str(e)}")

@router.get("/alerts/location/{lat}/{lon}")
async def get_location_alerts(lat: float, lon: float, radius_km: float = 50):
    """Get alerts for a specific location within radius"""
    try:
        location_alerts = []
        
        for alert in alerts_storage:
            # Check if alert affects this location
            if _is_location_affected(lat, lon, alert, radius_km):
                location_alerts.append(alert)
        
        # Also generate some realistic area-specific alerts
        generated_alerts = _generate_location_alerts(lat, lon)
        location_alerts.extend(generated_alerts)
        
        return {
            "alerts": location_alerts,
            "location": {"latitude": lat, "longitude": lon, "radius_km": radius_km},
            "total_count": len(location_alerts),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Location alerts error: {str(e)}")

def _is_location_affected(lat: float, lon: float, alert: Dict, radius_km: float) -> bool:
    """Check if location is within alert's affected area"""
    
    alert_location = alert.get("location", {})
    
    if not alert_location.get("latitude") or not alert_location.get("longitude"):
        return False
    
    # Simple distance calculation (haversine would be more accurate)
    alert_lat = alert_location["latitude"]
    alert_lon = alert_location["longitude"]
    
    # Rough distance calculation in km
    lat_diff = abs(lat - alert_lat) * 111  # 1 degree ≈ 111 km
    lon_diff = abs(lon - alert_lon) * 111 * abs(math.cos(math.radians(lat)))
    distance = (lat_diff**2 + lon_diff**2)**0.5
    
    return distance <= radius_km

def _generate_location_alerts(lat: float, lon: float) -> List[Dict]:
    """Generate realistic alerts for a location"""
    
    generated_alerts = []
    
    # Weather-based alerts
    if random.random() < 0.3:  # 30% chance of weather alert
        weather_alert = {
            "id": f"weather_{random.randint(1000, 9999)}",
            "type": "weather",
            "severity": random.choice(["low", "moderate", "high"]),
            "title": random.choice([
                "Severe Weather Watch", "High Wind Advisory", 
                "Heavy Rain Warning", "Storm Alert"
            ]),
            "description": "Weather conditions may pose risks to the area. Monitor updates and take appropriate precautions.",
            "location": {"latitude": lat, "longitude": lon},
            "affected_areas": [f"Within 25km of coordinates {lat:.2f}, {lon:.2f}"],
            "created_at": (datetime.now() - timedelta(hours=random.randint(1, 12))).isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=random.randint(6, 48))).isoformat(),
            "status": "active",
            "source": "weather_monitoring_system",
            "recommended_actions": [
                "monitor_weather_updates", 
                "secure_loose_items", 
                "avoid_unnecessary_travel"
            ]
        }
        generated_alerts.append(weather_alert)
    
    # Fire risk alerts
    if random.random() < 0.2 and abs(lat) > 25:  # 20% chance in fire-prone areas
        fire_alert = {
            "id": f"fire_{random.randint(1000, 9999)}",
            "type": "wildfire",
            "severity": random.choice(["moderate", "high"]),
            "title": "Fire Weather Warning",
            "description": "Dry conditions and winds create elevated fire risk. Exercise extreme caution with any ignition sources.",
            "location": {"latitude": lat, "longitude": lon},
            "affected_areas": [f"Fire risk zone near {lat:.2f}, {lon:.2f}"],
            "created_at": (datetime.now() - timedelta(hours=random.randint(2, 24))).isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=random.randint(12, 72))).isoformat(),
            "status": "active",
            "source": "fire_monitoring_system",
            "recommended_actions": [
                "no_outdoor_burning", 
                "prepare_evacuation_routes", 
                "monitor_fire_conditions"
            ]
        }
        generated_alerts.append(fire_alert)
    
    # Earthquake preparedness (for seismic zones)
    if random.random() < 0.1 and _is_seismic_zone(lat, lon):  # 10% chance in seismic areas
        earthquake_alert = {
            "id": f"earthquake_{random.randint(1000, 9999)}",
            "type": "earthquake",
            "severity": "moderate",
            "title": "Earthquake Preparedness Reminder",
            "description": "This area has elevated seismic activity. Ensure earthquake preparedness measures are in place.",
            "location": {"latitude": lat, "longitude": lon},
            "affected_areas": [f"Seismic zone including {lat:.2f}, {lon:.2f}"],
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 7))).isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
            "status": "active",
            "source": "seismic_monitoring_system",
            "recommended_actions": [
                "earthquake_kit_ready", 
                "secure_heavy_objects", 
                "know_evacuation_routes"
            ]
        }
        generated_alerts.append(earthquake_alert)
    
    return generated_alerts

def _is_seismic_zone(lat: float, lon: float) -> bool:
    """Check if location is in a known seismic zone"""
    seismic_zones = [
        {"lat_range": (32, 42), "lon_range": (-125, -114)},  # California
        {"lat_range": (35, 45), "lon_range": (135, 145)},    # Japan
        {"lat_range": (-45, -35), "lon_range": (165, 180)},  # New Zealand
        {"lat_range": (36, 42), "lon_range": (25, 35)},      # Turkey/Greece
    ]
    
    for zone in seismic_zones:
        if (zone["lat_range"][0] <= lat <= zone["lat_range"][1] and 
            zone["lon_range"][0] <= lon <= zone["lon_range"][1]):
            return True
    
    return False

@router.post("/alerts/emergency")
async def create_emergency_alert(emergency_data: Dict[str, Any]):
    """Create high-priority emergency alert"""
    try:
        # Emergency alerts get highest priority
        emergency_alert = {
            "type": "emergency",
            "severity": "critical",
            "title": emergency_data.get("title", "Emergency Alert"),
            "description": emergency_data.get("description", "Emergency situation detected"),
            "location": emergency_data.get("location", {}),
            "affected_areas": emergency_data.get("affected_areas", []),
            "emergency_contacts": [
                {"service": "Emergency Services", "number": "911"},
                {"service": "Local Emergency Management", "number": "emergency_line"},
                {"service": "Alert Aid Support", "number": "alert_support"}
            ],
            "recommended_actions": emergency_data.get("recommended_actions", [
                "follow_official_instructions",
                "evacuate_if_instructed", 
                "stay_informed"
            ]),
            "source": "emergency_system"
        }
        
        return await create_alert(emergency_alert)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency alert error: {str(e)}")

@router.get("/alerts/statistics")
async def get_alert_statistics():
    """Get alert system statistics"""
    try:
        total_alerts = len(alerts_storage)
        active_alerts = len([a for a in alerts_storage 
                           if datetime.fromisoformat(a["expires_at"]) > datetime.now()])
        
        severity_counts = {}
        type_counts = {}
        
        for alert in alerts_storage:
            # Count by severity
            severity = alert["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by type
            alert_type = alert["type"]
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "expired_alerts": total_alerts - active_alerts,
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

# Initialize with some sample alerts
def _initialize_sample_alerts():
    """Initialize system with sample alerts for demonstration"""
    
    sample_alerts = [
        {
            "type": "weather",
            "severity": "moderate",
            "title": "Severe Weather Watch",
            "description": "Strong winds and heavy rain expected in the region. Exercise caution when traveling.",
            "location": {"latitude": 37.7749, "longitude": -122.4194},  # San Francisco
            "affected_areas": ["San Francisco Bay Area", "Peninsula Region"],
            "recommended_actions": ["monitor_weather", "secure_outdoor_items", "avoid_unnecessary_travel"],
            "source": "national_weather_service"
        },
        {
            "type": "wildfire",
            "severity": "high",
            "title": "Red Flag Fire Warning",
            "description": "Critical fire weather conditions. No outdoor burning permitted. High winds and low humidity create extreme fire danger.",
            "location": {"latitude": 34.0522, "longitude": -118.2437},  # Los Angeles
            "affected_areas": ["Los Angeles County", "Ventura County", "Orange County"],
            "recommended_actions": ["no_open_flames", "prepare_evacuation_plan", "monitor_fire_updates"],
            "source": "cal_fire"
        }
    ]
    
    for alert_data in sample_alerts:
        try:
            # Use the create_alert function to properly initialize
            import asyncio
            asyncio.create_task(create_alert(alert_data))
        except Exception:
            pass  # Ignore errors during initialization

# Initialize sample alerts when module loads
# _initialize_sample_alerts()  # Commented out to avoid async issues