"""
Offline/Edge Support Module for AgriCredAI
Provides fallback data, offline query capabilities, and edge-ready functionality
Implements data caching, offline forms, and SMS simulation for rural environments
"""

import sqlite3
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OfflineDataCache:
    """Represents cached data for offline use"""
    data_type: str
    data: Any
    timestamp: datetime
    expiry: datetime
    source: str
    confidence: float
    hash: str
    size_bytes: int

@dataclass
class OfflineQuery:
    """Represents an offline query that will be processed when online"""
    query_id: str
    timestamp: datetime
    query_type: str
    query_data: Dict[str, Any]
    farmer_id: Optional[str] = None
    contact_info: Optional[str] = None
    priority: str = "normal"  # low, normal, high, urgent
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class SMSMessage:
    """Represents an SMS message for offline communication"""
    message_id: str
    timestamp: datetime
    recipient: str
    message: str
    language: str
    priority: str
    status: str  # pending, sent, delivered, failed
    delivery_time: Optional[datetime] = None

class OfflineEdgeSupport:
    """Manages offline and edge functionality for rural environments"""
    
    def __init__(self, cache_dir: str = "offline_cache", db_path: str = "offline_data.db"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = db_path
        self.setup_offline_database()
        self.offline_queries = []
        self.sms_queue = []
        self.last_sync = None
        
    def setup_offline_database(self):
        """Setup offline database for storing cached data and offline queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Offline data cache table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS offline_cache (
            id INTEGER PRIMARY KEY,
            data_type TEXT NOT NULL,
            data_hash TEXT UNIQUE NOT NULL,
            data_blob BLOB NOT NULL,
            timestamp DATETIME NOT NULL,
            expiry DATETIME NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL,
            size_bytes INTEGER NOT NULL
        )
        """)
        
        # Offline queries table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS offline_queries (
            id INTEGER PRIMARY KEY,
            query_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME NOT NULL,
            query_type TEXT NOT NULL,
            query_data TEXT NOT NULL,
            farmer_id TEXT,
            contact_info TEXT,
            priority TEXT NOT NULL,
            status TEXT NOT NULL,
            result TEXT,
            error_message TEXT
        )
        """)
        
        # SMS queue table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sms_queue (
            id INTEGER PRIMARY KEY,
            message_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME NOT NULL,
            recipient TEXT NOT NULL,
            message TEXT NOT NULL,
            language TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL,
            delivery_time DATETIME
        )
        """)
        
        # Farmer offline profiles table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS farmer_offline_profiles (
            id INTEGER PRIMARY KEY,
            farmer_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            phone TEXT,
            village TEXT,
            district TEXT,
            state TEXT,
            land_size REAL,
            crop_type TEXT,
            last_updated DATETIME NOT NULL,
            profile_hash TEXT NOT NULL
        )
        """)
        
        conn.commit()
        conn.close()
    
    def cache_data(self, data_type: str, data: Any, source: str, 
                   confidence: float, ttl_hours: int = 24) -> str:
        """Cache data for offline use"""
        try:
            # Serialize data
            if isinstance(data, pd.DataFrame):
                data_blob = pickle.dumps(data)
            elif isinstance(data, dict):
                data_blob = json.dumps(data, default=str).encode('utf-8')
            else:
                data_blob = pickle.dumps(data)
            
            # Generate hash
            data_hash = hashlib.md5(data_blob).hexdigest()
            
            # Calculate expiry
            timestamp = datetime.now()
            expiry = timestamp + timedelta(hours=ttl_hours)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT OR REPLACE INTO offline_cache 
            (data_type, data_hash, data_blob, timestamp, expiry, source, confidence, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (data_type, data_hash, data_blob, timestamp, expiry, source, 
                  confidence, len(data_blob)))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cached {data_type} data from {source}, size: {len(data_blob)} bytes")
            return data_hash
            
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            return ""
    
    def get_cached_data(self, data_type: str, max_age_hours: int = 24) -> Optional[Any]:
        """Retrieve cached data if available and not expired"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get most recent valid data
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            cursor.execute("""
            SELECT data_blob, timestamp, source, confidence 
            FROM offline_cache 
            WHERE data_type = ? AND timestamp > ? AND expiry > ?
            ORDER BY timestamp DESC 
            LIMIT 1
            """, (data_type, cutoff_time, datetime.now()))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                data_blob, timestamp, source, confidence = result
                
                # Try to deserialize
                try:
                    if data_type in ['weather', 'market', 'soil']:
                        data = json.loads(data_blob.decode('utf-8'))
                    else:
                        data = pickle.loads(data_blob)
                    
                    logger.info(f"Retrieved cached {data_type} data from {source} (age: {datetime.now() - timestamp})")
                    return data
                    
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached {data_type} data: {e}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached data: {e}")
            return None
    
    def get_offline_weather_data(self, region: str) -> Optional[Dict[str, Any]]:
        """Get offline weather data for a region"""
        # Try to get from cache
        cached_data = self.get_cached_data(f"weather_{region}", max_age_hours=6)
        if cached_data:
            return cached_data
        
        # Fallback to static regional data
        static_weather = self._get_static_weather_data(region)
        if static_weather:
            # Cache the static data
            self.cache_data(f"weather_{region}", static_weather, "static_fallback", 0.6, ttl_hours=24)
            return static_weather
        
        return None
    
    def get_offline_market_data(self, commodity: str, region: str) -> Optional[Dict[str, Any]]:
        """Get offline market data for a commodity and region"""
        # Try to get from cache
        cached_data = self.get_cached_data(f"market_{commodity}_{region}", max_age_hours=12)
        if cached_data:
            return cached_data
        
        # Fallback to static market data
        static_market = self._get_static_market_data(commodity, region)
        if static_market:
            # Cache the static data
            self.cache_data(f"market_{commodity}_{region}", static_market, "static_fallback", 0.5, ttl_hours=24)
            return static_market
        
        return None
    
    def get_offline_soil_data(self, region: str) -> Optional[Dict[str, Any]]:
        """Get offline soil data for a region"""
        # Try to get from cache
        cached_data = self.get_cached_data(f"soil_{region}", max_age_hours=168)  # 1 week
        if cached_data:
            return cached_data
        
        # Fallback to static soil data
        static_soil = self._get_static_soil_data(region)
        if static_soil:
            # Cache the static data
            self.cache_data(f"soil_{region}", static_soil, "static_fallback", 0.8, ttl_hours=168)
            return static_soil
        
        return None
    
    def _get_static_weather_data(self, region: str) -> Optional[Dict[str, Any]]:
        """Get static weather data for offline use"""
        static_weather = {
            "Punjab": {
                "main": {"temp": 28, "humidity": 65, "pressure": 1013},
                "weather": [{"description": "clear sky", "main": "Clear"}],
                "wind": {"speed": 5},
                "seasonal_pattern": "kharif",
                "rainfall_outlook": "moderate",
                "risk_factors": ["heat stress", "water scarcity"]
            },
            "Maharashtra": {
                "main": {"temp": 32, "humidity": 70, "pressure": 1010},
                "weather": [{"description": "partly cloudy", "main": "Clouds"}],
                "wind": {"speed": 8},
                "seasonal_pattern": "kharif",
                "rainfall_outlook": "good",
                "risk_factors": ["monsoon variability", "flood risk"]
            },
            "Uttar Pradesh": {
                "main": {"temp": 30, "humidity": 68, "pressure": 1012},
                "weather": [{"description": "clear sky", "main": "Clear"}],
                "wind": {"speed": 6},
                "seasonal_pattern": "rabi",
                "rainfall_outlook": "moderate",
                "risk_factors": ["frost risk", "cold stress"]
            },
            "Karnataka": {
                "main": {"temp": 29, "humidity": 72, "pressure": 1011},
                "weather": [{"description": "scattered clouds", "main": "Clouds"}],
                "wind": {"speed": 7},
                "seasonal_pattern": "kharif",
                "rainfall_outlook": "good",
                "risk_factors": ["drought risk", "irrigation dependency"]
            }
        }
        
        return static_weather.get(region, static_weather.get("Punjab"))
    
    def _get_static_market_data(self, commodity: str, region: str) -> Optional[Dict[str, Any]]:
        """Get static market data for offline use"""
        static_markets = {
            "wheat": {
                "Punjab": {"price": 2200, "trend": "stable", "demand": "high"},
                "Maharashtra": {"price": 2100, "trend": "stable", "demand": "medium"},
                "Uttar Pradesh": {"price": 2150, "trend": "stable", "demand": "high"},
                "Karnataka": {"price": 2080, "trend": "stable", "demand": "medium"}
            },
            "rice": {
                "Punjab": {"price": 2500, "trend": "stable", "demand": "medium"},
                "Maharashtra": {"price": 2400, "trend": "stable", "demand": "high"},
                "Uttar Pradesh": {"price": 2450, "trend": "stable", "demand": "high"},
                "Karnataka": {"price": 2380, "trend": "stable", "demand": "medium"}
            },
            "cotton": {
                "Punjab": {"price": 5000, "trend": "increasing", "demand": "high"},
                "Maharashtra": {"price": 4800, "trend": "increasing", "demand": "very_high"},
                "Uttar Pradesh": {"price": 4900, "trend": "increasing", "demand": "high"},
                "Karnataka": {"price": 4750, "trend": "increasing", "demand": "high"}
            }
        }
        
        commodity_data = static_markets.get(commodity.lower(), {})
        return commodity_data.get(region, commodity_data.get("Punjab"))
    
    def _get_static_soil_data(self, region: str) -> Optional[Dict[str, Any]]:
        """Get static soil data for offline use"""
        static_soils = {
            "Punjab": {
                "soil_type": "alluvial",
                "ph": 7.0,
                "organic_carbon": 0.7,
                "nitrogen": 280,
                "phosphorus": 25,
                "potassium": 250,
                "recommendations": ["balanced fertilization", "organic matter addition"]
            },
            "Maharashtra": {
                "soil_type": "black",
                "ph": 7.5,
                "organic_carbon": 0.6,
                "nitrogen": 240,
                "phosphorus": 20,
                "potassium": 300,
                "recommendations": ["phosphorus application", "organic farming"]
            },
            "Uttar Pradesh": {
                "soil_type": "alluvial",
                "ph": 6.8,
                "organic_carbon": 0.5,
                "nitrogen": 200,
                "phosphorus": 18,
                "potassium": 180,
                "recommendations": ["nitrogen management", "soil conservation"]
            },
            "Karnataka": {
                "soil_type": "red",
                "ph": 6.5,
                "organic_carbon": 0.5,
                "nitrogen": 180,
                "phosphorus": 15,
                "potassium": 150,
                "recommendations": ["micronutrient application", "organic matter"]
            }
        }
        
        return static_soils.get(region, static_soils.get("Punjab"))
    
    def create_offline_query(self, query_type: str, query_data: Dict[str, Any], 
                            farmer_id: Optional[str] = None, contact_info: Optional[str] = None,
                            priority: str = "normal") -> str:
        """Create an offline query for later processing"""
        query_id = hashlib.md5(f"{datetime.now().isoformat()}{query_type}".encode()).hexdigest()[:8]
        
        offline_query = OfflineQuery(
            query_id=query_id,
            timestamp=datetime.now(),
            query_type=query_type,
            query_data=query_data,
            farmer_id=farmer_id,
            contact_info=contact_info,
            priority=priority
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO offline_queries 
        (query_id, timestamp, query_type, query_data, farmer_id, contact_info, priority, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (query_id, offline_query.timestamp, query_type, 
              json.dumps(query_data), farmer_id, contact_info, priority, "pending"))
        
        conn.commit()
        conn.close()
        
        # Add to memory
        self.offline_queries.append(offline_query)
        
        logger.info(f"Created offline query {query_id} of type {query_type}")
        return query_id
    
    def get_pending_offline_queries(self) -> List[OfflineQuery]:
        """Get all pending offline queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT query_id, timestamp, query_type, query_data, farmer_id, contact_info, priority, status
        FROM offline_queries 
        WHERE status = 'pending'
        ORDER BY 
            CASE priority 
                WHEN 'urgent' THEN 1 
                WHEN 'high' THEN 2 
                WHEN 'normal' THEN 3 
                WHEN 'low' THEN 4 
            END,
            timestamp ASC
        """)
        
        queries = []
        for row in cursor.fetchall():
            query = OfflineQuery(
                query_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                query_type=row[2],
                query_data=json.loads(row[3]),
                farmer_id=row[4],
                contact_info=row[5],
                priority=row[6],
                status=row[7]
            )
            queries.append(query)
        
        conn.close()
        return queries
    
    def process_offline_query(self, query: OfflineQuery) -> Dict[str, Any]:
        """Process an offline query using available offline data"""
        try:
            if query.query_type == "weather_inquiry":
                return self._process_weather_inquiry(query)
            elif query.query_type == "market_inquiry":
                return self._process_market_inquiry(query)
            elif query.query_type == "soil_inquiry":
                return self._process_soil_inquiry(query)
            elif query.query_type == "credit_inquiry":
                return self._process_credit_inquiry(query)
            else:
                return {"error": f"Unknown query type: {query.query_type}"}
                
        except Exception as e:
            logger.error(f"Failed to process offline query {query.query_id}: {e}")
            return {"error": str(e)}
    
    def _process_weather_inquiry(self, query: OfflineQuery) -> Dict[str, Any]:
        """Process weather inquiry using offline data"""
        location = query.query_data.get('location', 'Punjab')
        crop_type = query.query_data.get('crop_type', 'wheat')
        
        weather_data = self.get_offline_weather_data(location)
        if not weather_data:
            return {"error": "No weather data available for this location"}
        
        # Basic weather analysis
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        
        # Simple crop impact assessment
        if crop_type.lower() == 'wheat':
            if temp < 15:
                impact = "Cold stress - consider frost protection"
            elif temp > 30:
                impact = "Heat stress - ensure adequate irrigation"
            else:
                impact = "Optimal conditions for wheat growth"
        else:
            impact = "Weather conditions appear favorable"
        
        return {
            "query_type": "weather_inquiry",
            "location": location,
            "crop_type": crop_type,
            "temperature": temp,
            "humidity": humidity,
            "weather_condition": weather_data['weather'][0]['description'],
            "crop_impact": impact,
            "risk_factors": weather_data.get('risk_factors', []),
            "data_source": "offline_cache",
            "confidence": 0.7,
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_market_inquiry(self, query: OfflineQuery) -> Dict[str, Any]:
        """Process market inquiry using offline data"""
        commodity = query.query_data.get('commodity', 'wheat')
        location = query.query_data.get('location', 'Punjab')
        
        market_data = self.get_offline_market_data(commodity, location)
        if not market_data:
            return {"error": f"No market data available for {commodity} in {location}"}
        
        return {
            "query_type": "market_inquiry",
            "commodity": commodity,
            "location": location,
            "current_price": market_data['price'],
            "trend": market_data['trend'],
            "demand": market_data['demand'],
            "recommendation": self._generate_market_recommendation(market_data),
            "data_source": "offline_cache",
            "confidence": 0.6,
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_soil_inquiry(self, query: OfflineQuery) -> Dict[str, Any]:
        """Process soil inquiry using offline data"""
        location = query.query_data.get('location', 'Punjab')
        crop_type = query.query_data.get('crop_type', 'wheat')
        
        soil_data = self.get_offline_soil_data(location)
        if not soil_data:
            return {"error": "No soil data available for this location"}
        
        # Generate soil recommendations
        recommendations = soil_data['recommendations'].copy()
        
        # Add crop-specific recommendations
        if crop_type.lower() == 'wheat':
            if soil_data['ph'] < 6.5:
                recommendations.append("Apply lime to raise soil pH")
            if soil_data['nitrogen'] < 200:
                recommendations.append("Increase nitrogen application")
        
        return {
            "query_type": "soil_inquiry",
            "location": location,
            "crop_type": crop_type,
            "soil_type": soil_data['soil_type'],
            "ph": soil_data['ph'],
            "nutrients": {
                "nitrogen": soil_data['nitrogen'],
                "phosphorus": soil_data['phosphorus'],
                "potassium": soil_data['potassium']
            },
            "organic_carbon": soil_data['organic_carbon'],
            "recommendations": recommendations,
            "data_source": "offline_cache",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_credit_inquiry(self, query: OfflineQuery) -> Dict[str, Any]:
        """Process credit inquiry using offline data"""
        farmer_profile = query.query_data.get('farmer_profile', {})
        
        # Basic offline credit assessment
        risk_score = 0.5  # Default moderate risk
        
        # Adjust based on available profile data
        if farmer_profile.get('payment_history_score', 0) > 0.8:
            risk_score -= 0.2
        if farmer_profile.get('debt_to_income_ratio', 0) > 0.5:
            risk_score += 0.2
        if farmer_profile.get('land_size', 0) > 5:
            risk_score -= 0.1
        
        # Determine recommendation
        if risk_score < 0.4:
            recommendation = "APPROVE"
            confidence = "high"
        elif risk_score < 0.7:
            recommendation = "REVIEW"
            confidence = "medium"
        else:
            recommendation = "REJECT"
            confidence = "high"
        
        return {
            "query_type": "credit_inquiry",
            "farmer_id": query.farmer_id,
            "risk_score": risk_score,
            "recommendation": recommendation,
            "confidence": confidence,
            "key_factors": list(farmer_profile.keys()),
            "limitations": ["Limited profile data available", "Offline assessment"],
            "data_source": "offline_profile",
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_market_recommendation(self, market_data: Dict[str, Any]) -> str:
        """Generate market recommendation based on data"""
        if market_data['trend'] == 'increasing':
            if market_data['demand'] == 'very_high':
                return "Excellent time to sell - high demand and rising prices"
            elif market_data['demand'] == 'high':
                return "Good time to sell - prices are increasing"
            else:
                return "Consider selling - prices trending up"
        elif market_data['trend'] == 'stable':
            if market_data['demand'] == 'high':
                return "Stable market - good time to sell"
            else:
                return "Market stable - monitor for changes"
        else:
            return "Prices declining - consider holding or diversifying"
    
    def create_sms_message(self, recipient: str, message: str, language: str = 'en',
                          priority: str = 'normal') -> str:
        """Create an SMS message for offline communication"""
        message_id = hashlib.md5(f"{datetime.now().isoformat()}{recipient}".encode()).hexdigest()[:8]
        
        sms = SMSMessage(
            message_id=message_id,
            timestamp=datetime.now(),
            recipient=recipient,
            message=message,
            language=language,
            priority=priority,
            status="pending"
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO sms_queue 
        (message_id, timestamp, recipient, message, language, priority, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (message_id, sms.timestamp, recipient, message, language, priority, "pending"))
        
        conn.commit()
        conn.close()
        
        # Add to memory
        self.sms_queue.append(sms)
        
        logger.info(f"Created SMS message {message_id} to {recipient}")
        return message_id
    
    def get_pending_sms_messages(self) -> List[SMSMessage]:
        """Get all pending SMS messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT message_id, timestamp, recipient, message, language, priority, status
        FROM sms_queue 
        WHERE status = 'pending'
        ORDER BY 
            CASE priority 
                WHEN 'urgent' THEN 1 
                WHEN 'high' THEN 2 
                WHEN 'normal' THEN 3 
                WHEN 'low' THEN 4 
            END,
            timestamp ASC
        """)
        
        messages = []
        for row in cursor.fetchall():
            sms = SMSMessage(
                message_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                recipient=row[2],
                message=row[3],
                language=row[4],
                priority=row[5],
                status=row[6]
            )
            messages.append(sms)
        
        conn.close()
        return messages
    
    def simulate_sms_delivery(self, message_id: str) -> bool:
        """Simulate SMS delivery (for demo purposes)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update status to delivered
            cursor.execute("""
            UPDATE sms_queue 
            SET status = 'delivered', delivery_time = ?
            WHERE message_id = ?
            """, (datetime.now(), message_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"SMS {message_id} delivered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to simulate SMS delivery: {e}")
            return False
    
    def export_offline_data_summary(self) -> Dict[str, Any]:
        """Export summary of offline data and capabilities"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count cached data
        cursor.execute("SELECT COUNT(*) FROM offline_cache")
        cache_count = cursor.fetchone()[0]
        
        # Count pending queries
        cursor.execute("SELECT COUNT(*) FROM offline_queries WHERE status = 'pending'")
        pending_queries = cursor.fetchone()[0]
        
        # Count pending SMS
        cursor.execute("SELECT COUNT(*) FROM sms_queue WHERE status = 'pending'")
        pending_sms = cursor.fetchone()[0]
        
        # Get cache size
        cursor.execute("SELECT SUM(size_bytes) FROM offline_cache")
        total_size = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "offline_capabilities": {
                "weather_data": "Available for major regions",
                "market_data": "Available for key commodities",
                "soil_data": "Available for major regions",
                "credit_assessment": "Basic offline assessment available"
            },
            "cache_statistics": {
                "total_items": cache_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "last_updated": self.last_sync.isoformat() if self.last_sync else None
            },
            "pending_items": {
                "queries": pending_queries,
                "sms_messages": pending_sms
            },
            "supported_regions": ["Punjab", "Maharashtra", "Uttar Pradesh", "Karnataka"],
            "supported_commodities": ["wheat", "rice", "cotton", "sugarcane", "soybean"],
            "data_freshness": {
                "weather": "6 hours",
                "market": "12 hours", 
                "soil": "1 week"
            }
        }
    
    def clear_expired_cache(self) -> int:
        """Clear expired cache entries and return count of cleared items"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count expired items
            cursor.execute("SELECT COUNT(*) FROM offline_cache WHERE expiry < ?", (datetime.now(),))
            expired_count = cursor.fetchone()[0]
            
            # Delete expired items
            cursor.execute("DELETE FROM offline_cache WHERE expiry < ?", (datetime.now(),))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared {expired_count} expired cache entries")
            return expired_count
            
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
            return 0

# Global instance
offline_support = OfflineEdgeSupport()
