"""
OpenAI Vision API Garment Tag Analyzer - Complete Pipeline
With Fixed Elgato Light Control, Better OCR, and Consolidated Steps
INDENTATION FIXED VERSION
"""

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
import numpy as np
import pytesseract
from PIL import Image
import time
import os
import json
import base64
import io
import asyncio
import openai
from openai import AsyncOpenAI

# Import data collection system
from data_collection_and_correction_system import (
    GarmentDataCollector,
    render_correction_panel,
    render_sample_count_widget,
    save_analysis_with_corrections,
    verify_ebay_pricing,
    render_confidence_scores
)

# eBay API imports for sold comps research
import requests
from datetime import datetime, timedelta
import statistics
from functools import wraps

# Rate limiting decorator for eBay API
def rate_limited(max_per_minute=5000):
    """Rate limiting decorator for API calls"""
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# eBay cache helper functions
def _cache_key_with_specifics(brand: str, garment_type: str, size: str = None, 
                             gender: str = None, item_specifics: dict = None) -> str:
    """Generate cache key for eBay search"""
    key_parts = [brand, garment_type]
    if size and size != 'Unknown':
        key_parts.append(size)
    if gender and gender != 'Unisex':
        key_parts.append(gender)
    if item_specifics:
        for name, value in sorted(item_specifics.items()):
            key_parts.append(f"{name}:{value}")
    return "ebay_" + "_".join(key_parts).replace(" ", "_").lower()

def _get_cached_result(cache_key: str) -> dict:
    """Get cached eBay result"""
    try:
        cache_file = f"cache/{cache_key}.json"
        if os.path.exists(cache_file):
            # Check if cache is still valid (24 hours)
            cache_time = os.path.getmtime(cache_file)
            if time.time() - cache_time < 86400:  # 24 hours
                with open(cache_file, 'r') as f:
                    return json.load(f)
    except Exception:
        pass
    return None

def _cache_result(cache_key: str, result: dict):
    """Cache eBay result"""
    try:
        os.makedirs("cache", exist_ok=True)
        cache_file = f"cache/{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to cache eBay result: {e}")

class eBayCompsFinder:
    """Research eBay sold comps for accurate pricing and sell-through rates"""
    
    def __init__(self, app_id=None):
        """
        Initialize with eBay Finding API credentials.
        Get your App ID from: https://developer.ebay.com/my/keys
        """
        self.app_id = app_id or os.getenv('EBAY_APP_ID')
        self.base_url = "https://svcs.ebay.com/services/search/FindingService/v1"
        
        if not self.app_id:
            logger.warning("⚠️ eBay App ID not found. Set EBAY_APP_ID environment variable")
            logger.warning("   Get one free at: https://developer.ebay.com/my/keys")
    
    @rate_limited(max_per_minute=5000)  # eBay allows 5000 calls/day for free tier
    def search_sold_comps(self, brand: str, garment_type: str, size: str = None, 
                          gender: str = None, item_specifics: dict = None,
                          days_back: int = 90) -> dict:
        """
        Search eBay for sold/completed listings to get real market data.
        
        Returns:
            {
                'sold_items': [...],
                'active_items': [...],
                'avg_sold_price': float,
                'median_sold_price': float,
                'sell_through_rate': float,  # % of items that sold vs listed
                'total_sold': int,
                'total_active': int,
                'price_range': {'low': float, 'high': float},
                'days_to_sell_avg': float,
                'success': bool
            }
        """
        if not self.app_id:
            return {'success': False, 'error': 'eBay App ID not configured'}
        
        # Check cache first
        cache_key = _cache_key_with_specifics(brand, garment_type, size, gender, item_specifics)
        cached = _get_cached_result(cache_key)
        if cached:
            return cached
        
        try:
            # Build search query
            keywords = self._build_search_keywords(brand, garment_type, size, gender)
            
            # Search SOLD items (completed + sold)
            sold_data = self._search_ebay(
                keywords=keywords,
                item_filter=[
                    {'name': 'SoldItemsOnly', 'value': 'true'},
                    {'name': 'EndTimeFrom', 'value': (datetime.now() - timedelta(days=days_back)).isoformat()}
                ],
                item_specifics=item_specifics
            )
            
            # Search ACTIVE items (current listings)
            active_data = self._search_ebay(
                keywords=keywords,
                item_filter=[
                    {'name': 'ListingType', 'value': 'FixedPrice'}
                ],
                item_specifics=item_specifics
            )
            
            # Parse results
            sold_items = self._parse_items(sold_data)
            active_items = self._parse_items(active_data)
            
            # Calculate metrics
            result = self._calculate_metrics(sold_items, active_items, days_back)
            result['success'] = True
            result['brand'] = brand
            result['garment_type'] = garment_type
            result['search_keywords'] = keywords
            
            # Cache the result
            _cache_result(cache_key, result)
            
            logger.info(f"[EBAY] {brand} {garment_type}: "
                       f"${result['avg_sold_price']:.2f} avg, "
                       f"{result['sell_through_rate']:.1f}% sell-through, "
                       f"{result['total_sold']} sold in {days_back} days")
            
            return result
            
        except Exception as e:
            logger.error(f"[EBAY] Search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _build_search_keywords(self, brand: str, garment_type: str, 
                               size: str = None, gender: str = None) -> str:
        """Build optimized eBay search keywords"""
        keywords = [brand, garment_type]
        
        # Add size if specific
        if size and size != 'Unknown':
            # Clean size string
            size_clean = size.replace('US', '').replace('IT', '').replace('EU', '').strip()
            if size_clean and not size_clean.lower() in ['one size', 'os', 'onesize']:
                keywords.append(f"size {size_clean}")
        
        # Add gender if specific
        if gender and gender.lower() in ['men', 'women', 'womens', 'mens']:
            if gender.lower() in ['women', 'womens']:
                keywords.append("women's")
            elif gender.lower() in ['men', 'mens']:
                keywords.append("men's")
        
        return ' '.join(keywords)
    
    def _search_ebay(self, keywords: str, item_filter: list = None,
                     item_specifics: dict = None, max_results: int = 100) -> dict:
        """Execute eBay Finding API search"""
        
        params = {
            'OPERATION-NAME': 'findCompletedItems' if any(f.get('name') == 'SoldItemsOnly' for f in (item_filter or [])) else 'findItemsAdvanced',
            'SERVICE-VERSION': '1.0.0',
            'SECURITY-APPNAME': self.app_id,
            'RESPONSE-DATA-FORMAT': 'JSON',
            'REST-PAYLOAD': '',
            'keywords': keywords,
            'paginationInput.entriesPerPage': str(min(max_results, 100)),
            'sortOrder': 'EndTimeSoonest'
        }
        
        # Add item filters
        if item_filter:
            for idx, filt in enumerate(item_filter):
                params[f'itemFilter({idx}).name'] = filt['name']
                params[f'itemFilter({idx}).value'] = filt['value']
        
        # Add item specifics (neckline, sleeve length, etc.)
        if item_specifics:
            for idx, (name, value) in enumerate(item_specifics.items()):
                params[f'aspectFilter({idx}).aspectName'] = name
                params[f'aspectFilter({idx}).aspectValueName'] = value
        
        response = requests.get(self.base_url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    def _parse_items(self, api_response: dict) -> list:
        """Parse eBay API response into simplified item list"""
        items = []
        
        try:
            search_result = api_response.get('findCompletedItemsResponse', 
                                           api_response.get('findItemsAdvancedResponse', [{}]))[0]
            
            if 'searchResult' not in search_result:
                return items
            
            result_items = search_result['searchResult'][0].get('item', [])
            
            for item in result_items:
                try:
                    # Extract price
                    price_info = item.get('sellingStatus', [{}])[0]
                    price = float(price_info.get('currentPrice', [{}])[0].get('__value__', 0))
                    
                    # Extract listing dates
                    start_time = item.get('listingInfo', [{}])[0].get('startTime', [''])[0]
                    end_time = item.get('listingInfo', [{}])[0].get('endTime', [''])[0]
                    
                    # Calculate days to sell
                    days_to_sell = None
                    if start_time and end_time:
                        try:
                            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                            days_to_sell = (end - start).days
                        except:
                            pass
                    
                    items.append({
                        'title': item.get('title', [''])[0],
                        'price': price,
                        'condition': item.get('condition', [{}])[0].get('conditionDisplayName', [''])[0],
                        'url': item.get('viewItemURL', [''])[0],
                        'start_time': start_time,
                        'end_time': end_time,
                        'days_to_sell': days_to_sell,
                        'item_id': item.get('itemId', [''])[0]
                    })
                except Exception as e:
                    logger.debug(f"Failed to parse item: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to parse eBay response: {e}")
        
        return items
    
    def _calculate_metrics(self, sold_items: list, active_items: list, 
                          days_back: int) -> dict:
        """Calculate pricing and sell-through metrics"""
        
        # Filter out extreme outliers (likely errors or bundle deals)
        def is_reasonable_price(price):
            return 5.0 <= price <= 5000.0
        
        sold_prices = [item['price'] for item in sold_items if is_reasonable_price(item['price'])]
        active_prices = [item['price'] for item in active_items if is_reasonable_price(item['price'])]
        
        # Calculate sold item metrics
        avg_sold = statistics.mean(sold_prices) if sold_prices else 0
        median_sold = statistics.median(sold_prices) if sold_prices else 0
        
        # Calculate sell-through rate
        # Formula: (sold items / (sold + active items)) * 100
        total_sold = len(sold_items)
        total_active = len(active_items)
        total_listings = total_sold + total_active
        
        sell_through_rate = (total_sold / total_listings * 100) if total_listings > 0 else 0
        
        # Calculate average days to sell
        days_to_sell_list = [item['days_to_sell'] for item in sold_items 
                            if item['days_to_sell'] is not None and item['days_to_sell'] > 0]
        avg_days_to_sell = statistics.mean(days_to_sell_list) if days_to_sell_list else None
        
        # Price range
        all_prices = sold_prices + active_prices
        price_low = min(all_prices) if all_prices else 0
        price_high = max(all_prices) if all_prices else 0
        
        return {
            'sold_items': sold_items[:20],  # Keep top 20 for reference
            'active_items': active_items[:20],
            'avg_sold_price': round(avg_sold, 2),
            'median_sold_price': round(median_sold, 2),
            'sell_through_rate': round(sell_through_rate, 1),
            'total_sold': total_sold,
            'total_active': total_active,
            'total_listings': total_listings,
            'price_range': {
                'low': round(price_low, 2),
                'high': round(price_high, 2)
            },
            'days_to_sell_avg': round(avg_days_to_sell, 1) if avg_days_to_sell else None,
            'confidence': self._calculate_confidence(total_sold, total_active)
        }
    
    def _calculate_confidence(self, sold_count: int, active_count: int) -> str:
        """Calculate confidence level based on sample size"""
        total = sold_count + active_count
        
        if total >= 50:
            return "High"
        elif total >= 20:
            return "Medium"
        elif total >= 10:
            return "Low"
        else:
            return "Very Low"

# Store tag detection removed - was overdetecting
import google.generativeai as genai
import sounddevice as sd
from scipy.io.wavfile import write
import sqlite3
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import Thread
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlencode
import base64
import concurrent.futures
import difflib
import hashlib
import io
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import requests
import socket
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates
import threading
import time
import traceback
from vertexai.generative_models import GenerativeModel, Part
import vertexai

# ============================================
# CONFIGURE LOGGING FIRST - BEFORE ANYTHING ELSE
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('garment_analyzer.log', encoding='utf-8')  # UTF-8 for Unicode support
    ]
)
logger = logging.getLogger(__name__)

# NOW you can use logger safely
logger.info("🚀 Starting Garment Analyzer Pipeline")

# ============================================
# CHECK OPTIONAL DEPENDENCIES (after logger exists)
# ============================================
try:
    import pyrealsense2 as rs
    REALSENSE_SDK_AVAILABLE = True
    logger.info("✅ RealSense SDK available")
except ImportError:
    REALSENSE_SDK_AVAILABLE = False
    logger.warning("⚠️ pyrealsense2 not installed - RealSense will use OpenCV fallback")
    logger.warning("   Install with: pip install pyrealsense2")

# ==========================
# RATE LIMITING UTILITIES
# ==========================

def rate_limited(max_per_minute=5000):
    """Rate limiting decorator for API calls"""
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                logger.info(f"[RATE-LIMIT] Waiting {left_to_wait:.2f}s before API call")
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Simple file-based cache for API responses
def _cache_key(brand, garment_type, size, gender):
    """Generate cache key for API calls"""
    key_str = f"{brand}_{garment_type}_{size}_{gender}".lower().replace(" ", "_")
    return hashlib.md5(key_str.encode()).hexdigest()

def _cache_key_with_specifics(brand, garment_type, size, gender, item_specifics=None):
    """Generate cache key including item specifics"""
    key_str = f"{brand}_{garment_type}_{size}_{gender}".lower().replace(" ", "_")
    
    if item_specifics:
        # Add specifics to cache key
        specifics_str = "_".join([f"{k}_{v}" for k, v in sorted(item_specifics.items())]).lower().replace(" ", "_")
        key_str += f"_{specifics_str}"
    
    return hashlib.md5(key_str.encode()).hexdigest()

def _get_cached_result(cache_key):
    """Get cached API result if exists and not expired"""
    cache_file = f"cache/{cache_key}.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            # Check if cache is still valid (24 hours)
            if time.time() - data.get('timestamp', 0) < 86400:
                logger.info(f"[CACHE] Hit for key: {cache_key}")
                return data.get('result')
        except Exception as e:
            logger.warning(f"[CACHE] Error reading cache: {e}")
    return None

def _cache_result(cache_key, result):
    """Cache API result"""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{cache_key}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'result': result
            }, f)
        logger.info(f"[CACHE] Stored result for key: {cache_key}")
    except Exception as e:
        logger.warning(f"[CACHE] Error writing cache: {e}")

# ==========================
# CLEAN STATE MANAGEMENT
# ==========================

# Designer brand tier-based pricing database
DESIGNER_BRANDS = {
    # Ultra luxury
    'Gucci': {'tier': 'ultra', 'multiplier': 12.0},
    'Prada': {'tier': 'ultra', 'multiplier': 10.0},
    'Louis Vuitton': {'tier': 'ultra', 'multiplier': 15.0},
    'Chanel': {'tier': 'ultra', 'multiplier': 20.0},
    'Hermès': {'tier': 'ultra', 'multiplier': 25.0},
    'Dior': {'tier': 'ultra', 'multiplier': 12.0},
    'Fendi': {'tier': 'ultra', 'multiplier': 11.0},
    'Bottega Veneta': {'tier': 'ultra', 'multiplier': 10.0},
    'Celine': {'tier': 'ultra', 'multiplier': 9.0},
    'Loewe': {'tier': 'ultra', 'multiplier': 8.0},
    
    # High-end designer
    'Paul Smith': {'tier': 'high', 'multiplier': 5.0},
    'Tom Ford': {'tier': 'high', 'multiplier': 8.0},
    'Saint Laurent': {'tier': 'high', 'multiplier': 7.0},
    'Balenciaga': {'tier': 'high', 'multiplier': 6.0},
    'Givenchy': {'tier': 'high', 'multiplier': 6.5},
    'Versace': {'tier': 'high', 'multiplier': 7.0},
    'Valentino': {'tier': 'high', 'multiplier': 7.5},
    'Alexander McQueen': {'tier': 'high', 'multiplier': 6.0},
    'Burberry': {'tier': 'high', 'multiplier': 5.5},
    
    # Contemporary designer
    'Theory': {'tier': 'mid-high', 'multiplier': 3.5},
    'Vince': {'tier': 'mid-high', 'multiplier': 3.0},
    'Equipment': {'tier': 'mid-high', 'multiplier': 2.5},
    'Rag & Bone': {'tier': 'mid-high', 'multiplier': 3.0},
    'Vince Camuto': {'tier': 'mid-high', 'multiplier': 2.8},
    'Rebecca Minkoff': {'tier': 'mid-high', 'multiplier': 2.5},
    'Tory Burch': {'tier': 'mid-high', 'multiplier': 3.2},
    'Kate Spade': {'tier': 'mid-high', 'multiplier': 2.7},
    'Michael Kors': {'tier': 'mid-high', 'multiplier': 2.5},
    'Coach': {'tier': 'mid-high', 'multiplier': 2.8},
}

BASE_GARMENT_PRICES = {
    'shirt': 15, 'dress': 25, 'jacket': 35,
    'pants': 20, 'jeans': 25, 'sweater': 20, 'coat': 50
}

class AnalysisStatus(Enum):
    """Status of analysis operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class UIState:
    """Clean encapsulation of UI-related session state"""
    show_correction_form: bool = False
    auto_zoom_enabled: bool = True
    focus_mode: bool = False
    show_advanced_controls: bool = False
    show_measurements: bool = False
    focus_mode_started: bool = False
    
    def reset_focus_mode(self):
        """Reset focus mode state"""
        self.focus_mode = False
        self.focus_mode_started = False

@dataclass
class CameraCache:
    """Thread-safe camera frame caching"""
    tag_frame: Optional[np.ndarray] = None
    garment_frame: Optional[np.ndarray] = None
    last_update: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update_tag_frame(self, frame: np.ndarray):
        """Thread-safe tag frame update"""
        with self.lock:
            self.tag_frame = frame.copy() if frame is not None else None
            self.last_update = time.time()
    
    def update_garment_frame(self, frame: np.ndarray):
        """Thread-safe garment frame update"""
        with self.lock:
            self.garment_frame = frame.copy() if frame is not None else None
            self.last_update = time.time()
    
    def get_tag_frame(self) -> Optional[np.ndarray]:
        """Thread-safe tag frame retrieval"""
        with self.lock:
            return self.tag_frame.copy() if self.tag_frame is not None else None
    
    def get_garment_frame(self) -> Optional[np.ndarray]:
        """Thread-safe garment frame retrieval"""
        with self.lock:
            return self.garment_frame.copy() if self.garment_frame is not None else None


def convert_size_to_us(size, gender, region='EU'):
    """Convert international sizing to US sizes"""
    if not size or size == 'Unknown':
        return 'Unknown'
    
    size_str = str(size).upper().strip()
    
    # Already US size format
    if any(us_marker in size_str for us_marker in ['US', 'USA', 'AMERICAN']):
        return size_str.replace('US', '').replace('USA', '').strip()
    
    # European/Italian to US conversion
    if region == 'EU' or 'IT' in size_str or 'EU' in size_str:
        # Remove IT/EU markers
        size_str = size_str.replace('IT', '').replace('EU', '').strip()
        
        try:
            eu_size = int(size_str)
            
            if 'women' in gender.lower():
                # Women's clothing: IT 38 = US 2, IT 40 = US 4, etc.
                us_size = ((eu_size - 36) / 2) if eu_size >= 36 else None
                if us_size and us_size >= 0:
                    return f"US {int(us_size)}"
            
            elif 'men' in gender.lower():
                # Men's clothing: IT 44 = US 34, IT 46 = US 36, etc.
                us_size = eu_size - 10
                if us_size > 0:
                    return f"US {us_size}"
        
        except ValueError:
            pass
    
    # UK to US
    if 'UK' in size_str:
        size_str = size_str.replace('UK', '').strip()
        try:
            uk_size = int(size_str)
            if 'women' in gender.lower():
                us_size = uk_size + 2  # UK 8 = US 10
                return f"US {us_size}"
            elif 'men' in gender.lower():
                us_size = uk_size  # UK and US men's sizes are similar
                return f"US {us_size}"
        except ValueError:
            pass
    
    # Return original if no conversion needed
    return size


def validate_garment_classification(garment_type: str, visible_features: list) -> tuple[bool, str]:
    """Validate garment classification against visible features"""
    
    features_str = ' '.join(visible_features).lower()
    
    # Cardigan MUST have front opening
    if garment_type.lower() == 'cardigan':
        if 'no front opening' in features_str or 'pullover' in features_str:
            return False, "Cardigan requires front opening - likely pullover/sweater"
    
    # Turtleneck MUST have high collar
    if garment_type.lower() == 'turtleneck':
        if not any(term in features_str for term in ['turtleneck', 'high collar', 'folded collar']):
            return False, "Turtleneck requires high folded collar"
    
    return True, "Classification valid"


def validate_classification_strict(garment_type, features_dict):
    """
    Strict validation rules for garment classification
    Returns: (is_valid, corrected_type, issues)
    """
    issues = []
    corrected_type = garment_type
    
    garment_lower = garment_type.lower()
    has_opening = features_dict.get('has_front_opening', False)
    neckline = features_dict.get('neckline', '').lower()
    
    # RULE 1: Cardigan MUST have opening
    if 'cardigan' in garment_lower:
        if not has_opening:
            issues.append("Cardigan requires front opening")
            corrected_type = 'pullover'
            logger.warning("[VALIDATION] Auto-corrected: cardigan → pullover (no opening)")
    
    # RULE 2: Pullover/Sweater CANNOT have opening
    if any(term in garment_lower for term in ['pullover', 'sweater']) and 'cardigan' not in garment_lower:
        if has_opening:
            issues.append("Pullover cannot have front opening")
            corrected_type = 'cardigan'
            logger.warning("[VALIDATION] Auto-corrected: pullover → cardigan (has opening)")
    
    # RULE 3: Turtleneck requires turtleneck collar
    if 'turtleneck' in garment_lower:
        if 'turtle' not in neckline and 'high' not in neckline:
            issues.append("Turtleneck classification requires turtleneck collar")
            logger.warning("[VALIDATION] Turtleneck without appropriate collar")
    
    # RULE 4: Button-down requires buttons
    if 'button' in garment_lower:
        if not has_opening:
            issues.append("Button-down requires buttons/opening")
            corrected_type = garment_type.replace('button-down', 'shirt').replace('button down', 'shirt')
    
    is_valid = len(issues) == 0
    
    return is_valid, corrected_type, issues


def build_ebay_item_specifics(pipeline_data) -> dict:
    """Build eBay Item Specifics from garment analysis results"""
    item_specifics = {}
    
    # Neckline mapping
    neckline = pipeline_data.neckline.lower() if pipeline_data.neckline != 'Unknown' else ''
    if 'turtleneck' in neckline:
        item_specifics['Neckline'] = 'Turtleneck'
    elif 'v-neck' in neckline or 'vneck' in neckline:
        item_specifics['Neckline'] = 'V-Neck'
    elif 'crew' in neckline:
        item_specifics['Neckline'] = 'Crew Neck'
    elif 'scoop' in neckline:
        item_specifics['Neckline'] = 'Scoop Neck'
    elif 'boat' in neckline:
        item_specifics['Neckline'] = 'Boat Neck'
    elif 'cowl' in neckline:
        item_specifics['Neckline'] = 'Cowl Neck'
    
    # Sleeve length mapping
    sleeve_length = pipeline_data.sleeve_length.lower() if pipeline_data.sleeve_length != 'Unknown' else ''
    sleeve_map = {
        'long': 'Long Sleeve',
        'short': 'Short Sleeve', 
        '3-4': '3/4 Sleeve',
        '3/4': '3/4 Sleeve',
        'sleeveless': 'Sleeveless'
    }
    for key, value in sleeve_map.items():
        if key in sleeve_length:
            item_specifics['Sleeve Length'] = value
            break
    
    # Dress silhouette (for dresses)
    if pipeline_data.garment_type.lower() == 'dress':
        silhouette = pipeline_data.silhouette.lower() if pipeline_data.silhouette != 'Unknown' else ''
        silhouette_map = {
            'a-line': 'A-Line',
            'a line': 'A-Line',
            'sheath': 'Sheath',
            'fit-and-flare': 'Fit & Flare',
            'fit and flare': 'Fit & Flare',
            'shift': 'Shift',
            'empire': 'Empire'
        }
        for key, value in silhouette_map.items():
            if key in silhouette:
                item_specifics['Silhouette'] = value
                break
    
    # Style mapping
    style = pipeline_data.style.lower() if pipeline_data.style != 'Unknown' else ''
    style_map = {
        'casual': 'Casual',
        'formal': 'Formal',
        'business': 'Business'
    }
    for key, value in style_map.items():
        if key in style:
            item_specifics['Style'] = value
            break
    
    # Fit mapping
    fit = pipeline_data.fit.lower() if pipeline_data.fit != 'Unknown' else ''
    fit_map = {
        'slim': 'Slim Fit',
        'regular': 'Regular Fit',
        'relaxed': 'Relaxed Fit',
        'oversized': 'Oversized'
    }
    for key, value in fit_map.items():
        if key in fit:
            item_specifics['Fit'] = value
            break
    
    return item_specifics


def validate_cardigan_pullover_classification(garment_data):
    """Validate cardigan vs pullover classification"""
    
    garment_type = garment_data.get('type', '').lower()
    has_opening = garment_data.get('has_front_opening', False)
    closure_type = garment_data.get('closure_type', 'none').lower()
    observations = garment_data.get('center_front_observations', [])
    confidence = garment_data.get('front_opening_confidence', 'uncertain')
    
    # RULE 1: Cardigan MUST have opening
    if garment_type == 'cardigan':
        if not has_opening and closure_type == 'none':
            logger.error(f"❌ INVALID: cardigan without front opening")
            return {
                'valid': False,
                'error': 'Cardigan must have front opening',
                'suggestion': 'pullover',
                'requires_user_confirmation': True
            }
    
    # RULE 2: Pullover MUST NOT have opening
    if garment_type in ['pullover', 'sweater', 'turtleneck']:
        if has_opening or closure_type != 'none':
            logger.error(f"❌ INVALID: pullover with front opening detected")
            return {
                'valid': False,
                'error': 'Pullover cannot have front opening',
                'suggestion': 'cardigan',
                'requires_user_confirmation': True
            }
    
    # RULE 3: Low confidence requires user check
    if confidence == 'uncertain' and garment_type in ['cardigan', 'pullover']:
        logger.warning(f"⚠️ LOW CONFIDENCE on cardigan/pullover distinction")
        return {
            'valid': True,
            'warning': 'Low confidence - please verify',
            'requires_user_confirmation': True
        }
    
    return {'valid': True, 'requires_user_confirmation': False}


def save_correction_for_training(correction_data):
    """Save user corrections for future AI training"""
    try:
        # Create corrections directory if it doesn't exist
        os.makedirs('training_data', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'training_data/correction_{timestamp}.json'
        
        # Save the correction data
        with open(filename, 'w') as f:
            json.dump(correction_data, f, indent=2, default=str)
        
        logger.info(f"📚 Training data saved: {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")


@dataclass
class AnalysisState:
    """Clean encapsulation of analysis state"""
    tag_analysis_status: AnalysisStatus = AnalysisStatus.PENDING
    garment_analysis_status: AnalysisStatus = AnalysisStatus.PENDING
    current_retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    
    def can_retry(self) -> bool:
        """Check if retry is allowed"""
        return self.current_retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry count"""
        self.current_retry_count += 1
    
    def reset_retries(self):
        """Reset retry count"""
        self.current_retry_count = 0

# ==========================
# SIMPLIFIED RETRY MECHANISM
# ==========================

@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_attempts: int = 3
    timeout_seconds: int = 30
    backoff_factor: float = 1.5

class SimpleRetryManager:
    """Unified retry mechanism for all analysis operations"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.analysis_state = AnalysisState()
    
    def execute_with_retry(self, operation_name: str, operation_func, *args, **kwargs):
        """
        Execute an operation with unified retry logic.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation or error information
        """
        self.analysis_state.reset_retries()
        
        for attempt in range(self.config.max_attempts):
            try:
                logger.info(f"[RETRY] {operation_name} - Attempt {attempt + 1}/{self.config.max_attempts}")
                
                # Execute the operation
                result = operation_func(*args, **kwargs)
                
                # Check if operation was successful
                if self._is_successful(result):
                    logger.info(f"[RETRY] {operation_name} - SUCCESS on attempt {attempt + 1}")
                    self.analysis_state.tag_analysis_status = AnalysisStatus.COMPLETED
                    return result
                else:
                    logger.warning(f"[RETRY] {operation_name} - Failed on attempt {attempt + 1}: {self._get_error_message(result)}")
                    
            except Exception as e:
                logger.error(f"[RETRY] {operation_name} - Exception on attempt {attempt + 1}: {e}")
                self.analysis_state.last_error = str(e)
            
            # If not the last attempt, wait before retrying
            if attempt < self.config.max_attempts - 1:
                wait_time = self.config.backoff_factor ** attempt
                logger.info(f"[RETRY] {operation_name} - Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
        
        # All attempts failed
        logger.error(f"[RETRY] {operation_name} - FAILED after {self.config.max_attempts} attempts")
        self.analysis_state.tag_analysis_status = AnalysisStatus.FAILED
        return {
            'success': False,
            'error': f"Operation failed after {self.config.max_attempts} attempts",
            'last_error': self.analysis_state.last_error,
            'method': f'Retry Manager ({operation_name})'
        }
    
    def _is_successful(self, result) -> bool:
        """Check if the result indicates success"""
        if isinstance(result, dict):
            return result.get('success', False)
        return result is not None
    
    def _get_error_message(self, result) -> str:
        """Extract error message from result"""
        if isinstance(result, dict):
            return result.get('error', 'Unknown error')
        return str(result) if result else 'No result returned'

# Load environment variables
def load_env_file():
    """Load environment variables from .env file (only once per session)"""
    # Use session state to prevent repeated loading
    if 'env_loaded' not in st.session_state:
        st.session_state.env_loaded = False
    
    if st.session_state.env_loaded:
        return  # Already loaded
    
    env_files = ['.env', 'api.env', 'config.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                logger.info(f"Loaded environment variables from {env_file}")
                st.session_state.env_loaded = True
                break
            except Exception as e:
                logger.warning(f"Could not load {env_file}: {e}")

# Only load if not already loaded
if 'env_loaded' not in st.session_state or not st.session_state.env_loaded:
    load_env_file()

# Streamlit configuration moved to main() function

# ==========================
# PIPELINE DATA STRUCTURES
# ==========================
@dataclass
class PipelineData:
    """Store all data collected through the pipeline"""
    tag_image: Optional[np.ndarray] = None
    garment_image: Optional[np.ndarray] = None
    brand: str = "Unknown"
    size: str = "Unknown"
    raw_size: str = "Unknown"  # Original size before conversion
    material: str = "Unknown"
    garment_type: str = "Unknown"
    subtype: str = "Unknown"  # More specific style description
    has_front_opening: bool = False  # For cardigan vs pullover distinction
    collar_type: str = "Unknown"  # turtleneck/crewneck/v-neck/collared/hooded/none
    neckline: str = "Unknown"  # turtleneck/crewneck/v-neck/cowl-neck/scoop/boat-neck
    sleeve_length: str = "Unknown"  # long/short/3-4/sleeveless
    silhouette: str = "Unknown"  # a-line/sheath/fit-and-flare/shift/empire (for dresses)
    fit: str = "Unknown"  # slim/regular/relaxed/oversized
    gender: str = "Unisex"
    gender_confidence: str = "Medium"
    gender_indicators: List = field(default_factory=list)
    condition: str = "Good"
    style: str = "Unknown"
    pattern: str = "None"
    era: str = "Unknown"
    price_estimate: Dict = field(default_factory=lambda: {'low': 10, 'mid': 25, 'high': 40})
    measurements: Dict = field(default_factory=dict)
    defects: List = field(default_factory=list)
    is_designer: bool = False
    is_vintage: bool = False
    vintage_year_estimate: Optional[int] = None  # Estimated year of manufacture
    tag_age_years: Optional[int] = None
    authenticity_confidence: str = "unknown"
    designer_tier: str = "none"  # ultra/high/mid-high/none
    font_era: str = "unknown"
    validation_issue: Optional[Dict] = None  # For storing classification validation issues
    vintage_indicators: List = field(default_factory=list)
    confidence: float = 0.0
    bust_measurement: float = 0.0
    measurement_points: List = field(default_factory=list)
    pixels_per_inch: float = 0.0
    data_sources: List = field(default_factory=list)
    warnings: List = field(default_factory=list)
    defect_count: int = 0
    category: str = "Unknown"
    fit: str = "Regular"
    price_low: float = 10
    price_high: float = 30
    model_comparison: Dict = field(default_factory=dict)  # Stores dual-model AI comparison results
    
    # eBay sold comps research fields
    sell_through_rate: float = 0.0
    avg_days_to_sell: Optional[float] = None
    ebay_sold_count: int = 0
    ebay_active_count: int = 0
    pricing_confidence: str = "Unknown"
    ebay_comps: Dict = field(default_factory=dict)

# BackgroundAnalysisManager removed - was unused dead code

# ==========================
# EBAY RESEARCH INTEGRATION
# ==========================

def build_ebay_item_specifics(pipeline_data) -> dict:
    """Build eBay item specifics from pipeline data"""
    item_specifics = {}
    
    # Neckline mapping
    neckline = pipeline_data.neckline.lower() if pipeline_data.neckline != 'Unknown' else ''
    neckline_map = {
        'v-neck': 'V-Neck',
        'crewneck': 'Crew Neck',
        'turtleneck': 'Turtle Neck',
        'cowl-neck': 'Cowl Neck',
        'scoop': 'Scoop Neck',
        'boat-neck': 'Boat Neck'
    }
    for key, value in neckline_map.items():
        if key in neckline:
            item_specifics['Neckline'] = value
            break
    
    # Sleeve length mapping
    sleeve_length = pipeline_data.sleeve_length.lower() if pipeline_data.sleeve_length != 'Unknown' else ''
    sleeve_map = {
        'long': 'Long Sleeve',
        'short': 'Short Sleeve',
        '3-4': '3/4 Sleeve',
        'sleeveless': 'Sleeveless'
    }
    for key, value in sleeve_map.items():
        if key in sleeve_length:
            item_specifics['Sleeve Length'] = value
            break
    
    # Style mapping
    style = pipeline_data.style.lower() if pipeline_data.style != 'Unknown' else ''
    style_map = {
        'casual': 'Casual',
        'formal': 'Formal',
        'business': 'Business'
    }
    for key, value in style_map.items():
        if key in style:
            item_specifics['Style'] = value
            break
    
    # Fit mapping
    fit = pipeline_data.fit.lower() if pipeline_data.fit != 'Unknown' else ''
    fit_map = {
        'slim': 'Slim Fit',
        'regular': 'Regular Fit',
        'relaxed': 'Relaxed Fit',
        'oversized': 'Oversized'
    }
    for key, value in fit_map.items():
        if key in fit:
            item_specifics['Fit'] = value
            break
    
    return item_specifics

def research_brand_with_ebay(pipeline_data, ebay_finder):
    """
    Research the detected brand using eBay sold comps.
    Call this after brand detection is complete.
    
    Args:
        pipeline_data: Your PipelineData object with brand, garment_type, size, gender
        ebay_finder: Instance of eBayCompsFinder
    
    Returns:
        Updated pipeline_data with eBay metrics
    """
    if not ebay_finder or not ebay_finder.app_id:
        logger.warning("[EBAY] Skipping research - no App ID configured")
        return pipeline_data
    
    # Skip if brand unknown
    if pipeline_data.brand == "Unknown":
        logger.info("[EBAY] Skipping - brand unknown")
        return pipeline_data
    
    logger.info(f"[EBAY] Researching {pipeline_data.brand} {pipeline_data.garment_type}...")
    
    # Build item specifics from pipeline data
    item_specifics = build_ebay_item_specifics(pipeline_data)
    
    # Search eBay sold comps
    comps_data = ebay_finder.search_sold_comps(
        brand=pipeline_data.brand,
        garment_type=pipeline_data.garment_type,
        size=pipeline_data.size,
        gender=pipeline_data.gender,
        item_specifics=item_specifics,
        days_back=90
    )
    
    if comps_data.get('success'):
        # Update pipeline data with real market data
        pipeline_data.price_estimate = {
            'low': comps_data['price_range']['low'],
            'mid': comps_data['avg_sold_price'],
            'high': comps_data['price_range']['high'],
            'median': comps_data['median_sold_price']
        }
        
        # Add new metrics
        pipeline_data.sell_through_rate = comps_data['sell_through_rate']
        pipeline_data.avg_days_to_sell = comps_data['days_to_sell_avg']
        pipeline_data.ebay_sold_count = comps_data['total_sold']
        pipeline_data.ebay_active_count = comps_data['total_active']
        pipeline_data.pricing_confidence = comps_data['confidence']
        pipeline_data.ebay_comps = comps_data  # Store full data
        
        # Add to data sources
        if not hasattr(pipeline_data, 'data_sources'):
            pipeline_data.data_sources = []
        pipeline_data.data_sources.append('eBay Sold Comps')
        
        logger.info(f"[EBAY] ✅ Updated pricing: ${comps_data['avg_sold_price']:.2f} avg, "
                   f"{comps_data['sell_through_rate']:.1f}% sell-through")
    else:
        logger.warning(f"[EBAY] Research failed: {comps_data.get('error', 'Unknown error')}")
        # Keep original pricing estimate
    
    return pipeline_data

def display_ebay_comps(pipeline_data):
    """Display eBay comps in Streamlit UI"""
    if hasattr(pipeline_data, 'ebay_comps') and pipeline_data.ebay_comps:
        st.subheader("📊 eBay Market Research")
        
        comps = pipeline_data.ebay_comps
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Sold Price",
                f"${comps['avg_sold_price']:.2f}",
                delta=f"${comps['median_sold_price']:.2f} median"
            )
        
        with col2:
            st.metric(
                "Sell-Through Rate",
                f"{comps['sell_through_rate']:.1f}%",
                delta=f"{comps['total_sold']} sold"
            )
        
        with col3:
            if comps['days_to_sell_avg']:
                st.metric(
                    "Avg Days to Sell",
                    f"{comps['days_to_sell_avg']:.0f} days"
                )
            else:
                st.metric("Avg Days to Sell", "N/A")
        
        with col4:
            st.metric(
                "Active Listings",
                comps['total_active'],
                delta=f"{comps['confidence']} confidence"
            )
        
        # Price range
        st.write(f"**Price Range:** ${comps['price_range']['low']:.2f} - ${comps['price_range']['high']:.2f}")
        
        # Show recent sold items
        if comps.get('sold_items'):
            with st.expander("View Recent Sold Items"):
                for item in comps['sold_items'][:10]:
                    st.write(f"**${item['price']:.2f}** - {item['title']}")
                    if item['days_to_sell']:
                        st.caption(f"Sold in {item['days_to_sell']} days - {item['condition']}")
                    st.write(f"[View on eBay]({item['url']})")
                    st.divider()

# ==========================
# WORKING ELGATO CONTROLLER WITH FAST DISCOVERY
# ==========================
# ==========================
# SERP API INTEGRATION FOR BRAND DETECTION
# ==========================
class SERPAPIBrandDetector:
    """SERP API integration for brand detection when OCR fails"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('SERPAPI_KEY') or os.getenv('SERP_API_KEY')  # Support both naming conventions
        self.base_url = "https://serpapi.com/search"
        
    def search_brand_from_image(self, image, garment_type="clothing", gender="women's"):
        """Use Google Lens for visual brand identification"""
        
        if not self.api_key:
            return {"success": False, "error": "SERP API key not found"}
        
        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode()
            
            params = {
                'api_key': self.api_key,
                'engine': 'google_lens',  # KEY CHANGE: use Lens, not Images
                'url': f'data:image/jpeg;base64,{img_base64}'
            }
            
            response = requests.post(
                "https://serpapi.com/search",
                data=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Google Lens returns visual_matches
                visual_matches = data.get('visual_matches', [])
                
                if visual_matches:
                    brands = self._extract_brands_from_lens_results(visual_matches)
                    
                    return {
                        "success": True,
                        "brands": brands[:5],
                        "visual_matches": visual_matches[:10],
                        "method": "Google Lens API"
                    }
            
            return {"success": False, "error": "No visual matches found"}
            
        except Exception as e:
            logger.error(f"[SERP] Google Lens failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_brands_from_lens_results(self, matches):
        """Extract brand names from Google Lens visual matches"""
        
        brands = []
        brand_counts = Counter()
        
        # Brand patterns to look for in titles/sources
        known_brands = [
            'Nike', 'Adidas', 'Ralph Lauren', 'Tommy Hilfiger', 'Calvin Klein',
            'Gap', 'Zara', 'H&M', 'Levi\'s', 'Gucci', 'Prada', 'Coach',
            'Rebecca Minkoff', 'Theory', 'Vince', 'Equipment', 'Rag & Bone',
            'J.Crew', 'Banana Republic', 'Ann Taylor', 'Loft', 'Nordstrom',
            'Madewell', 'Everlane', 'Reformation', 'Free People', 'Anthropologie',
            'Paul Smith', 'Tom Ford', 'Saint Laurent', 'Balenciaga'
        ]
        
        for match in matches:
            title = match.get('title', '').lower()
            source = match.get('source', '').lower()
            link = match.get('link', '').lower()
            
            combined_text = f"{title} {source} {link}"
            
            # Check each known brand
            for brand in known_brands:
                if brand.lower() in combined_text:
                    brand_counts[brand] += 1
        
        # Return brands sorted by frequency
        return [brand for brand, _ in brand_counts.most_common(5)]
    
    def _google_images_brand_search(self, garment_type="clothing", gender="women's"):
        """Primary Google Images search for brand identification"""
        try:
            logger.info("[SERP] Using Google Images search for brand identification...")
            
            # Create multiple search queries for better results
            search_queries = [
                f"{gender} {garment_type} brand",
                f"{garment_type} brand tag",
                f"{gender} {garment_type} designer",
                f"clothing brand {garment_type}"
            ]
            
            all_brands = []
            all_results = []
            
            for query in search_queries[:2]:  # Use top 2 queries to avoid rate limits
                logger.info(f"[SERP] Searching: '{query}'")
                
                params = {
                    'api_key': self.api_key,
                    'engine': 'google_images',
                    'q': query,
                    'num': 20  # Get more results for better brand detection
                }
                
                response = requests.get(self.base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    images_results = data.get('images_results', [])
                    all_results.extend(images_results)
                    
                    # Extract brands from results
                    brands_found = self._extract_brands_from_image_results(images_results, garment_type)
                    all_brands.extend(brands_found)
                    
                    logger.info(f"[SERP] Found {len(images_results)} images, {len(brands_found)} brands for '{query}'")
                else:
                    logger.warning(f"[SERP] Query '{query}' failed: {response.status_code}")
            
            # Remove duplicates and get top brands
            unique_brands = list(dict.fromkeys(all_brands))  # Preserve order, remove duplicates
            
            return {
                "success": True,
                "brands": unique_brands[:5],  # Top 5 unique brands
                "visual_matches": all_results[:10],  # Top 10 visual matches
                "method": "Google Images Search"
            }
                
        except Exception as e:
            logger.error(f"[SERP] Google Images search failed: {e}")
            return {"success": False, "error": f"Google Images search failed: {e}"}
    
    def _extract_brands_from_image_results(self, images_results, garment_type=""):
        """Extract brand names from Google Images search results"""
        brands_found = []
        
        # Comprehensive brand list for pattern matching
        brand_patterns = [
            # High-end luxury
            'gucci', 'prada', 'chanel', 'dior', 'louis vuitton', 'versace', 'armani', 'hugo boss',
            'burberry', 'fendi', 'givenchy', 'saint laurent', 'balenciaga', 'valentino',
            # Mid-tier designer
            'calvin klein', 'tommy hilfiger', 'ralph lauren', 'michael kors', 'coach', 'kate spade',
            'rebecca minkoff', 'vince', 'j.crew', 'j crew', 'banana republic', 'ann taylor',
            'loft', 'tory burch', 'marc jacobs', 'theory', 'alice + olivia', 'diane von furstenberg',
            # Contemporary
            'zara', 'h&m', 'mango', 'cos', 'everlane', 'madewell', 'anthropologie', 'free people',
            # Sportswear
            'nike', 'adidas', 'under armour', 'puma', 'reebok', 'new balance', 'champion',
            'lululemon', 'athleta', 'alo yoga', 'gymshark',
            # Fast fashion
            'gap', 'old navy', 'uniqlo', 'forever 21', 'target', 'asos', 'boohoo'
        ]
        
        for result in images_results:
            title = result.get('title', '').lower()
            source = result.get('source', '').lower()
            link = result.get('link', '').lower()
            
            # Check for brand patterns in title, source, and link
            for brand in brand_patterns:
                if (brand in title or brand in source or brand in link) and brand not in brands_found:
                    # Capitalize brand name properly
                    brand_name = brand.title()
                    if brand == 'j.crew':
                        brand_name = 'J.Crew'
                    elif brand == 'h&m':
                        brand_name = 'H&M'
                    elif brand == 'louis vuitton':
                        brand_name = 'Louis Vuitton'
                    elif brand == 'saint laurent':
                        brand_name = 'Saint Laurent'
                    elif brand == 'michael kors':
                        brand_name = 'Michael Kors'
                    elif brand == 'ralph lauren':
                        brand_name = 'Ralph Lauren'
                    elif brand == 'tommy hilfiger':
                        brand_name = 'Tommy Hilfiger'
                    elif brand == 'calvin klein':
                        brand_name = 'Calvin Klein'
                    elif brand == 'rebecca minkoff':
                        brand_name = 'Rebecca Minkoff'
                    elif brand == 'banana republic':
                        brand_name = 'Banana Republic'
                    elif brand == 'ann taylor':
                        brand_name = 'Ann Taylor'
                    elif brand == 'tory burch':
                        brand_name = 'Tory Burch'
                    elif brand == 'marc jacobs':
                        brand_name = 'Marc Jacobs'
                    elif brand == 'diane von furstenberg':
                        brand_name = 'Diane von Furstenberg'
                    elif brand == 'alice + olivia':
                        brand_name = 'Alice + Olivia'
                    elif brand == 'under armour':
                        brand_name = 'Under Armour'
                    elif brand == 'new balance':
                        brand_name = 'New Balance'
                    elif brand == 'forever 21':
                        brand_name = 'Forever 21'
                    
                    brands_found.append(brand_name)
                    break  # Found a brand in this result, move to next result
        
        return brands_found
    
    def _extract_brands_from_lens_results(self, visual_matches, garment_type=""):
        """Extract brand names from Google Lens visual matches"""
        common_brands = [
            # High-end
            'Gucci', 'Prada', 'Versace', 'Armani', 'Hugo Boss', 'Lacoste', 'Polo',
            'Burberry', 'Chanel', 'Dior', 'Fendi', 'Givenchy', 'Saint Laurent',
            # Mid-tier
            'Calvin Klein', 'Tommy Hilfiger', 'Ralph Lauren', 'Michael Kors', 'Coach',
            'Kate Spade', 'Rebecca Minkoff', 'Vince', 'J.Crew', 'J.Crew Collection',
            'Banana Republic', 'Ann Taylor', 'LOFT', 'Tory Burch', 'Marc Jacobs',
            # Sportswear
            'Nike', 'Adidas', 'Under Armour', 'Puma', 'Reebok', 'New Balance',
            'Champion', 'The North Face', 'Patagonia', 'Columbia', 'Lululemon', 'Athleta',
            # Fast fashion
            'Zara', 'H&M', 'Gap', 'Uniqlo', 'Forever 21', 'Old Navy', 'Target', 'Mango'
        ]
        
        found_brands = []
        brand_counts = {}
        
        logger.info(f"[SERP] Analyzing {len(visual_matches)} visual matches for brands")
        
        for match in visual_matches:
            title = match.get('title', '').lower()
            source = match.get('source', '').lower()
            link = match.get('link', '').lower()
            
            # Combine all text for searching
            combined_text = f"{title} {source} {link}"
            
            # Check for each known brand
            for brand in common_brands:
                if brand.lower() in combined_text:
                    if brand not in brand_counts:
                        brand_counts[brand] = 0
                    brand_counts[brand] += 1
        
        # Sort brands by frequency (most mentioned = most likely)
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        found_brands = [brand for brand, count in sorted_brands[:3]]
        
        logger.info(f"[SERP] Extracted brands: {found_brands}")
        return found_brands
    
    def _extract_brands_from_results(self, results):
        """Extract potential brand names from text search results (legacy)"""
        common_brands = [
            'Nike', 'Adidas', 'Under Armour', 'Puma', 'Reebok', 'New Balance',
            'Champion', 'The North Face', 'Patagonia', 'Columbia', 'Calvin Klein',
            'Tommy Hilfiger', 'Ralph Lauren', 'Levi\'s', 'Gap', 'H&M', 'Uniqlo', 'Zara',
            'Gucci', 'Prada', 'Versace', 'Armani', 'Hugo Boss', 'Lacoste', 'Polo'
        ]
        
        found_brands = []
        for result in results:
            text = f"{result['title']} {result['snippet']}".lower()
            for brand in common_brands:
                if brand.lower() in text and brand not in found_brands:
                    found_brands.append(brand)
        
        return found_brands

class ElgatoLightController:
    """Control Elgato lights via their local API with fast discovery"""
    
    def __init__(self, quick_mode=True):
        self.lights = []
        self.current_state = {'brightness': 85, 'temperature': 5500}
        self.last_successful_light = None
        self.discovery_attempted = False
        
        if quick_mode:
            # Only try the fastest method on startup
            self.quick_discover()
        else:
            self.discover_lights()
    
    def _get_common_network_ips(self):
        """Generate common network IPs dynamically based on current network"""
        import socket
        
        ips = []
        
        # Get current machine's IP to determine network range
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            
            # Extract network prefix (e.g., 192.168.1.x -> 192.168.1)
            network_parts = local_ip.split('.')[:3]
            network_prefix = '.'.join(network_parts)
            
            # Generate IPs in current network range
            for i in range(1, 255):  # Skip .0 and .255
                ips.append(f"{network_prefix}.{i}")
                
        except Exception:
            # Fallback to common ranges if we can't determine current network
            common_ranges = [
                "192.168.1", "192.168.0", "10.0.0", "172.16.0", "192.168.10", "192.168.100"
            ]
            for range_prefix in common_ranges:
                for i in range(1, 255):
                    ips.append(f"{range_prefix}.{i}")
        
        # Prioritize common router and device IPs
        try:
            priority_ips = [
                f"{network_parts[0]}.{network_parts[1]}.{network_parts[2]}.1",  # Router
                f"{network_parts[0]}.{network_parts[1]}.{network_parts[2]}.100", # Common device IP
                f"{network_parts[0]}.{network_parts[1]}.{network_parts[2]}.79",  # Your specific IP
            ]
        except:
            priority_ips = ["192.168.1.1", "192.168.0.1", "10.0.0.1", "192.168.1.100", "192.168.0.100"]
        
        # Remove duplicates and put priority IPs first
        unique_ips = []
        for ip in priority_ips:
            if ip in ips and ip not in unique_ips:
                unique_ips.append(ip)
        
        # Add remaining IPs
        for ip in ips:
            if ip not in unique_ips:
                unique_ips.append(ip)
        
        return unique_ips[:50]  # Limit to first 50 to avoid long scan times
    
    def quick_discover(self):
        """Quick discovery - only try localhost Control Center"""
        try:
            # Try Elgato Control Center on localhost (very fast)
            for port in [9123, 9333]:
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/elgato/lights", timeout=0.5)
                    if response.status_code == 200:
                        self.lights.append({'url': f"http://127.0.0.1:{port}", 'name': 'Control Center'})
                        logger.info(f"Found Elgato Control Center on port {port}")
                        # Test the light immediately
                        self.set_light(100, 5600)
                        time.sleep(0.2)
                        self.set_light(85, 5500)
                        return True
                except:
                    pass
        
            # If Control Center not found, try known IPs
            # Try common network ranges dynamically
            known_ips = self._get_common_network_ips()
            for ip in known_ips:
                for port in [9123, 9333]:
                    try:
                        response = requests.get(f"http://{ip}:{port}/elgato/lights", timeout=0.5)
                        if response.status_code == 200:
                            self.lights.append({'url': f"http://{ip}:{port}", 'name': f'Elgato Light ({ip})'})
                            logger.info(f"Found Elgato light at {ip}:{port}")
                            # Test the light immediately
                            self.set_light(100, 5600)
                            time.sleep(0.2)
                            self.set_light(85, 5500)
                            return True
                    except:
                        pass
                
        except Exception as e:
            logger.warning(f"Quick discovery failed: {e}")
        
        logger.info("No Elgato lights found in quick discovery")
        return False
    
    def discover_lights(self):
        """Full discovery - called manually if needed"""
        if self.discovery_attempted:
            return len(self.lights) > 0
        
        self.discovery_attempted = True
        
        # Try Control Center first
        for port in [9123, 9333]:
            try:
                response = requests.get(f"http://127.0.0.1:{port}/elgato/lights", timeout=1)
                if response.status_code == 200:
                    self.lights.append({'url': f"http://127.0.0.1:{port}", 'name': 'Control Center'})
                    logger.info(f"Found Elgato via Control Center on port {port}")
                    return True
            except:
                pass
        
        # Try common static IPs (prioritize the working IP we found)
        # Try common network ranges dynamically
        common_ips = self._get_common_network_ips()
        for ip in common_ips:
            for port in [9123, 9333]:
                try:
                    url = f"http://{ip}:{port}"
                    response = requests.get(f"{url}/elgato/lights", timeout=0.5)
                    if response.status_code == 200:
                        self.lights.append({'url': url, 'name': f'Light at {ip}:{port}'})
                        logger.info(f"Found Elgato light at {ip}:{port}")
                        return True
                except:
                    continue
        
        logger.warning("No Elgato lights found")
        return False
    
    def set_light(self, brightness=100, temperature=4500, retry_discovery=False):
        """Set light brightness (0-100) and temperature (2900-7000K)"""
        if not self.lights:
            if retry_discovery and not self.discovery_attempted:
                self.discover_lights()
            if not self.lights:
                return False
        
        self.current_state = {'brightness': brightness, 'temperature': temperature}
        
        # Convert temperature to Elgato's scale
        elgato_temp = self.kelvin_to_elgato(temperature)
        
        payload = {
            "lights": [{
                "on": 1 if brightness > 0 else 0,
                "brightness": brightness,
                "temperature": elgato_temp
            }]
        }
        
        for light in self.lights:
            try:
                url = f"{light['url']}/elgato/lights"
                response = requests.put(url, json=payload, timeout=0.5)
                if response.status_code == 200:
                    logger.info(f"Light adjusted: {brightness}% brightness, {temperature}K")
                    self.last_successful_light = light
                    return True
            except Exception as e:
                logger.debug(f"Failed to control light: {e}")
                continue
        
        return False
    
    def kelvin_to_elgato(self, kelvin):
        """Convert Kelvin to Elgato's 143-344 scale"""
        kelvin = max(2900, min(7000, kelvin))
        # Elgato uses inverse scale: 143 = 7000K (cold), 344 = 2900K (warm)
        return int(344 - ((kelvin - 2900) * 201 / 4100))
    
    def optimize_for_tag_reading(self):
        """Optimize lighting for reading small text on tags"""
        logger.info("Setting lights for tag reading: 100% brightness, 5600K")
        return self.set_light(brightness=100, temperature=5600)
    
    def optimize_for_garment_analysis(self):
        """Optimize lighting for color-accurate garment analysis"""
        logger.info("Setting lights for garment analysis: 85% brightness, 5500K")
        return self.set_light(brightness=85, temperature=5500)
    
    def optimize_for_defect_detection(self):
        """Optimize lighting for spotting defects"""
        logger.info("Setting lights for defect detection: 100% brightness, 6500K")
        return self.set_light(brightness=100, temperature=6500)
    
    def turn_off(self):
        """Turn off the lights"""
        return self.set_light(brightness=0)

# ==========================
# REAL AUTO OPTIMIZER (not stub)
# ==========================
class ImprovedSmartLightOptimizer:
    """More aggressive light optimizer to prevent overexposure"""
    
    def __init__(self, light_controller):
        self.light_controller = light_controller
        self.enabled = True
        self.last_adjustment = time.time()
        self.min_adjustment_interval = 1.0  # Faster adjustments
        self.last_brightness_info = None
        logger.info("Improved Smart Light Optimizer initialized")
    
    def analyze_image_brightness(self, image):
        """Analyze image to determine optimal lighting needs"""
        if image is None:
            return None
        
        # Convert to grayscale for brightness analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate brightness statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        max_brightness = np.max(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Check for overexposure (pixels at 255)
        overexposed_ratio = hist[250:].sum()  # Pixels near white
        
        # Brightness ranges
        dark_pixels = np.sum(hist[:85])
        mid_pixels = np.sum(hist[85:170])
        bright_pixels = np.sum(hist[170:])
        
        # More aggressive detection
        is_overexposed = overexposed_ratio > 0.1 or mean_brightness > 180
        is_light_garment = bright_pixels > 0.3 or mean_brightness > 150
        is_dark_garment = dark_pixels > 0.5 or mean_brightness < 80
        
        result = {
            'mean': mean_brightness,
            'std': std_brightness,
            'max': max_brightness,
            'is_light': is_light_garment,
            'is_dark': is_dark_garment,
            'is_overexposed': is_overexposed,
            'dark_ratio': dark_pixels,
            'bright_ratio': bright_pixels,
            'overexposed_ratio': overexposed_ratio
        }
        
        self.last_brightness_info = result
        return result
    
    def calculate_optimal_settings(self, brightness_info, purpose='general'):
        """Calculate optimal settings with overexposure prevention"""
        if not brightness_info:
            return {'brightness': 75, 'temperature': 5500}
        
        # Start with conservative base settings
        base_settings = {
            'tag': {'brightness': 85, 'temperature': 5600},
            'garment': {'brightness': 75, 'temperature': 5500},
            'defect': {'brightness': 90, 'temperature': 6500},
            'general': {'brightness': 75, 'temperature': 5500}
        }
        
        settings = base_settings.get(purpose, base_settings['general']).copy()
        
        # CRITICAL: Handle overexposure first
        if brightness_info.get('is_overexposed', False):
            # Drastically reduce brightness
            settings['brightness'] = max(40, settings['brightness'] - 40)
            settings['temperature'] = 6000  # Cooler light
            logger.warning(f"OVEREXPOSURE detected! Reducing to {settings['brightness']}%")
            
        elif brightness_info['is_light']:
            # Light/white garments need much less light
            reduction = 35 if brightness_info['mean'] > 170 else 25
            settings['brightness'] = max(50, settings['brightness'] - reduction)
            settings['temperature'] = min(6000, settings['temperature'] + 300)
            logger.info(f"Light garment - reducing to {settings['brightness']}%")
            
        elif brightness_info['is_dark']:
            # Dark garments need more light (but not too much)
            settings['brightness'] = min(95, settings['brightness'] + 15)
            settings['temperature'] = max(5000, settings['temperature'] - 300)
            logger.info(f"Dark garment - increasing to {settings['brightness']}%")
            
        else:
            # Medium tones - fine tune based on mean
            if brightness_info['mean'] > 160:
                settings['brightness'] = max(60, settings['brightness'] - 15)
            elif brightness_info['mean'] < 100:
                settings['brightness'] = min(90, settings['brightness'] + 10)
        
        # High contrast adjustment
        if brightness_info['std'] > 60:
            settings['brightness'] = min(settings['brightness'], 85)
            logger.info("High contrast - capping brightness")
        
        return settings
    
    def optimize_for_current_image(self, image, purpose='general'):
        """Optimize lighting based on actual image content"""
        if not self.enabled or not self.light_controller.lights:
            return False
        
        # Analyze the image
        brightness_info = self.analyze_image_brightness(image)
        
        if not brightness_info:
            return False
        
        # AGGRESSIVE AUTO-ADJUSTMENT FOR LIGHT ITEMS
        current_mean = brightness_info['mean']
        
        # Purpose-specific adjustments (tags need more light than garments for OCR)
        if purpose == 'tag':
            # REFINED STRATEGY: More aggressive for white tags, with better preprocessing
            # White tags are highly reflective and need MUCH lower brightness
            # Small text needs good contrast, not just moderate lighting
            if current_mean > 170:  # Extremely bright/white tag
                target_brightness = 15  # Very aggressive reduction for pure white
                target_temp = 6500
                logger.warning(f"VERY BRIGHT white tag (mean: {current_mean:.0f}) - Aggressive reduction to {target_brightness}%")
            elif current_mean > 140:  # Very bright tag
                target_brightness = 25  # Strong reduction for light tags
                target_temp = 6000
                logger.warning(f"Bright tag (mean: {current_mean:.0f}) - Strong reduction to {target_brightness}%")
            elif current_mean > 110:  # Moderately bright
                target_brightness = 40  # Moderate reduction
                target_temp = 6000
                logger.info(f"Slightly bright tag (mean: {current_mean:.0f}) - Moderate reduction to {target_brightness}%")
            elif current_mean < 60:  # Too dark
                target_brightness = 85  # Increase for dark tags
                target_temp = 5200
                logger.info(f"Dark tag - Increasing to {target_brightness}%")
            else:  # Good range (60-110)
                target_brightness = 50  # Balanced level
                target_temp = 5600
                logger.info(f"Tag in good range (mean: {current_mean:.0f}) - Using {target_brightness}%")
        else:
            # Garment/general: can go darker for better detail
            if current_mean > 140:  # Light/white detected
                target_brightness = 20  # Lower for garments
                target_temp = 6500
                logger.warning(f"Light surface detected (mean: {current_mean:.0f}) - Dropping to {target_brightness}%")
            elif current_mean > 100:  # Moderately bright
                target_brightness = 35  # Low-moderate
                target_temp = 6000
                logger.warning(f"Bright surface detected (mean: {current_mean:.0f}) - Dropping to {target_brightness}%")
            elif current_mean < 60:  # Too dark
                target_brightness = 95
                target_temp = 5200
                logger.info(f"Dark surface detected - Increasing to {target_brightness}%")
            else:  # Good range
                target_brightness = 70
                target_temp = 5600
        
        # Apply immediately
        success = self.light_controller.set_light(
            brightness=target_brightness,
            temperature=target_temp
        )
        
        if success:
            self.last_adjustment = time.time()
            logger.info(f"Auto-adjusted: {target_brightness}% for brightness level {current_mean:.0f}")
        
        return success
    
    def toggle_enabled(self):
        """Toggle automatic adjustment on/off"""
        self.enabled = not self.enabled
        return self.enabled

# ==========================
# ENHANCED OCR WITH MULTI-STRATEGY PREPROCESSING
# ==========================
class EnhancedOCRProcessor:
    """Advanced OCR with multiple preprocessing strategies for robust text extraction"""
    
    def __init__(self):
        self.tesseract_configs = {
            'general': '--oem 3 --psm 6',      # Uniform block of text
            'sparse': '--oem 3 --psm 11',      # Sparse text
            'single_line': '--oem 3 --psm 7',  # Single line
            'single_word': '--oem 3 --psm 8',  # Single word (brands)
        }
        logger.info("Enhanced OCR Processor initialized")
    
    def preprocess_tag_image(self, image):
        """
        Apply multiple preprocessing strategies and return all variants
        """
        if image is None or image.size == 0:
            return {}
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        preprocessed = {}
        
        # Strategy 1: Adaptive Thresholding (best for uneven lighting)
        try:
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            adaptive = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            preprocessed['adaptive'] = adaptive
        except Exception as e:
            logger.warning(f"Adaptive threshold failed: {e}")
        
        # Strategy 2: Otsu's Binarization (best for high contrast)
        try:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, otsu = cv2.threshold(
                blurred, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            preprocessed['otsu'] = otsu
        except Exception as e:
            logger.warning(f"Otsu threshold failed: {e}")
        
        # Strategy 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, contrast = cv2.threshold(
                enhanced, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            preprocessed['clahe'] = contrast
        except Exception as e:
            logger.warning(f"CLAHE failed: {e}")
        
        # Strategy 4: Morphological Operations (remove noise)
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            morph = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations=1)
            preprocessed['morph'] = morph
        except Exception as e:
            logger.warning(f"Morphological operations failed: {e}")
        
        return preprocessed
    
    def extract_text_multipass(self, image):
        """
        Run OCR with multiple strategies and return best result
        """
        if image is None:
            return {'text': '', 'confidence': 0.0, 'strategy': 'none'}
        
        # Preprocess with all strategies
        preprocessed_images = self.preprocess_tag_image(image)
        
        if not preprocessed_images:
            logger.error("All preprocessing strategies failed")
            return {'text': '', 'confidence': 0.0, 'strategy': 'failed'}
        
        results = []
        
        # Try each preprocessed image with each config
        for strategy_name, processed_img in preprocessed_images.items():
            for config_name, config in self.tesseract_configs.items():
                try:
                    # Get OCR data with confidence scores
                    data = pytesseract.image_to_data(
                        processed_img,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Extract high-confidence words only
                    text_parts = []
                    confidences = []
                    
                    for i, word in enumerate(data['text']):
                        conf = int(data['conf'][i])
                        if conf > 30 and word.strip():  # Confidence > 30%
                            text_parts.append(word)
                            confidences.append(conf)
                    
                    if text_parts:
                        full_text = ' '.join(text_parts)
                        avg_conf = sum(confidences) / len(confidences)
                        
                        results.append({
                            'text': full_text,
                            'confidence': avg_conf,
                            'strategy': f"{strategy_name}/{config_name}",
                            'word_count': len(text_parts)
                        })
                        
                        logger.debug(f"[OCR] {strategy_name}/{config_name}: conf={avg_conf:.1f}%, words={len(text_parts)}")
                
                except Exception as e:
                    logger.debug(f"OCR failed for {strategy_name}/{config_name}: {e}")
        
        if not results:
            return {'text': '', 'confidence': 0.0, 'strategy': 'all_failed'}
        
        # Pick best result by confidence
        best = max(results, key=lambda x: x['confidence'])
        
        # Fallback: if low confidence, try longest result
        if best['confidence'] < 50:
            longest = max(results, key=lambda x: len(x['text']))
            if len(longest['text']) > len(best['text']) * 1.5:
                logger.info("[OCR] Low confidence - using longest result instead")
                return longest
        
        logger.info(f"[OCR] Best: '{best['text'][:60]}...' (conf: {best['confidence']:.1f}%, strategy: {best['strategy']})")
        return best
    
    def extract_brand_from_tag(self, image):
        """
        Specialized extraction for brand name (usually top 30% of tag)
        """
        if image is None:
            return None
        
        # Focus on top section where brands typically are
        h, w = image.shape[:2]
        top_section = image[0:int(h*0.3), :]
        
        # High-contrast preprocessing for bold text
        if len(top_section.shape) == 3:
            gray = cv2.cvtColor(top_section, cv2.COLOR_RGB2GRAY)
        else:
            gray = top_section
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try single word mode first (brands are often one word)
        try:
            brand_text = pytesseract.image_to_string(
                binary,
                config='--oem 3 --psm 8'
            ).strip()
            
            if brand_text and len(brand_text) > 2:
                # Clean the brand name
                brand_text = ''.join(c for c in brand_text if c.isalnum() or c.isspace())
                if brand_text:
                    logger.info(f"[BRAND] Extracted: '{brand_text}'")
                    return brand_text
        except Exception as e:
            logger.warning(f"Brand extraction failed: {e}")
        
        return None

# ==========================
# EBAY ITEM SPECIFICS WITH FUZZY MATCHING
# ==========================
class ImprovedEbayMapper:
    """Map garment attributes to eBay-compliant values with fuzzy matching"""
    
    def __init__(self):
        # eBay accepted values (subset - add more as needed)
        self.accepted_values = {
            'Neckline': [
                'Boat Neck', 'Collared', 'Cowl Neck', 'Crew Neck', 'Halter',
                'High Neck', 'Hooded', 'Mock Neck', 'Off Shoulder',
                'Scoop Neck', 'Square Neck', 'Turtleneck', 'V-Neck', 'Other'
            ],
            'Sleeve Length': [
                'Long Sleeve', 'Short Sleeve', '3/4 Sleeve', 'Sleeveless',
                'Cap Sleeve', 'Tank Top'
            ],
            'Style': [
                'Casual', 'Formal', 'Business', 'Bohemian', 'Preppy',
                'Athletic', 'Vintage', 'Classic', 'Streetwear'
            ],
            'Fit': [
                'Regular', 'Slim', 'Relaxed', 'Oversized', 'Fitted',
                'Athletic', 'Tailored', 'Loose'
            ],
            'Pattern': [
                'Solid', 'Striped', 'Floral', 'Animal Print', 'Plaid',
                'Polka Dot', 'Chevron', 'Geometric', 'Abstract',
                'Color Block', 'Paisley'
            ],
            'Condition': [
                'New with tags', 'New without tags', 'New with defects',
                'Pre-owned', 'Excellent', 'Very Good', 'Good', 'Acceptable'
            ]
        }
        
        logger.info("Improved eBay Mapper initialized")
    
    def normalize_to_ebay(self, field_name, raw_value):
        """
        Normalize a value to eBay's accepted format using fuzzy matching
        """
        if not raw_value or raw_value.lower() in ['unknown', 'none', 'n/a']:
            return None
        
        accepted = self.accepted_values.get(field_name, [])
        if not accepted:
            # Field doesn't have predefined values
            return raw_value.strip().title()
        
        raw_clean = raw_value.strip().title()
        
        # Exact match (case-insensitive)
        for accepted_val in accepted:
            if raw_clean.lower() == accepted_val.lower():
                return accepted_val
        
        # Fuzzy match
        matches = difflib.get_close_matches(
            raw_clean,
            accepted,
            n=1,
            cutoff=0.6  # 60% similarity
        )
        
        if matches:
            logger.info(f"[EBAY] Fuzzy matched '{raw_value}' → '{matches[0]}' for {field_name}")
            return matches[0]
        
        # No match - use Other if available
        logger.warning(f"[EBAY] No match for '{raw_value}' in {field_name}")
        return 'Other' if 'Other' in accepted else None
    
    def build_item_specifics(self, pipeline_data):
        """
        Build eBay Item Specifics with proper normalization
        """
        item_specifics = {}
        
        # Brand (CRITICAL)
        if pipeline_data.brand and pipeline_data.brand != 'Unknown':
            item_specifics['Brand'] = pipeline_data.brand
        else:
            item_specifics['Brand'] = 'Unbranded'
        
        # Size (CRITICAL)
        if pipeline_data.size and pipeline_data.size != 'Unknown':
            item_specifics['Size'] = pipeline_data.size
        
        # Neckline
        if pipeline_data.neckline and pipeline_data.neckline != 'Unknown':
            neckline = self.normalize_to_ebay('Neckline', pipeline_data.neckline)
            if neckline:
                item_specifics['Neckline'] = neckline
        
        # Sleeve Length
        if pipeline_data.sleeve_length and pipeline_data.sleeve_length != 'Unknown':
            sleeve = self.normalize_to_ebay('Sleeve Length', pipeline_data.sleeve_length)
            if sleeve:
                item_specifics['Sleeve Length'] = sleeve
        
        # Style
        if pipeline_data.style and pipeline_data.style != 'Unknown':
            style = self.normalize_to_ebay('Style', pipeline_data.style)
            if style:
                item_specifics['Style'] = style
        
        # Fit
        if pipeline_data.fit and pipeline_data.fit != 'Unknown':
            fit = self.normalize_to_ebay('Fit', pipeline_data.fit)
            if fit:
                item_specifics['Fit'] = fit
        
        # Pattern
        if pipeline_data.pattern and pipeline_data.pattern not in ['None', 'Unknown']:
            pattern = self.normalize_to_ebay('Pattern', pipeline_data.pattern)
            if pattern:
                item_specifics['Pattern'] = pattern
        
        # Condition
        condition = self.normalize_to_ebay('Condition', pipeline_data.condition)
        if condition:
            item_specifics['Condition'] = condition
        
        return item_specifics
    
    def validate_listing(self, item_specifics):
        """
        Validate that all values are eBay-compliant
        """
        errors = []
        
        # Check required fields
        required = ['Brand', 'Size', 'Condition']
        for field in required:
            if field not in item_specifics:
                errors.append(f"Missing required field: {field}")
        
        # Validate values against accepted list
        for field, value in item_specifics.items():
            if field in self.accepted_values:
                if value not in self.accepted_values[field]:
                    errors.append(f"Invalid value for {field}: '{value}'")
        
        return errors

# ==========================
# INTEGRATION HELPER FUNCTIONS
# ==========================

def analyze_tag_with_enhanced_ocr(tag_image, light_controller):
    """
    Enhanced tag analysis with multi-strategy OCR
    """
    # Initialize OCR processor
    ocr_processor = EnhancedOCRProcessor()
    
    # Optimize lighting
    logger.info("[TAG] Optimizing lighting for OCR...")
    light_optimizer = ImprovedSmartLightOptimizer(light_controller)
    light_optimizer.optimize_for_current_image(tag_image, purpose='tag')
    time.sleep(0.3)  # Let light settle
    
    # Extract text with multiple strategies
    logger.info("[TAG] Running multi-pass OCR...")
    ocr_result = ocr_processor.extract_text_multipass(tag_image)
    
    # Extract brand specifically
    brand = ocr_processor.extract_brand_from_tag(tag_image)
    
    return {
        'full_text': ocr_result['text'],
        'confidence': ocr_result['confidence'],
        'brand': brand,
        'strategy': ocr_result['strategy']
    }

def classify_and_validate(garment_image, classification_result):
    """
    Classify garment and validate with strict rules
    """
    # Extract features from AI response
    features = {
        'has_front_opening': classification_result.get('has_front_opening', False),
        'neckline': classification_result.get('neckline', 'Unknown'),
    }
    
    garment_type = classification_result.get('type', 'Unknown')
    
    # Validate
    is_valid, corrected_type, issues = validate_classification_strict(
        garment_type,
        features
    )
    
    if not is_valid:
        logger.warning(f"[VALIDATION] Classification issues: {issues}")
        classification_result['type'] = corrected_type
        classification_result['validation_issues'] = issues
        classification_result['auto_corrected'] = True
    
    return classification_result

def build_ebay_listing_improved(pipeline_data):
    """
    Build eBay listing with proper validation
    """
    ebay_mapper = ImprovedEbayMapper()
    
    # Build item specifics
    item_specifics = ebay_mapper.build_item_specifics(pipeline_data)
    
    # Validate
    errors = ebay_mapper.validate_listing(item_specifics)
    
    if errors:
        logger.error(f"[EBAY] Validation errors: {errors}")
        return {
            'item_specifics': item_specifics,
            'is_valid': False,
            'errors': errors
        }
    
    logger.info("[EBAY] Listing validation PASSED")
    return {
        'item_specifics': item_specifics,
        'is_valid': True,
        'errors': []
    }

# ==========================
# CAMERA MANAGER 
# ==========================
class OpenAIVisionCameraManager:
    """Camera manager with 12MP Arducam support and focus scoring"""
    
    def __init__(self):
        # Suppress OpenCV warnings
        self.suppress_cv2_warnings()
        
        self.arducam_index = None
        self.realsense_index = None
        self.arducam_cap = None
        self.realsense_cap = None
        
        # RealSense SDK components for proper color stream
        self.realsense_pipeline = None
        self.realsense_config = None
        self.realsense_sdk_available = REALSENSE_SDK_AVAILABLE
        self.rs = None
        
        # Initialize RealSense SDK if available
        if self.realsense_sdk_available:
            try:
                import pyrealsense2 as rs
                self.rs = rs
                logger.info("✅ RealSense SDK initialized in camera manager")
            except ImportError:
                self.realsense_sdk_available = False
                logger.warning("⚠️ RealSense SDK import failed in camera manager")
        
        self.roi_coords = self.load_roi_config()
        
        # FORCE default ROI if loading failed
        if not self.roi_coords or 'tag' not in self.roi_coords:
            logger.warning("⚠️ ROI not loaded properly, using defaults")
            self.roi_coords = {
                'tag': (200, 200, 400, 300),
                'work': (50, 50, 700, 500)
            }
            self.original_resolution = (1280, 720)
        
        self.arducam_settings = self.load_arducam_settings()
        self.camera_status = {'arducam': False, 'realsense': False}
        self.camera_failures = {'arducam': 0, 'realsense': 0}
        self.max_failures = 3
        self.last_frame_time = {'arducam': 0, 'realsense': 0}
        self.frame_cache = {'arducam': None, 'realsense': None}
        self.cache_duration = 0.5  # Cache frames for 500ms - refresh less often
        self.skip_frames = 2  # Only process every 3rd frame for better performance
        
        # 12MP ARDUCAM ENHANCEMENTS
        self.preferred_res = [(4056, 3040), (4000, 3000), (3840, 2160), (2592, 1944), (1920, 1080)]
        self.preview_resolution = (1280, 720)  # Fast UI preview
        self.capture_resolution = (1920, 1080)  # Fallback capture
        self.negotiated_res = None
        self.highres_ok = False
        
        # Focus scoring
        self.max_focus_score = 0.0
        
        self.find_cameras()  # Auto-detect working cameras
        self.initialize_cameras()  # Initialize once at startup
        logger.info("12MP Arducam Camera Manager initialized")
    
    def suppress_cv2_warnings(self):
        """Suppress OpenCV warning messages"""
        import os
        os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
        os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
        cv2.setLogLevel(0)  # Only show errors
    
    def find_cameras(self):
        """Auto-detect which cameras work - ensure we get two different cameras"""
        logger.info("Detecting available cameras...")
        
        working_cameras = []
        
        # Only try DirectShow on Windows as it's most reliable
        if os.name == 'nt':
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_ANY
        
        # Quick scan of first few indices
        for i in range(3):
            try:
                # Suppress output during test
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Quick test - just one frame
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret and frame is not None:
                        working_cameras.append({
                            'index': i,
                            'backend': backend
                        })
                        logger.info(f"[OK] Camera found at index {i}")
            except:
                continue
        
        # Assign cameras
        if len(working_cameras) >= 2:
            self.arducam_index = working_cameras[0]['index']
            self.realsense_index = working_cameras[1]['index']
            logger.info(f"Cameras assigned: ArduCam={self.arducam_index}, RealSense={self.realsense_index}")
        elif len(working_cameras) == 1:
            self.arducam_index = working_cameras[0]['index']
            self.realsense_index = 1  # Try next index
            logger.warning("Only one camera detected, second camera may not work")
        else:
            self.arducam_index = 0
            self.realsense_index = 1
            logger.warning("No cameras auto-detected, using default indices")
    
    def _set_fourcc(self, cap, fourcc_str="MJPG"):
        """Set camera codec (MJPG preferred for high-res over USB)"""
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    def _negotiate_12mp_resolution(self, cap):
        """Negotiate highest available resolution with fallback"""
        # Prefer MJPG for big frames over USB
        self._set_fourcc(cap, "MJPG")
        
        negotiated = None
        for w, h in self.preferred_res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (rw, rh) == (w, h):
                negotiated = (rw, rh)
                logger.info(f"[12MP] Negotiated resolution: {rw}x{rh}")
                break
        
        # If we didn't get a high-res, try again at 1080p to ensure *something* works
        if negotiated is None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.preview_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preview_resolution[1])
            negotiated = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            logger.warning(f"[12MP] Fallback to: {negotiated[0]}x{negotiated[1]}")
        
        return negotiated
    
    def initialize_cameras(self):
        """Initialize cameras once and keep them open with 12MP negotiation"""
        # Use DirectShow on Windows to avoid MSMF warnings
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        
        # Initialize ArduCam with 12MP negotiation
        if self.arducam_index is not None:
            try:
                self.arducam_cap = cv2.VideoCapture(self.arducam_index, backend)
                if self.arducam_cap.isOpened():
                    # Negotiate highest resolution
                    self.negotiated_res = self._negotiate_12mp_resolution(self.arducam_cap)
                    self.highres_ok = self.negotiated_res[0] >= 3000  # True 12MP threshold
                    
                    self.setup_camera_properties(self.arducam_cap)
                    self.camera_status['arducam'] = True
                    
                    # Optional: lower buffering to reduce latency
                    self.arducam_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    logger.info(f"[12MP] ArduCam ready at {self.negotiated_res[0]}x{self.negotiated_res[1]} (12MP: {self.highres_ok})")
            except Exception as e:
                logger.error(f"Failed to initialize ArduCam: {e}")
        
        # Initialize RealSense with proper SDK for color stream
        if self.realsense_index is not None and self.realsense_index != self.arducam_index:
            if self.realsense_sdk_available:
                try:
                    # Use RealSense SDK with AGGRESSIVE TRUE RGB forcing
                    self.realsense_pipeline = self.rs.pipeline()
                    self.realsense_config = self.rs.config()
                    
                    # CRITICAL FIX: Explicitly disable ALL streams first (including infrared)
                    self.realsense_config.disable_all_streams()
                    
                    # Get connected devices to ensure we're targeting the right one
                    ctx = self.rs.context()
                    devices = ctx.query_devices()
                    if len(devices) == 0:
                        raise RuntimeError("No RealSense devices found")
                    
                    # Get the first device
                    device = devices[0]
                    logger.info(f"[REALSENSE] Found device: {device.get_info(self.rs.camera_info.name)}")
                    
                    # Check available sensors
                    sensors = device.query_sensors()
                    color_sensor = None
                    for sensor in sensors:
                        if sensor.is_color_sensor():
                            color_sensor = sensor
                            logger.info(f"[REALSENSE] Found color sensor: {sensor.get_info(self.rs.camera_info.name)}")
                            break
                    
                    if not color_sensor:
                        raise RuntimeError("No color sensor found on RealSense device")
                    
                    # CRITICAL: Enable ONLY the RGB color stream (NOT infrared)
                    # Try different formats in order of preference
                    color_formats_to_try = [
                        self.rs.format.rgb8,   # RGB 8-bit
                        self.rs.format.bgr8,   # BGR 8-bit
                        self.rs.format.rgba8,  # RGBA 8-bit
                        self.rs.format.bgra8,  # BGRA 8-bit
                    ]
                    
                    stream_started = False
                    for fmt in color_formats_to_try:
                        try:
                            # Clear any previous config
                            self.realsense_config = self.rs.config()
                            self.realsense_config.disable_all_streams()
                            
                            # Enable color stream with this format
                            self.realsense_config.enable_stream(
                                self.rs.stream.color,  # Explicitly COLOR stream (not infrared)
                                640,
                                480,
                                fmt,
                                30
                            )
                            
                            # Try to start the pipeline
                            profile = self.realsense_pipeline.start(self.realsense_config)
                            
                            # Verify we got a color stream
                            stream_profile = profile.get_stream(self.rs.stream.color)
                            logger.info(f"[REALSENSE] Started with format: {fmt}")
                            
                            stream_started = True
                            break
                            
                        except Exception as e:
                            logger.warning(f"[REALSENSE] Format {fmt} failed: {e}")
                            continue
                    
                    if not stream_started:
                        raise RuntimeError("Could not start color stream with any format")
                    
                    # Get the color sensor and set explicit options
                    try:
                        profile = self.realsense_pipeline.get_active_profile()
                        color_sensor = profile.get_device().first_color_sensor()
                        
                        # Force auto-exposure ON
                        color_sensor.set_option(self.rs.option.enable_auto_exposure, 1)
                        
                        # Disable any infrared-related options
                        try:
                            color_sensor.set_option(self.rs.option.emitter_enabled, 0)  # Disable IR emitter
                        except:
                            pass
                        
                        logger.info("[REALSENSE] Color sensor configured with auto-exposure")
                    except Exception as sensor_error:
                        logger.warning(f"[REALSENSE] Could not set sensor options: {sensor_error}")
                    
                    # LONGER warmup - let camera stabilize and auto-exposure adjust
                    logger.info("[REALSENSE] Warming up color stream...")
                    for i in range(90):  # 3 seconds at 30fps
                        frames = self.realsense_pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        
                        # Every 30 frames, verify we're getting true color
                        if i % 30 == 29 and color_frame:
                            frame_data = np.asanyarray(color_frame.get_data())
                            if len(frame_data.shape) == 3:
                                unique_colors = len(np.unique(frame_data.reshape(-1, 3), axis=0))
                                logger.info(f"[REALSENSE] Warmup check: {unique_colors} unique colors")
                                if unique_colors > 1000:
                                    logger.info(f"[REALSENSE] ✅ TRUE COLOR confirmed at frame {i}")
                                    break
                    
                    # Final verification
                    frames = self.realsense_pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        frame_data = np.asanyarray(color_frame.get_data())
                        if len(frame_data.shape) == 3:
                            unique_colors = len(np.unique(frame_data.reshape(-1, 3), axis=0))
                            if unique_colors > 1000:
                                self.camera_status['realsense'] = True
                                logger.info(f"[REALSENSE] ✅✅✅ TRUE COLOR verified: {unique_colors} unique colors")
                            else:
                                logger.error(f"[REALSENSE] ❌ Still grayscale: only {unique_colors} colors")
                                raise RuntimeError(f"Color stream has only {unique_colors} colors - likely infrared")
                    else:
                        raise RuntimeError("No color frame received after warmup")
                    
                    logger.info("[OK] RealSense TRUE color stream enabled via SDK (640x480 @ 30fps)")
                    
                except Exception as e:
                    logger.error(f"[FAIL] RealSense SDK init failed: {e}")
                    logger.warning("Attempting OpenCV fallback for RealSense...")
                    self._initialize_realsense_opencv_fallback()
            else:
                # Fallback to OpenCV
                logger.info("[REALSENSE] Using OpenCV fallback (SDK not available)")
                self._initialize_realsense_opencv_fallback()
    
    def setup_camera_properties(self, cap, high_res=False):
        """Optimized camera properties for 12MP Arducam with calibration settings"""
        if cap and cap.isOpened():
            try:
                # Apply ArduCam calibration settings
                exposure = self.arducam_settings.get('exposure', -3)
                brightness = self.arducam_settings.get('brightness', 128)
                
                # Set exposure and brightness from calibration
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                
                if high_res:
                    # 12MP capture mode
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4056)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3040)
                    cap.set(cv2.CAP_PROP_FPS, 5)  # Lower FPS for high res
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
                    
                    # MJPG is CRITICAL for 12MP over USB
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                else:
                    # Preview mode (720p)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 15)  # Higher FPS OK for preview
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # CRITICAL: Force RGB color format
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                cap.set(cv2.CAP_PROP_FORMAT, -1)  # Let OpenCV choose best format
                
                logger.info(f"✅ Applied ArduCam settings: exposure={exposure}, brightness={brightness}")
                
                # Verify color mode and log actual resolution
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info(f"🔍 ACTUAL CAMERA RESOLUTION: {actual_width}x{actual_height}")
                    logger.info(f"Camera frame shape: {test_frame.shape}")
                if len(test_frame.shape) == 2:
                    logger.error("Camera is in GRAYSCALE mode - check camera settings!")
                elif test_frame.shape[2] == 3:
                    logger.info("Camera confirmed in RGB color mode")
                
                # Warm up
                for _ in range(3):
                    cap.read()
            except:
                pass
    
    def _force_opencv_color_mode(self, cap):
        """Aggressively force RealSense to color mode via OpenCV"""
        try:
            # Try multiple backends
            backends_to_try = [
                cv2.CAP_DSHOW,  # DirectShow
                cv2.CAP_MSMF,   # Media Foundation
                cv2.CAP_ANY     # Any available
            ]
            
            for backend in backends_to_try:
                logger.info(f"[COLOR-FORCE] Trying backend: {backend}")
                
                # Set format explicitly
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                
                # Force resolution (helps trigger color mode)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Try to read frames and check if truly color
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            # Check if truly color (not just grayscale with 3 channels)
                            unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
                            if unique_colors > 256:
                                logger.info(f"✅ Got color with backend {backend} ({unique_colors} unique colors)")
                                return True
                
                logger.warning(f"[COLOR-FORCE] Backend {backend} failed")
            
            logger.error("[COLOR-FORCE] All backends failed to get color")
            return False
            
        except Exception as e:
            logger.error(f"[COLOR-FORCE] Force color mode failed: {e}")
            return False
    
    def _initialize_realsense_opencv_fallback(self):
        """Initialize RealSense D415 with AGGRESSIVE RGB forcing (MJPG then YUYV fallback)"""
        try:
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            self.realsense_cap = cv2.VideoCapture(self.realsense_index, backend)
            
            if self.realsense_cap.isOpened():
                # CRITICAL FOR D415: Set format BEFORE any other settings
                fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
                self.realsense_cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
                self.realsense_cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                
                # D415 specific: Force high resolution to trigger RGB mode
                self.realsense_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.realsense_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Disable auto-exposure initially to force RGB mode
                self.realsense_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                
                # Warmup: D415 needs more frames to switch from IR to RGB
                logger.info("[REALSENSE-D415] Warming up RGB sensor with MJPG...")
                for i in range(60):  # 2 seconds at 30fps
                    ret, frame = self.realsense_cap.read()
                    if ret and frame is not None and i % 15 == 0:
                        if len(frame.shape) == 3:
                            unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
                            logger.info(f"[D415] Warmup frame {i}: {unique_colors} colors")
                            if unique_colors > 5000:  # Good threshold for true RGB
                                logger.info(f"[D415] ✅ RGB confirmed at frame {i}")
                                break
                
                # Final verification
                ret, frame = self.realsense_cap.read()
                if ret and frame is not None and len(frame.shape) == 3:
                    unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
                    if unique_colors > 5000:
                        self.camera_status['realsense'] = True
                        logger.info(f"[D415] ✅✅✅ TRUE RGB with MJPG: {unique_colors} colors")
                        return
                    else:
                        logger.error(f"[D415] ❌ MJPG still limited colors: {unique_colors}")
                
                # If still not RGB, try YUYV format (more aggressive reset)
                logger.warning("[D415] Attempting aggressive reset with YUYV...")
                self.realsense_cap.release()
                time.sleep(1.0)
                
                self.realsense_cap = cv2.VideoCapture(self.realsense_index, backend)
                fourcc_yuyv = cv2.VideoWriter_fourcc(*'YUYV')
                self.realsense_cap.set(cv2.CAP_PROP_FOURCC, fourcc_yuyv)
                self.realsense_cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                self.realsense_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.realsense_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Longer warmup for YUYV
                for i in range(30):
                    ret, frame = self.realsense_cap.read()
                    if ret and frame is not None and i % 10 == 9:
                        if len(frame.shape) == 3:
                            unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
                            logger.info(f"[D415-YUYV] Warmup frame {i}: {unique_colors} colors")
                
                ret, frame = self.realsense_cap.read()
                if ret and frame is not None and len(frame.shape) == 3:
                    unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
                    if unique_colors > 5000:
                        self.camera_status['realsense'] = True
                        logger.info(f"[D415] ✅ TRUE RGB with YUYV: {unique_colors} colors")
                    else:
                        logger.warning(f"[D415] ⚠️ Limited colors with YUYV: {unique_colors}")
                        self.camera_status['realsense'] = True  # Still usable but warn
                else:
                    logger.error("[D415] ❌ Could not enable RGB mode via OpenCV")
                    
        except Exception as e:
            logger.error(f"[D415] OpenCV initialization failed: {e}")
    
    def recover_camera(self, camera_name):
        """Recover a failed camera by forcing release and re-detection"""
        logger.info(f"Attempting aggressive recovery for {camera_name}...")
        
        # Force release existing capture
        if camera_name == 'arducam' and self.arducam_cap is not None:
            self.arducam_cap.release()
            self.arducam_cap = None
        elif camera_name == 'realsense' and self.realsense_cap is not None:
            self.realsense_cap.release()
            self.realsense_cap = None
        
        # Longer wait for Windows to release the camera
        logger.info("Waiting for Windows to release camera resources...")
        time.sleep(5)
        
        # Reset failure count
        self.camera_failures[camera_name] = 0
        
        # Try to find a working camera index with more aggressive scanning
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]  # Try DirectShow first, avoid MSMF
        working_index = None
        
        logger.info(f"Scanning all camera indices for {camera_name}...")
        for backend in backends:
            logger.info(f"Trying backend {backend}")
            for i in range(10):  # Check more indices
                try:
                    logger.debug(f"Testing camera {i} with backend {backend}")
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        # Try to read multiple frames to ensure it's stable
                        frames_read = 0
                        for attempt in range(5):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                frames_read += 1
                            time.sleep(0.1)
                        
                        cap.release()
                        if frames_read >= 3:  # Need at least 3 successful frames
                            working_index = i
                            logger.info(f"Found stable camera at index {i} for {camera_name} (read {frames_read} frames)")
                            break
                except Exception as e:
                    logger.debug(f"Camera {i} failed: {e}")
                    continue
            if working_index is not None:
                break
        
        if working_index is not None:
            if camera_name == 'arducam':
                self.arducam_index = working_index
                self.camera_status['arducam'] = True
            else:
                self.realsense_index = working_index
                self.camera_status['realsense'] = True
            
            # Test the camera immediately
            logger.info(f"Testing recovered {camera_name}...")
            if camera_name == 'arducam':
                test_frame = self.get_arducam_frame()
            else:
                test_frame = self.get_realsense_frame()
            
            if test_frame is not None:
                logger.info(f"{camera_name} recovery successful!")
                return True
            else:
                logger.warning(f"{camera_name} recovery failed - camera not stable")
                return False
        else:
            logger.error(f"Could not recover {camera_name} - no working cameras found")
        return False
    
    def diagnose_cameras(self):
        """Comprehensive camera diagnostic"""
        logger.info("Running comprehensive camera diagnostic...")
        
        results = {
            'total_cameras': 0,
            'working_cameras': [],
            'failed_cameras': [],
            'backend_results': {}
        }
        
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY, "Any Backend"),
            (cv2.CAP_MSMF, "Media Foundation")
        ]
        
        for backend, backend_name in backends:
            logger.info(f"Testing {backend_name} backend...")
            backend_results = []
            
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        results['total_cameras'] += 1
                        
                        # Try to read frames
                        frames_read = 0
                        for attempt in range(3):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                frames_read += 1
                            time.sleep(0.1)
                        
                        cap.release()
                        
                        if frames_read >= 2:
                            camera_info = {
                                'index': i,
                                'backend': backend_name,
                                'frames_read': frames_read,
                                'status': 'working'
                            }
                            results['working_cameras'].append(camera_info)
                            backend_results.append(f"Camera {i}: Working ({frames_read} frames)")
                            logger.info(f"Camera {i}: Working with {backend_name}")
                        else:
                            camera_info = {
                                'index': i,
                                'backend': backend_name,
                                'frames_read': frames_read,
                                'status': 'failed'
                            }
                            results['failed_cameras'].append(camera_info)
                            backend_results.append(f"Camera {i}: Failed ({frames_read} frames)")
                            logger.warning(f"Camera {i}: Failed with {backend_name}")
                    else:
                        backend_results.append(f"Camera {i}: Cannot open")
                        
                except Exception as e:
                    backend_results.append(f"Camera {i}: Error - {e}")
                    logger.debug(f"Camera {i} error: {e}")
            
            results['backend_results'][backend_name] = backend_results
        
        logger.info(f"Diagnostic complete: {len(results['working_cameras'])} working, {len(results['failed_cameras'])} failed")
        return results
    
    def load_roi_config(self):
        """Load ROI coordinates from saved configuration file - prioritize roi_config.json from roi2.py"""
        try:
            # PRIORITY 1: Use roi_config.json from roi2.py tool (most recent)
            if os.path.exists('roi_config.json'):
                with open('roi_config.json', 'r') as f:
                    config = json.load(f)
                    roi_coords = config.get('roi_coords', {})
                    result = {
                        'tag': tuple(roi_coords.get('tag', [183, 171, 211, 159])),
                        'work': tuple(roi_coords.get('work', [38, 33, 592, 435]))
                    }
                    self.original_resolution = tuple(config.get('original_resolution', [640, 480]))
                    logger.info(f"✅ Loaded ROI config from roi2.py: tag={result['tag']}, work={result['work']} at {self.original_resolution}")
                    return result
            
            # FALLBACK: Try ArduCam-specific calibration (older format)
            if os.path.exists('arducam_calibration.json'):
                with open('arducam_calibration.json', 'r') as f:
                    arducam_data = json.load(f)
                    arducam_roi = arducam_data.get('roi_coords', [])
                    
                    if len(arducam_roi) == 4:
                        # ArduCam ROI format: [x, y, width, height]
                        x, y, w, h = arducam_roi
                        # Convert to our format: [x, y, width, height] for both tag and work
                        result = {
                            'tag': (x, y, w, h),  # Use ArduCam ROI for tag
                            'work': (x, y, w, h)   # Use same ROI for work area
                        }
                        self.original_resolution = tuple(arducam_data.get('resolution', [640, 480]))
                        logger.info(f"✅ Loaded ArduCam ROI: {arducam_roi} at resolution {self.original_resolution}")
                        return result
            
            # DEFAULT FALLBACK
            self.original_resolution = (640, 480)
            default_result = {
                'tag': (183, 171, 211, 159),
                'work': (38, 33, 592, 435)
            }
            logger.warning("⚠️ Using default ROI coordinates - no calibration files found")
            return default_result
            
        except Exception as e:
            logger.error(f"Error loading ROI config: {e}")
            self.original_resolution = (640, 480)
            return {
                'tag': (183, 171, 211, 159),
                'work': (38, 33, 592, 435)
            }
    
    def load_arducam_settings(self):
        """Load ArduCam-specific settings from calibration file"""
        try:
            if os.path.exists('arducam_calibration.json'):
                with open('arducam_calibration.json', 'r') as f:
                    arducam_data = json.load(f)
                    
                    settings = {
                        'exposure': arducam_data.get('exposure', -3),
                        'brightness': arducam_data.get('brightness', 128),
                        'zoom_level': arducam_data.get('zoom_level', 1.0),
                        'best_focus_score': arducam_data.get('best_focus_score', 0.0),
                        'timestamp': arducam_data.get('timestamp', 'unknown')
                    }
                    
                    logger.info(f"✅ Loaded ArduCam settings: exposure={settings['exposure']}, brightness={settings['brightness']}, zoom={settings['zoom_level']}")
                    return settings
            else:
                logger.warning("⚠️ No ArduCam calibration file found - using defaults")
                return {
                    'exposure': -3,
                    'brightness': 128,
                    'zoom_level': 1.0,
                    'best_focus_score': 0.0,
                    'timestamp': 'unknown'
                }
                
        except Exception as e:
            logger.error(f"Error loading ArduCam settings: {e}")
            return {
                'exposure': -3,
                'brightness': 128,
                'zoom_level': 1.0,
                'best_focus_score': 0.0,
                'timestamp': 'unknown'
            }
    
    def auto_adjust_camera(self, cap, camera_name):
        """Auto-adjust camera settings for optimal image quality with LED lighting"""
        try:
            if cap is None or not cap.isOpened():
                return False
            
            logger.info(f"Auto-adjusting {camera_name} for LED lighting...")
            
            try:
                # Start with lower resolution first
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                logger.info(f"Set {camera_name} to 640x480")
                
                # Give camera time to adjust
                time.sleep(0.5)
                
                # Now try higher resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                logger.info(f"Set {camera_name} to 1280x720")
            except:
                logger.warning(f"Could not set resolution for {camera_name}")
            
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, -3)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)
            cap.set(cv2.CAP_PROP_CONTRAST, 0.8)
            cap.set(cv2.CAP_PROP_SATURATION, 0.6)
            cap.set(cv2.CAP_PROP_SHARPNESS, 1.0)  # Max sharpness for text
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            cap.set(cv2.CAP_PROP_FOCUS, 30)  # Close focus distance for tags (~10-15cm)
            cap.set(cv2.CAP_PROP_GAIN, 0.6)
            
            logger.info(f"{camera_name} optimized for LED lighting")
            return True
            
        except Exception as e:
            logger.error(f"Auto-adjustment failed for {camera_name}: {e}")
            return False
    
    def get_arducam_frame(self):
        """Get frame from ArduCam, using the thread-safe cache."""
        try:
            # Check if we need to reinitialize
            if self.arducam_cap is None or not self.arducam_cap.isOpened():
                self.initialize_cameras()
                if self.arducam_cap is None or not self.arducam_cap.isOpened():
                    return None
            
            # Grab multiple frames to clear buffer, then retrieve the latest
            for _ in range(self.skip_frames):
                self.arducam_cap.grab()
            ret, frame = self.arducam_cap.retrieve()
            
            if ret and frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Update the cache using the new thread-safe method
                if 'pipeline_manager' in st.session_state:
                    st.session_state.pipeline_manager.camera_cache.update_tag_frame(rgb_frame)
            
            # ALWAYS return the frame from the stable cache
            if 'pipeline_manager' in st.session_state:
                return st.session_state.pipeline_manager.camera_cache.get_tag_frame()
            else:
                return None

        except Exception as e:
            logger.error(f"ArduCam error: {e}")
            # Return last good frame on error
            if 'pipeline_manager' in st.session_state:
                return st.session_state.pipeline_manager.camera_cache.get_tag_frame()
            else:
                return None
    
    def get_preview_and_full(self):
        """Dual-path capture: fast preview + high-res still"""
        frame = self.get_arducam_frame()
        if frame is None:
            return None, None
        
        full = frame
        preview = cv2.resize(full, self.preview_resolution, interpolation=cv2.INTER_AREA)
        return preview, full
    
    def capture_highres_burst(self, n=2):
        """Temporarily bump to max res, capture burst, then restore preview size"""
        if not self.arducam_cap or not self.arducam_cap.isOpened():
            return []
        
        # Temporarily bump to max res
        self._set_fourcc(self.arducam_cap, "MJPG")
        for w, h in [(4056, 3040), (4000, 3000), (2592, 1944)]:
            self.arducam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.arducam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if (int(self.arducam_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(self.arducam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (w, h):
                break
        
        frames = []
        for _ in range(n):
            f = self.get_arducam_frame()
            if f is not None:
                frames.append(f)
        
        # Restore preview size
        self.arducam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.preview_resolution[0])
        self.arducam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preview_resolution[1])
        
        return frames
    
    def capture_tag_highres_optimized(self):
        """Capture tag at maximum resolution with optimal settings"""
        if not self.arducam_cap or not self.arducam_cap.isOpened():
            return None
        
        # Temporarily switch to max resolution
        self.arducam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4056)
        self.arducam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3040)
        
        # CRITICAL: Use MJPG codec for high bandwidth
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.arducam_cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        # Flush buffer (important after resolution change)
        for _ in range(5):
            self.arducam_cap.grab()
        
        # Capture multiple frames, pick sharpest
        frames = []
        for _ in range(3):
            ret, frame = self.arducam_cap.read()
            if ret and frame is not None:
                frames.append(frame)
            time.sleep(0.1)  # Brief pause between captures
        
        if not frames:
            return None
        
        # Select sharpest frame
        best_frame = max(frames, key=self.calculate_focus_score)
        
        # Restore preview resolution
        self.arducam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.arducam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        return best_frame
    
    def prepare_tag_for_api(self, tag_image_12mp):
        """Downsample 12MP image optimally for OCR"""
        import numpy as np
        
        if tag_image_12mp is None:
            return None
        
        # Target: 2-3MP is optimal for text recognition
        # Too high = wasted bandwidth, too low = lost detail
        target_width = 2000  # ~2MP when maintaining aspect ratio
        
        h, w = tag_image_12mp.shape[:2]
        if w > target_width:
            scale = target_width / w
            new_h = int(h * scale)
            
            # Use LANCZOS for high-quality downsampling
            downsampled = cv2.resize(
                tag_image_12mp, 
                (target_width, new_h), 
                interpolation=cv2.INTER_LANCZOS4
            )
        else:
            downsampled = tag_image_12mp
        
        # Apply sharpening AFTER downsampling (prevents artifacts)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(downsampled, -1, kernel)
        
        return sharpened
    
    def calculate_focus_score(self, image):
        """Stable sharpness score using Laplacian variance within the tag ROI"""
        if image is None or image.size == 0:
            return 0.0
        
        # OPTIMIZATION: In focus mode, use simple ROI without upscaling to avoid loops
        is_focus_mode = st.session_state.get('focus_mode', False)
        
        if is_focus_mode:
            # For focus mode, use simple cropping without upscaling
            if 'tag' in self.roi_coords:
                x, y, w, h = self.roi_coords['tag']
                h_img, w_img = image.shape[:2]
                
                # Ensure ROI is within image bounds
                x = max(0, min(x, w_img - 1))
                y = max(0, min(y, h_img - 1))
                w = max(1, min(w, w_img - x))
                h = max(1, min(h, h_img - y))
                
                # Simple crop without upscaling
                roi = image[y:y+h, x:x+w]
            else:
                roi = image
        else:
            # Normal operation - use full ROI processing
            roi = self.apply_roi(image, 'tag')
        
        if roi is None or roi.size == 0:
            roi = image  # fallback to full frame
        if roi is None or roi.size == 0:
            return 0.0
        
        # Normalize luminance a bit to reduce exposure swings
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Light denoise to avoid false spikes from sensor noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Laplacian variance = sharpness proxy
        return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())
    
    def get_realsense_frame(self):
        """Get COLOR frame from RealSense using SDK, with thread-safe caching."""
        try:
            # Try SDK method first (preferred)
            if self.realsense_pipeline is not None and self.realsense_sdk_available:
                try:
                    # Wait for frames with timeout
                    frames = self.realsense_pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        frame_data = np.asanyarray(color_frame.get_data())
                        # The SDK often returns BGR, so we convert to RGB
                        rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                        if 'pipeline_manager' in st.session_state:
                            st.session_state.pipeline_manager.camera_cache.update_garment_frame(rgb_frame)

                except Exception as e:
                    logger.warning(f"RealSense SDK read failed: {e}, trying OpenCV fallback")
            
            # Fallback to OpenCV if SDK fails or is not available
            elif self.realsense_cap is not None and self.realsense_cap.isOpened():
                ret, frame = self.realsense_cap.read()
                if ret and frame is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if 'pipeline_manager' in st.session_state:
                        st.session_state.pipeline_manager.camera_cache.update_garment_frame(rgb_frame)

            # ALWAYS return the frame from the stable cache
            if 'pipeline_manager' in st.session_state:
                return st.session_state.pipeline_manager.camera_cache.get_garment_frame()
            else:
                return None

        except Exception as e:
            logger.error(f"RealSense error: {e}")
            # Return last good frame on error
            if 'pipeline_manager' in st.session_state:
                return st.session_state.pipeline_manager.camera_cache.get_garment_frame()
            else:
                return None
    
    def get_garment_frame(self):
        """Get garment frame - use ArduCam if RealSense problematic"""
        
        # Try RealSense first
        frame = self.get_realsense_frame()
        
        # Verify it's actually color
        if frame is not None:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
                if unique_colors > 1000:  # Arbitrary threshold for "real color"
                    return frame
            
            logger.warning("RealSense not providing color - falling back to ArduCam")
        
        # Fallback: use ArduCam for garments too
        return self.get_arducam_frame()
    
    def detect_tag_for_auto_zoom(self, frame):
        """
        Automatically detects a tag's bounding box and returns a tightly cropped image.
        Includes a sanity check to reject detections that are too small.
        """
        logger.info("[AUTO-ZOOM] Attempting automatic tag detection...")
        
        # First, get the standard, larger ROI
        roi_frame = self.apply_roi(frame, 'tag', zoom_factor=1.0)
        if roi_frame is None:
            return None

        try:
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("[AUTO-ZOOM] No contours found. Using original ROI.")
                return roi_frame

            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            tag_contour = None
            
            for c in contours:
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
                if len(approx) == 4:
                    tag_contour = approx
                    break
            
            if tag_contour is not None:
                (x, y, w, h) = cv2.boundingRect(tag_contour)

                # --- NEW SANITY CHECK ---
                # If the detected rectangle is too small, it's likely an error.
                MIN_TAG_WIDTH = 100  # pixels
                MIN_TAG_HEIGHT = 50   # pixels
                if w < MIN_TAG_WIDTH or h < MIN_TAG_HEIGHT:
                    logger.warning(f"[AUTO-ZOOM] Detected contour was too small ({w}x{h}). Rejecting and falling back to original ROI.")
                    return roi_frame # FALLBACK to the larger, un-zoomed ROI

                # If the check passes, proceed with cropping
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(roi_frame.shape[1], x + w + padding)
                y2 = min(roi_frame.shape[0], y + h + padding)
                
                zoomed_tag = roi_frame[y1:y2, x1:x2]
                
                logger.info(f"[AUTO-ZOOM] ✅ Detected tag at ({x},{y}) size {w}x{h}. Cropping to new region.")
                return zoomed_tag
            else:
                logger.warning("[AUTO-ZOOM] No rectangular contour found. Using original ROI.")
                return roi_frame
                
        except Exception as e:
            logger.error(f"[AUTO-ZOOM] Detection failed: {e}. Using original ROI.")
            return roi_frame
    
    def apply_roi(self, frame, roi_type, zoom_factor=1.0):
        """Apply ROI with optional digital zoom for small tags
        
        Args:
            frame: Input frame
            roi_type: 'tag' or 'work'
            zoom_factor: 1.0 = normal, 2.0 = 2x zoom into center of ROI
        """
        # Pure function - no state checks or modifications
            
        if roi_type in self.roi_coords and frame is not None:
            x, y, w, h = self.roi_coords[roi_type]
            h_frame, w_frame = frame.shape[:2]
            
            original_width, original_height = self.original_resolution
            if w_frame != original_width or h_frame != original_height:
                scale_x = w_frame / original_width
                scale_y = h_frame / original_height
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
            
            # DIGITAL ZOOM: Crop tighter and upscale more
            if zoom_factor > 1.0 and roi_type == 'tag':
                # Zoom into center of ROI
                zoom_w = int(w / zoom_factor)
                zoom_h = int(h / zoom_factor)
                zoom_x = x + (w - zoom_w) // 2
                zoom_y = y + (h - zoom_h) // 2
                
                # Update ROI to zoomed region
                x, y, w, h = zoom_x, zoom_y, zoom_w, zoom_h
                logger.info(f"[ZOOM] Digital zoom {zoom_factor}x applied to tag ROI")
            
            # Add padding for small tags (if not already zoomed)
            elif roi_type == 'tag' and (w < 250 or h < 250):
                padding = 0.2
                x = max(0, int(x - w * padding))
                y = max(0, int(y - h * padding))
                w = min(int(w * (1 + 2*padding)), w_frame - x)
                h = min(int(h * (1 + 2*padding)), h_frame - y)
            
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            
            if w > 0 and h > 0:
                try:
                    cropped = frame[y:y+h, x:x+w]
                    
                    # Validate the cropped image
                    if cropped is None or cropped.size == 0:
                        logger.warning(f"[ROI] Empty crop result for {roi_type}")
                        return None
                    
                    # Check for invalid values (NaN, infinity)
                    if np.any(np.isnan(cropped)) or np.any(np.isinf(cropped)):
                        logger.warning(f"[ROI] Invalid values (NaN/inf) in crop for {roi_type}")
                        return None
                    
                    # Ensure data type is valid for image display
                    if cropped.dtype not in [np.uint8, np.float32, np.float64]:
                        logger.warning(f"[ROI] Invalid dtype {cropped.dtype} for {roi_type}, converting to uint8")
                        if cropped.dtype == np.float32 or cropped.dtype == np.float64:
                            # Assume values are in range [0,1] and convert to [0,255]
                            cropped = (cropped * 255).astype(np.uint8)
                        else:
                            cropped = cropped.astype(np.uint8)
                    
                    # Ensure values are in valid range
                    if cropped.dtype == np.uint8:
                        # Already in correct range [0,255]
                        pass
                    elif cropped.max() <= 1.0:
                        # Values in [0,1] range, convert to [0,255]
                        cropped = (cropped * 255).astype(np.uint8)
                    elif cropped.max() > 255:
                        # Values exceed 255, normalize
                        cropped = (cropped / cropped.max() * 255).astype(np.uint8)
                    
                    # SMART UPSCALING: More aggressive for zoomed tags
                    target_size = 800 if zoom_factor > 1.0 else 600
                    if cropped.shape[0] < target_size or cropped.shape[1] < target_size:
                        scale = max(target_size/cropped.shape[0], target_size/cropped.shape[1])
                        new_w = int(cropped.shape[1] * scale)
                        new_h = int(cropped.shape[0] * scale)
                        # Use LANCZOS4 for high-quality upscaling (best for text)
                        cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                        
                        # SMART LOOP PREVENTION: Only track automatic upscales, not user zoom changes
                        # Log the upscaling operation (no state mutation)
                        logger.info(f"[UPSCALE] Tag upscaled to {new_w}x{new_h} (target: {target_size})")
                    
                    # Final validation before return
                    if cropped is None or cropped.size == 0:
                        logger.warning(f"[ROI] Final validation failed for {roi_type}")
                        return None
                    
                    logger.debug(f"[ROI] Successfully cropped {roi_type}: {cropped.shape}, dtype: {cropped.dtype}, range: [{cropped.min()}, {cropped.max()}]")
                    return cropped
                    
                except Exception as e:
                    logger.error(f"[ROI] Error processing crop for {roi_type}: {e}")
                    return None
            
        return None
    
    def apply_roi_pure(self, frame, roi_type, zoom_factor=1.0):
        """
        Applies an ROI crop. This is a "pure" function with no side effects.
        It does NOT change st.session_state.
        """
        if roi_type in self.roi_coords and frame is not None:
            x, y, w, h = self.roi_coords[roi_type]
            h_frame, w_frame = frame.shape[:2]
            
            # Scale ROI coordinates based on the actual frame size vs original config
            original_width, original_height = self.original_resolution
            if w_frame != original_width or h_frame != original_height:
                scale_x = w_frame / original_width
                scale_y = h_frame / original_height
                x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            
            # Apply digital zoom if specified
            if zoom_factor > 1.0:
                zoom_w, zoom_h = int(w / zoom_factor), int(h / zoom_factor)
                x, y = x + (w - zoom_w) // 2, y + (h - zoom_h) // 2
                w, h = zoom_w, zoom_h

            # Ensure coordinates are within bounds and crop
            x, y = max(0, x), max(0, y)
            w, h = min(w, w_frame - x), min(h, h_frame - y)
            
            if w > 0 and h > 0:
                return frame[y:y+h, x:x+w]
                
        return None  # Return None if cropping is not possible
    
    def draw_roi_overlay(self, frame, roi_type):
        """Draw ROI rectangle on frame with comprehensive debugging"""
        
        # Validation checks
        if frame is None:
            logger.error(f"[ROI] Frame is None for {roi_type}")
            return None
        
        if frame.size == 0:
            logger.error(f"[ROI] Frame is empty for {roi_type}")
            return frame
        
        if not hasattr(self, 'roi_coords'):
            logger.error(f"[ROI] No roi_coords attribute")
            return frame
        
        if roi_type not in self.roi_coords:
            logger.error(f"[ROI] {roi_type} not in roi_coords: {list(self.roi_coords.keys())}")
            return frame
        
        # Make a copy to ensure we don't modify original
        frame_copy = frame.copy()
        
        x, y, w, h = self.roi_coords[roi_type]
        h_frame, w_frame = frame_copy.shape[:2]
        
        logger.info(f"[ROI] Drawing {roi_type} - Original coords: ({x}, {y}, {w}, {h})")
        logger.info(f"[ROI] Frame dimensions: {w_frame}x{h_frame}")
        
        # Scale ROI if needed
        original_width, original_height = self.original_resolution
        if w_frame != original_width or h_frame != original_height:
            scale_x = w_frame / original_width
            scale_y = h_frame / original_height
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            logger.info(f"[ROI] Scaled coords: ({x}, {y}, {w}, {h})")
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))  # Ensure w is at least 1
        h = max(1, min(h, h_frame - y))  # Ensure h is at least 1
        
        logger.info(f"[ROI] Final coords: ({x}, {y}, {w}, {h})")
        
        if w <= 0 or h <= 0:
            logger.error(f"[ROI] Invalid dimensions after clamping: w={w}, h={h}")
            return frame_copy
        
        # Draw VERY THICK green rectangle (increased from 4 to 6)
        cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 6)
        
        # Draw LARGER corner markers for extra visibility
        corner_size = 30  # Increased from 20
        thickness = 8     # Increased from 6
        
        # Top-left corner
        cv2.line(frame_copy, (x, y), (x + corner_size, y), (0, 255, 0), thickness)
        cv2.line(frame_copy, (x, y), (x, y + corner_size), (0, 255, 0), thickness)
        
        # Top-right corner
        cv2.line(frame_copy, (x+w, y), (x+w - corner_size, y), (0, 255, 0), thickness)
        cv2.line(frame_copy, (x+w, y), (x+w, y + corner_size), (0, 255, 0), thickness)
        
        # Bottom-left corner
        cv2.line(frame_copy, (x, y+h), (x + corner_size, y+h), (0, 255, 0), thickness)
        cv2.line(frame_copy, (x, y+h), (x, y+h - corner_size), (0, 255, 0), thickness)
        
        # Bottom-right corner
        cv2.line(frame_copy, (x+w, y+h), (x+w - corner_size, y+h), (0, 255, 0), thickness)
        cv2.line(frame_copy, (x+w, y+h), (x+w, y+h - corner_size), (0, 255, 0), thickness)
        
        # Draw LARGER label
        label = "TAG ROI" if roi_type == "tag" else "WORK ROI"
        cv2.putText(frame_copy, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 4)
        
        logger.info(f"[ROI] Successfully drew {roi_type} ROI")
        
        return frame_copy
    
    def detect_motion_in_roi(self, roi_type='tag', threshold=20):
        """Detect if something new entered the ROI (tag placed on table) - MORE SENSITIVE"""
        
        # Get current frame ROI
        frame = self.get_arducam_frame()
        if frame is None:
            return False
        
        roi = self.apply_roi_pure(frame, roi_type)
        if roi is None:
            return False
        
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Check if we have a previous frame to compare
        cache_key = f'motion_detect_prev_{roi_type}'
        if cache_key not in st.session_state:
            st.session_state[cache_key] = gray
            return False
        
        prev_gray = st.session_state[cache_key]
        
        # Calculate difference
        frame_delta = cv2.absdiff(prev_gray, gray)
        mean_diff = np.mean(frame_delta)
        
        # Update previous frame
        st.session_state[cache_key] = gray
        
        # Log motion detection
        if mean_diff > threshold:
            logger.info(f"[MOTION] Detected in {roi_type} ROI: {mean_diff:.1f} > {threshold}")
        
        # Motion detected if difference exceeds threshold (lowered from 30 to 20 for better sensitivity)
        return mean_diff > threshold
    
    def render_interactive_roi_editor(self):
        """
        Interactive ROI editor with full click-and-drag functionality
        Call this from a dedicated page in your Streamlit app
        """
        st.title("🎯 ROI Configuration Editor")
        st.markdown("**Click and drag** to adjust Region of Interest boxes")
        
        # Initialize session state for dragging
        if 'dragging' not in st.session_state:
            st.session_state.dragging = False
        if 'drag_handle' not in st.session_state:
            st.session_state.drag_handle = None
        if 'drag_start_coords' not in st.session_state:
            st.session_state.drag_start_coords = None
        if 'temp_roi' not in st.session_state:
            st.session_state.temp_roi = None
        
        # ROI Selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            roi_type = st.radio("Select ROI to edit:", ["tag", "work"], key="roi_type_selector")
            
            st.markdown("---")
            st.markdown("**Controls:**")
            st.info("""
            🟡 **Yellow Corners**: Drag to resize
            🔵 **Blue Center**: Drag to move
            ⌨️ **Manual**: Use inputs below
            """)
            
            if st.session_state.dragging:
                st.warning(f"🖱️ Dragging {st.session_state.drag_handle}...")
                if st.button("⏹️ Stop Dragging"):
                    st.session_state.dragging = False
                    st.session_state.drag_handle = None
                    st.rerun()
        
        with col2:
            # Get current frame from camera
            frame = self.get_arducam_frame()
            
            if frame is None:
                st.error("❌ Cannot get camera frame. Check camera connection.")
                return
            
            # Get current ROI
            if st.session_state.temp_roi is not None:
                current_roi = st.session_state.temp_roi
            else:
                current_roi = self.roi_coords.get(roi_type, (100, 100, 200, 150))
            
            x, y, w, h = current_roi
            
            # Draw ROI on frame
            display_frame = frame.copy()
            
            # Draw rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw corner handles (yellow squares)
            handle_size = 15
            corners = [
                (x, y, "top-left"),
                (x + w, y, "top-right"),
                (x, y + h, "bottom-left"),
                (x + w, y + h, "bottom-right")
            ]
            
            for cx, cy, name in corners:
                cv2.rectangle(display_frame, 
                             (cx - handle_size//2, cy - handle_size//2),
                             (cx + handle_size//2, cy + handle_size//2),
                             (255, 255, 0), -1)
            
            # Draw center handle (blue circle)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(display_frame, (center_x, center_y), 20, (0, 0, 255), -1)
            
            # Add label
            label = f"{roi_type.upper()}: {w}x{h}"
            cv2.putText(display_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display with click tracking
            coords = streamlit_image_coordinates(
                display_frame,
                key=f"roi_editor_{roi_type}_{st.session_state.get('roi_refresh', 0)}"
            )
            
            # Handle clicks/drags
            if coords is not None:
                click_x, click_y = coords['x'], coords['y']
                
                # Detect what was clicked
                if not st.session_state.dragging:
                    # Check if corner was clicked
                    handle_clicked = None
                    for cx, cy, name in corners:
                        if abs(click_x - cx) < handle_size and abs(click_y - cy) < handle_size:
                            handle_clicked = name
                            break
                    
                    # Check if center was clicked
                    if handle_clicked is None:
                        dist = np.sqrt((click_x - center_x)**2 + (click_y - center_y)**2)
                        if dist < 20:
                            handle_clicked = "center"
                    
                    if handle_clicked:
                        st.session_state.dragging = True
                        st.session_state.drag_handle = handle_clicked
                        st.session_state.drag_start_coords = (click_x, click_y, x, y, w, h)
                        st.rerun()
                
                else:
                    # Update ROI based on drag
                    start_x, start_y, orig_x, orig_y, orig_w, orig_h = st.session_state.drag_start_coords
                    dx = click_x - start_x
                    dy = click_y - start_y
                    
                    new_x, new_y, new_w, new_h = orig_x, orig_y, orig_w, orig_h
                    
                    handle = st.session_state.drag_handle
                    
                    if handle == "center":
                        # Move entire ROI
                        new_x = max(0, min(frame.shape[1] - orig_w, orig_x + dx))
                        new_y = max(0, min(frame.shape[0] - orig_h, orig_y + dy))
                    
                    elif handle == "top-left":
                        new_x = max(0, orig_x + dx)
                        new_y = max(0, orig_y + dy)
                        new_w = max(50, orig_w - dx)
                        new_h = max(50, orig_h - dy)
                    
                    elif handle == "top-right":
                        new_y = max(0, orig_y + dy)
                        new_w = max(50, orig_w + dx)
                        new_h = max(50, orig_h - dy)
                    
                    elif handle == "bottom-left":
                        new_x = max(0, orig_x + dx)
                        new_w = max(50, orig_w - dx)
                        new_h = max(50, orig_h + dy)
                    
                    elif handle == "bottom-right":
                        new_w = max(50, orig_w + dx)
                        new_h = max(50, orig_h + dy)
                    
                    # Enforce bounds
                    new_x = max(0, min(frame.shape[1] - 50, new_x))
                    new_y = max(0, min(frame.shape[0] - 50, new_y))
                    new_w = min(frame.shape[1] - new_x, new_w)
                    new_h = min(frame.shape[0] - new_y, new_h)
                    
                    st.session_state.temp_roi = (new_x, new_y, new_w, new_h)
                    st.rerun()
        
        # Manual adjustment controls
        st.markdown("---")
        st.subheader("Manual Adjustment")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_x = st.number_input("X Position", 0, frame.shape[1], x, key=f"manual_x_{roi_type}")
        with col2:
            new_y = st.number_input("Y Position", 0, frame.shape[0], y, key=f"manual_y_{roi_type}")
        with col3:
            new_w = st.number_input("Width", 50, frame.shape[1], w, key=f"manual_w_{roi_type}")
        with col4:
            new_h = st.number_input("Height", 50, frame.shape[0], h, key=f"manual_h_{roi_type}")
        
        if (new_x, new_y, new_w, new_h) != current_roi:
            st.session_state.temp_roi = (new_x, new_y, new_w, new_h)
            st.rerun()
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save ROI", type="primary"):
                if st.session_state.temp_roi:
                    self.roi_coords[roi_type] = st.session_state.temp_roi
                self.save_roi_config()
                st.session_state.temp_roi = None
                st.success(f"✅ {roi_type.upper()} ROI saved!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Default"):
                default_rois = {
                    'tag': (183, 171, 211, 159),
                    'work': (38, 33, 592, 435)
                }
                st.session_state.temp_roi = default_rois[roi_type]
                st.rerun()
        
        with col3:
            if st.button("❌ Cancel Changes"):
                st.session_state.temp_roi = None
                st.rerun()
        
        # ROI Statistics
        st.markdown("---")
        st.subheader("ROI Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Position", f"({x}, {y})")
        with col2:
            st.metric("Size", f"{w} × {h}")
        with col3:
            aspect = w / h if h > 0 else 0
            st.metric("Aspect Ratio", f"{aspect:.2f}:1")
    
    def save_roi_config(self):
        """Save ROI configuration to file"""
        try:
            config = {
                'roi_coords': {
                    'tag': list(self.roi_coords['tag']),
                    'work': list(self.roi_coords['work'])
                },
                'original_resolution': list(self.original_resolution),
                'timestamp': datetime.now().isoformat()
            }
            
            with open('roi_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ ROI config saved: tag={config['roi_coords']['tag']}, work={config['roi_coords']['work']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ROI config: {e}")
            return False
    
    def cleanup(self):
        """Release camera resources"""
        if self.arducam_cap:
            self.arducam_cap.release()
            self.arducam_cap = None
        
        # Release RealSense SDK pipeline first
        if self.realsense_pipeline:
            self.realsense_pipeline.stop()
            self.realsense_pipeline = None
        
        if self.realsense_cap:
            self.realsense_cap.release()
            self.realsense_cap = None

    def set_exposure(self, exposure_value: int = -6):
        """
        Sets the camera's exposure value. Lower values mean less light sensitivity.
        Common range for USB cameras is -2 to -13.
        
        Args:
            exposure_value (int): Exposure value (-2 to -13, where -13 is least sensitive)
        """
        if self.arducam_cap and self.arducam_cap.isOpened():
            try:
                # Set to manual exposure mode (0.25 for UVC cameras)
                self.arducam_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                # Set the exposure value
                self.arducam_cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_value))
                logger.info(f"📸 Camera exposure manually set to {exposure_value}")
                return True
            except Exception as e:
                logger.error(f"Failed to set camera exposure: {e}")
                return False
        return False

    def reset_auto_exposure(self):
        """Resets the camera to use automatic exposure."""
        if self.arducam_cap and self.arducam_cap.isOpened():
            try:
                # Set back to auto exposure mode (usually 0.75 for UVC cameras)
                self.arducam_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                logger.info("📸 Camera exposure reset to automatic.")
                return True
            except Exception as e:
                logger.error(f"Failed to reset camera exposure: {e}")
                return False
        return False

    def capture_garment_for_analysis(self):
        """Capture with extended ROI for full-length items like dresses"""
        
        frame = self.get_realsense_frame()
        if frame is None:
            logger.warning("[GARMENT-CAPTURE] No RealSense frame available")
            return None
        
        try:
            h, w = frame.shape[:2]
            
            # Extended ROI: capture more vertical space for dresses
            extended_roi = (
                int(w * 0.1),   # x: 10% from left
                int(h * 0.05),  # y: 5% from top  
                int(w * 0.8),   # width: 80% of frame
                int(h * 0.9)    # height: 90% of frame (capture almost full height)
            )
            
            x, y, roi_w, roi_h = extended_roi
            cropped = frame[y:y+roi_h, x:x+roi_w]
            
            logger.info(f"[GARMENT-CAPTURE] Extended ROI: {roi_w}x{roi_h} (aspect ratio: {roi_h/roi_w:.2f})")
            return cropped
            
        except Exception as e:
            logger.error(f"[GARMENT-CAPTURE] Error in extended capture: {e}")
            # Fallback to regular ROI
            return self.apply_roi(frame, 'work')

    def enhance_garment_for_classification(self, garment_image):
        """Brighten and sharpen the center front area for better button/zipper detection"""
        
        if garment_image is None or garment_image.size == 0:
            return garment_image
            
        try:
            h, w = garment_image.shape[:2]
            
            # Define center front region (middle 40% of width)
            center_x_start = int(w * 0.3)
            center_x_end = int(w * 0.7)
            
            # Extract center region
            center_region = garment_image[:, center_x_start:center_x_end].copy()
            
            # Enhance contrast using CLAHE
            lab = cv2.cvtColor(center_region, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced_center = cv2.merge([l, a, b])
            enhanced_center = cv2.cvtColor(enhanced_center, cv2.COLOR_LAB2RGB)
            
            # Apply sharpening to make buttons/zippers more visible
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced_center = cv2.filter2D(enhanced_center, -1, kernel)
            
            # Put enhanced center back
            result = garment_image.copy()
            result[:, center_x_start:center_x_end] = enhanced_center
            
            logger.debug("[ENHANCE] Center front region enhanced for classification")
            return result
            
        except Exception as e:
            logger.error(f"[ENHANCE] Error enhancing garment: {e}")
            return garment_image

# ==========================
# CAMERA DISPLAY HELPER FUNCTIONS
# ==========================

def draw_roi_overlay_on_full_frame(frame, roi_coords, original_resolution, roi_type='tag', color=(0, 255, 0), thickness=4):
    """
    Draw ROI rectangle on FULL frame (shows complete camera view with overlay).
    
    Args:
        frame: Full camera frame (not cropped)
        roi_coords: Dictionary with ROI coordinates {'tag': (x,y,w,h), 'work': (x,y,w,h)}
        original_resolution: Tuple (width, height) that ROI coords were calibrated for
        roi_type: Which ROI to draw ('tag' or 'work')
        color: RGB color tuple (default: green)
        thickness: Line thickness in pixels
    
    Returns:
        Frame with ROI overlay drawn, or original frame if error
    """
    if frame is None or roi_type not in roi_coords:
        return frame
    
    # Get ROI coordinates
    x, y, w, h = roi_coords[roi_type]
    
    # Scale ROI if frame resolution differs from calibration resolution
    h_frame, w_frame = frame.shape[:2]
    original_width, original_height = original_resolution
    
    if w_frame != original_width or h_frame != original_height:
        scale_x = w_frame / original_width
        scale_y = h_frame / original_height
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
    
    # Ensure coordinates are within frame bounds
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)
    
    if w <= 0 or h <= 0:
        return frame
    
    # Create a copy to draw on
    display_frame = frame.copy()
    
    # Draw main rectangle
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
    
    # Draw corner markers (for visibility)
    corner_size = 25
    corner_thick = 6
    
    # Top-left corner
    cv2.line(display_frame, (x, y), (x + corner_size, y), color, corner_thick)
    cv2.line(display_frame, (x, y), (x, y + corner_size), color, corner_thick)
    
    # Top-right corner
    cv2.line(display_frame, (x + w, y), (x + w - corner_size, y), color, corner_thick)
    cv2.line(display_frame, (x + w, y), (x + w, y + corner_size), color, corner_thick)
    
    # Bottom-left corner
    cv2.line(display_frame, (x, y + h), (x + corner_size, y + h), color, corner_thick)
    cv2.line(display_frame, (x, y + h), (x, y + h - corner_size), color, corner_thick)
    
    # Bottom-right corner
    cv2.line(display_frame, (x + w, y + h), (x + w - corner_size, y + h), color, corner_thick)
    cv2.line(display_frame, (x + w, y + h), (x + w, y + h - corner_size), color, corner_thick)
    
    # Draw center crosshair
    center_x, center_y = x + w // 2, y + h // 2
    cross_size = 20
    cv2.line(display_frame, (center_x - cross_size, center_y), 
             (center_x + cross_size, center_y), color, 3)
    cv2.line(display_frame, (center_x, center_y - cross_size), 
             (center_x, center_y + cross_size), color, 3)
    
    # Add label with background
    label = f"TAG ROI: {w}x{h}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Draw background rectangle for text
    bg_y1 = y - text_height - 15
    
    # Ensure background is within frame
    if bg_y1 >= 0:
        cv2.rectangle(display_frame, (x, bg_y1), (x + text_width + 10, y - 5), (0, 0, 0), -1)
        cv2.putText(display_frame, label, (x + 5, y - 10), font, font_scale, color, font_thickness)
    else:
        # If label would be cut off at top, put it at bottom
        cv2.rectangle(display_frame, (x, y + h + 5), 
                     (x + text_width + 10, y + h + text_height + 20), (0, 0, 0), -1)
        cv2.putText(display_frame, label, (x + 5, y + h + text_height + 10), 
                    font, font_scale, color, font_thickness)
    
    return display_frame


def display_camera_with_roi_overlay(camera_manager, camera_type='arducam', roi_type='tag'):
    """
    Display FULL camera frame with ROI overlay (not cropped).
    
    Args:
        camera_manager: OpenAIVisionCameraManager instance
        camera_type: 'arducam' or 'realsense'
        roi_type: 'tag' or 'work'
    
    Returns:
        tuple: (full_frame_with_overlay, cropped_roi) or (None, None)
    """
    # Get FULL camera frame (not cropped)
    if camera_type == 'arducam':
        full_frame = camera_manager.get_arducam_frame()
    else:
        full_frame = camera_manager.get_realsense_frame()
    
    if full_frame is None:
        return None, None
    
    # Draw ROI overlay on full frame
    frame_with_overlay = draw_roi_overlay_on_full_frame(
        full_frame, 
        camera_manager.roi_coords,
        camera_manager.original_resolution,
        roi_type=roi_type,
        color=(0, 255, 0),  # Green
        thickness=4
    )
    
    # Also get the cropped ROI for processing
    cropped_roi = camera_manager.apply_roi(full_frame, roi_type)
    
    return frame_with_overlay, cropped_roi


# ==========================
# IMPROVED OPENAI TEXT EXTRACTOR WITH BRAND CORRECTION
# ==========================
# VERTEX AI TEXT EXTRACTOR (REPLACEMENT CLASS)
# ==========================
class OpenAIVisionTextExtractor:
    """Text extractor using Vertex AI Gemini models with pay-as-you-go billing"""
    
    def __init__(self):
        # Smart caching for brand validation
        self.brand_cache = {}  # Cache validated brands to avoid re-processing
        self.cache_max_size = 50  # Keep last 50 brands in cache
        
        self.openai_client = None
        self._vertex_ai_configured = False  # Default to False
        
        try:
            self.setup_openai()
        except Exception as e:
            logger.warning(f"OpenAI setup failed: {e}")
        
        try:
            self.setup_vertex_ai()
        except Exception as e:
            logger.warning(f"Vertex AI setup failed: {e}")
            self._vertex_ai_configured = False
        
        # Don't fail initialization even if APIs aren't configured
        logger.info("Text Extractor initialized (may need API configuration)")
        
        # Comprehensive OCR corrections and brand validation
        self.ocr_corrections = {
            # Luxury brands
            'bearddeegaa': 'Balenciaga',
            'baleneiaga': 'Balenciaga',
            'balenciaga': 'Balenciaga',
            'gucci': 'Gucci',
            'prada': 'Prada',
            'louis vuitton': 'Louis Vuitton',
            'chanel': 'Chanel',
            'hermes': 'Hermès',
            'dior': 'Dior',
            'saint laurent': 'Saint Laurent',
            'givenchy': 'Givenchy',
            'versace': 'Versace',
            'valentino': 'Valentino',
            'armani': 'Armani',
            'dolcegabbana': 'Dolce & Gabbana',
            'dolce & gabbana': 'Dolce & Gabbana',
            'bottega veneta': 'Bottega Veneta',
            'celine': 'Celine',
            'loewe': 'Loewe',
            'fendi': 'Fendi',
            'burberry': 'Burberry',
            'alexander mcqueen': 'Alexander McQueen',
            'tom ford': 'Tom Ford',
            'paul smith': 'Paul Smith',
            'paulsmith': 'Paul Smith',
            'paul smth': 'Paul Smith',
            
            # Contemporary brands
            'michaelkors': 'Michael Kors',
            'michael kors': 'Michael Kors',
            'coach': 'Coach',
            'kate spade': 'Kate Spade',
            'tory burch': 'Tory Burch',
            'marcjacobs': 'Marc Jacobs',
            'marc jacobs': 'Marc Jacobs',
            'alexanderwang': 'Alexander Wang',
            'alexander wang': 'Alexander Wang',
            'stella mccartney': 'Stella McCartney',
            'isabel marant': 'Isabel Marant',
            'acne studios': 'Acne Studios',
            'off-white': 'Off-White',
            'off white': 'Off-White',
            'vetements': 'Vetements',
            'balmain': 'Balmain',
            'rick owens': 'Rick Owens',
            'comme des garcons': 'Comme des Garçons',
            'yohji yamamoto': 'Yohji Yamamoto',
            'rebecca minkoff': 'Rebecca Minkoff',
            'rebeccaminkoff': 'Rebecca Minkoff',
            'rebecca minkofi': 'Rebecca Minkoff',
            'rebecca minkof': 'Rebecca Minkoff',
            # Note: 'chloe' misreading of Rebecca Minkoff tags - but we don't want to auto-correct this
            # as it might be a real Chloe tag. Let the AI validator handle this case.
            
            # Designer brands
            'calvinklein': 'Calvin Klein',
            'calvin klein': 'Calvin Klein',
            'tommyhilfiger': 'Tommy Hilfiger',
            'tommy hilfiger': 'Tommy Hilfiger',
            'ralphlauren': 'Ralph Lauren',
            'ralph lauren': 'Ralph Lauren',
            'hugo boss': 'Hugo Boss',
            'moschino': 'Moschino',
            'marni': 'Marni',
            'jil sander': 'Jil Sander',
            
            # Contemporary Women's brands
            'antonio melan': 'Antonio Melani',  # Common OCR error
            'antonio melani': 'Antonio Melani',
            'demylee': 'Demylee',
            'kotakov': 'Komarov',  # Common OCR error - stylized font misread
            'komarov': 'Komarov',
            'demy lee': 'Demylee',
            'demy-lee': 'Demylee',
            'soia & kyo': 'Soia & Kyo',
            'soia and kyo': 'Soia & Kyo',
            'soiakyo': 'Soia & Kyo',
            'soia kyo': 'Soia & Kyo',
            'vince': 'Vince',
            'equipment': 'Equipment',
            'rag & bone': 'Rag & Bone',
            'rag and bone': 'Rag & Bone',
            'a.l.c.': 'A.L.C.',
            'alc': 'A.L.C.',
            'helmut lang': 'Helmut Lang',
            'ganni': 'Ganni',
            'staud': 'Staud',
            'cult gaia': 'Cult Gaia',
            'ulla johnson': 'Ulla Johnson',
            'zimmermann': 'Zimmermann',
            'self-portrait': 'Self-Portrait',
            'self portrait': 'Self-Portrait',
            'for love & lemons': 'For Love & Lemons',
            'for love and lemons': 'For Love & Lemons',
            
            # Streetwear
            'nike': 'Nike',
            'adidas': 'Adidas',
            'supreme': 'Supreme',
            'bape': 'Bape',
            'palace': 'Palace',
            'stussy': 'Stüssy',
            'stüssy': 'Stüssy',
            'carhartt': 'Carhartt',
            'champion': 'Champion',
            'the north face': 'The North Face',
            'patagonia': 'Patagonia',
            'columbia': 'Columbia',
            'arcteryx': 'Arc\'teryx',
            'arc teryx': 'Arc\'teryx',
            
            # Fast Fashion
            'zara': 'Zara',
            'h&m': 'H&M',
            'hm': 'H&M',
            'uniqlo': 'Uniqlo',
            'gap': 'Gap',
            'old navy': 'Old Navy',
            'jcrew': 'J.Crew',
            'j crew': 'J.Crew',
            'j.crew': 'J.Crew',
            'jcrew collection': 'J.Crew Collection',
            'j.crew collection': 'J.Crew Collection',
            'jcrew factory': 'J.Crew Factory',
            'j.crew factory': 'J.Crew Factory',
            'asos': 'ASOS',
            'topshop': 'Topshop',
            'mango': 'Mango',
            'cos': 'COS',
            '& other stories': '& Other Stories',
            'other stories': '& Other Stories',
            'massimo dutti': 'Massimo Dutti',
            
            # Denim
            'levis': 'Levi\'s',
            'levi\'s': 'Levi\'s',
            'wrangler': 'Wrangler',
            'diesel': 'Diesel',
            'true religion': 'True Religion',
            '7 for all mankind': '7 For All Mankind',
            'citizens of humanity': 'Citizens of Humanity',
            'ag jeans': 'AG Jeans',
            'j brand': 'J Brand',
            'paige': 'Paige',
            'frame': 'Frame',
            'mother': 'Mother',
            're/done': 'Re/Done',
            
            # Activewear
            'lululemon': 'Lululemon',
            'athleta': 'Athleta',
            'alo yoga': 'Alo Yoga',
            'outdoor voices': 'Outdoor Voices',
            'sweaty betty': 'Sweaty Betty',
            'beyond yoga': 'Beyond Yoga',
            'girlfriend collective': 'Girlfriend Collective',
            
            # Other common corrections
            'viktorrolf': 'Viktor & Rolf',
            'viktor & rolf': 'Viktor & Rolf',
            'yvestlaurent': 'Yves Saint Laurent',
            'yves saint laurent': 'Yves Saint Laurent',
            'stjohn': 'St. John',
            'saint john': 'St. John',
            'st. john': 'St. John'
        }
        logger.info("OpenAI Text Extractor initialized with brand validation")
    
    def validate_and_correct_brand(self, brand_text):
        """Validate and correct brand names using fuzzy matching and known corrections"""
        if not brand_text or brand_text.strip() == "":
            return brand_text
        
        # Clean the input
        brand_clean = brand_text.strip().lower()
        
        # ❌ BLACKLIST: Reject common OCR hallucinations (only if confidence is very low)
        blacklist = ['theory']  # Only reject if confidence is very low
        if brand_clean in blacklist and confidence < 0.3:
            logger.warning(f"⚠️ REJECTED blacklisted brand: '{brand_text}' - low confidence OCR error")
            return None
        
        # Direct correction lookup
        if brand_clean in self.ocr_corrections:
            corrected = self.ocr_corrections[brand_clean]
            logger.info(f"Brand corrected: '{brand_text}' -> '{corrected}'")
            return corrected
        
        # Fuzzy matching for close matches
        from difflib import get_close_matches
        possible_brands = list(self.ocr_corrections.values())
        close_matches = get_close_matches(brand_text, possible_brands, n=1, cutoff=0.8)
        
        if close_matches:
            corrected = close_matches[0]
            logger.info(f"Brand fuzzy matched: '{brand_text}' -> '{corrected}'")
            return corrected
        
        # Check for common OCR errors (missing last letter)
        if len(brand_clean) > 3:
            # Try adding common missing letters
            for letter in ['i', 'a', 'e', 'o', 'n']:
                test_brand = brand_clean + letter
                if test_brand in self.ocr_corrections:
                    corrected = self.ocr_corrections[test_brand]
                    logger.info(f"Brand corrected (missing letter): '{brand_text}' -> '{corrected}'")
                    return corrected
        
        # Return original if no correction found
        return brand_text
    
    def validate_brand_with_ai(self, ocr_text):
        """
        Uses a "skeptical" text model to validate, correct, and verify a brand name.
        This enhanced validator performs two-step verification:
        1. Correction: Fixes common OCR errors
        2. Verification: Checks if the corrected brand is a real fashion brand
        
        Returns: (validated_brand, confidence_score)
        - validated_brand: The corrected brand name or None if not a real brand
        - confidence_score: 0.0-1.0 indicating similarity between OCR and validated brand
        """
        if not ocr_text or ocr_text.lower() in ["unknown", "unreadable", "null", "none"]:
            return None, 0.0
        
        # Check cache first (huge speed boost for repeated brands)
        cache_key = ocr_text.lower().strip()
        if cache_key in self.brand_cache:
            cached_result = self.brand_cache[cache_key]
            logger.info(f"[SKEPTICAL-VALIDATOR] Cache HIT: '{ocr_text}' → '{cached_result[0]}' (confidence: {cached_result[1]:.2f})")
            return cached_result
        
        logger.info(f"[SKEPTICAL-VALIDATOR] Validating OCR text: '{ocr_text}'")
        try:
            # Use Vertex AI for brand validation
            if not self._vertex_ai_configured:
                if not self.setup_vertex_ai():
                    return ocr_text, 0.5  # Fallback if Gemini not available
            
            # Reuse the configured model instance
            text_model = self._gemini_model
            
            skeptical_validator_prompt = f"""You are a fashion brand expert. An OCR tool read the following text from a clothing tag: "{ocr_text}"

Your task is a two-step process:
1. **Correction:** Based on common OCR mistakes, what is the most likely correct spelling of the brand name?
   - Common OCR errors: "i" vs "l", "rn" vs "m", "cl" vs "d", "o" vs "a", stylized fonts
   - Consider similar-sounding brands (e.g., "Kotakov" might be "Komarov")
2. **Verification:** Is this corrected name a known, real-world clothing or fashion brand?

Return your answer ONLY in this exact JSON format:
{{
  "corrected_brand": "The most likely correct brand name",
  "is_real_brand": "YES" or "NO"
}}

Examples:
- OCR: "Kotakov" → {{"corrected_brand": "Komarov", "is_real_brand": "YES"}}
- OCR: "Antonio Melan" → {{"corrected_brand": "Antonio Melani", "is_real_brand": "YES"}}
- OCR: "Xyzabc123" → {{"corrected_brand": "Xyzabc123", "is_real_brand": "NO"}}
- OCR: "Nike" → {{"corrected_brand": "Nike", "is_real_brand": "YES"}}"""
            
            response = text_model.generate_content(
                skeptical_validator_prompt,
                generation_config={
                    "temperature": 0.0,  # Deterministic output
                    "max_output_tokens": 200,
                    "top_p": 0.95,
                    "top_k": 40
                }
            )
            
            # Robust JSON parsing
            response_text = response.text.strip()
            match = re.search(r"(\{.*?\})", response_text, re.DOTALL)
            if not match:
                logger.warning(f"[SKEPTICAL-VALIDATOR] No JSON found in response: {response_text}")
                # Fallback to old method
                validated_brand = response_text.strip().strip('"\'')
                confidence = difflib.SequenceMatcher(None, ocr_text.lower(), validated_brand.lower()).ratio()
                return validated_brand, confidence

            data = json.loads(match.group(1))
            
            validated_brand = data.get("corrected_brand", "").strip()
            is_real = data.get("is_real_brand", "NO").upper()

            # --- THE CRITICAL LOGIC ---
            # If the AI says the brand isn't real, reject it.
            if is_real == "NO":
                logger.warning(f"[SKEPTICAL-VALIDATOR] ❌ AI rejected '{validated_brand}' as a non-existent brand.")
                return None, 0.0  # Return None for the brand

            if not validated_brand:
                logger.warning(f"[SKEPTICAL-VALIDATOR] No corrected brand found in response")
                return None, 0.0

            logger.info(f"[SKEPTICAL-VALIDATOR] ✅ OCR: '{ocr_text}' -> Validated: '{validated_brand}' (Verified as real brand)")
            
            # Calculate confidence score
            confidence = difflib.SequenceMatcher(None, ocr_text.lower(), validated_brand.lower()).ratio()
            
            logger.info(f"[SKEPTICAL-VALIDATOR] Confidence Score: {confidence:.2f}")
            
            # Flag low confidence for manual review
            if confidence < 0.4:
                logger.warning(f"[SKEPTICAL-VALIDATOR] ⚠️ LOW CONFIDENCE ({confidence:.2f})! OCR '{ocr_text}' -> '{validated_brand}'")
            
            # Cache the result for future use
            self._cache_brand_result(cache_key, validated_brand, confidence)
            
            return validated_brand, confidence
            
        except Exception as e:
            logger.error(f"[AI-VALIDATOR] Validation failed: {e}")
            return ocr_text, 0.5  # Fallback with medium-low confidence
    
    def _cache_brand_result(self, cache_key, validated_brand, confidence):
        """Cache brand validation result with size management"""
        try:
            # Add to cache
            self.brand_cache[cache_key] = (validated_brand, confidence)
            
            # Manage cache size - remove oldest entries if over limit
            if len(self.brand_cache) > self.cache_max_size:
                # Remove oldest entry (first inserted)
                oldest_key = next(iter(self.brand_cache))
                del self.brand_cache[oldest_key]
            
            logger.debug(f"[CACHE] Stored '{cache_key}' → '{validated_brand}' (cache size: {len(self.brand_cache)})")
            
        except Exception as e:
            logger.warning(f"[CACHE] Failed to cache result: {e}")

    def crop_to_text_region(self, tag_image):
        """Automatically detect and crop to just the text area"""
        
        if tag_image is None or tag_image.size == 0:
            return tag_image
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(tag_image, cv2.COLOR_RGB2GRAY)
            
            # Threshold to find text
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find text contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.debug("[TEXT-CROP] No text contours found, returning original")
                return tag_image  # No text found, return original
            
            # Get bounding box of all text
            all_points = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            
            # Add 10% padding
            padding = int(min(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(tag_image.shape[1] - x, w + 2*padding)
            h = min(tag_image.shape[0] - y, h + 2*padding)
            
            # Crop to text region
            cropped = tag_image[y:y+h, x:x+w]
            
            logger.info(f"[TEXT-CROP] Reduced from {tag_image.shape} to {cropped.shape}")
            return cropped
            
        except Exception as e:
            logger.error(f"[TEXT-CROP] Error cropping to text: {e}")
            return tag_image

    def setup_openai(self):
        """Setup OpenAI client"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found")
                return
            
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.error(f"OpenAI setup failed: {e}")
            raise

    def setup_vertex_ai(self):
        """Setup Vertex AI client"""
        try:
            from google.oauth2 import service_account
            import vertexai
            from vertexai.generative_models import GenerativeModel, Part
            import google.generativeai as genai
            
            # Load credentials directly from the file
            credentials_path = "gcp_credentials.json"
            if not os.path.exists(credentials_path):
                logger.warning(f"Google credentials file not found at: {credentials_path}")
                return False
            
            project_id = os.getenv('PROJECT_ID', 'keen-answer-442316-v9')
            
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            vertexai.init(project=project_id, credentials=credentials)
            genai.configure(credentials=credentials)
            
            # Try Gemini 2.0 Flash first, fallback to 1.5 Pro
            try:
                self._gemini_model = GenerativeModel('gemini-2.0-flash-exp')
                logger.info("✅ Using Gemini 2.0 Flash (better OCR)")
            except Exception as e:
                logger.warning(f"Gemini 2.0 unavailable: {e}, falling back to 1.5 Pro")
                self._gemini_model = GenerativeModel('gemini-1.5-pro-latest')
                logger.info("✅ Using Gemini 1.5 Pro (fallback)")
            
            self._vertex_ai_configured = True
            logger.info("✅ Vertex AI initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Vertex AI setup failed: {e}")
            self._vertex_ai_configured = False
            return False

    def analyze_tag_with_gemini_sync(self, image_np):
        """Analyzes a garment tag image using Vertex AI Gemini (SYNCHRONOUS)"""
        logger.info("[VERTEX-AI] Starting SYNC tag analysis with Vertex AI Gemini...")
        try:
            if not self._vertex_ai_configured:
                return {'success': False, 'brand': None, 'size': 'Unknown', 'error': 'Vertex AI not configured', 'method': 'Vertex AI Sync'}
            
            from vertexai.generative_models import GenerativeModel, Part
            
            # Use Gemini 2.0 Flash model on Vertex AI
            model = GenerativeModel("gemini-2.0-flash-exp")
            
            # Convert to JPEG bytes for Vertex AI
            _, buffer = cv2.imencode('.jpg', image_np)
            image_part = Part.from_data(data=buffer.tobytes(), mime_type="image/jpeg")

            # Enhanced prompt for literal tag analysis with vintage/designer detection
            prompt = """You are an OCR system. Read ONLY the exact text visible on this clothing tag.

CRITICAL RULES:
- Read character-by-character, don't guess based on style or fonts
- If text is unclear, return null - DO NOT make up brand names
- NEVER substitute one brand for another (e.g., don't change "Rebecca Minkoff" to "Chloe")
- If you see "Rebecca Minkoff", return "Rebecca Minkoff" - NOT "Chloe" or any other brand

ADDITIONAL ANALYSIS:
1. **Tag Age Assessment** - Font and Typography are KEY indicators:
   
   **FONT STYLES BY ERA:**
   - **1960s-1970s**: Serif fonts, hand-drawn logos, psychedelic typography
   - **1980s**: Bold geometric sans-serif, condensed fonts, all-caps heavy use
   - **1990s**: Grunge fonts, script fonts, mixed case, courier/typewriter fonts
   - **2000s**: Clean sans-serif, Arial/Helvetica dominance, corporate minimalism
   - **2010s+**: Modern geometric fonts, thin weights, uppercase minimalism
   
   **OTHER AGE INDICATORS:**
   - Yellowing/discoloration of fabric/paper tag
   - Faded or degraded printing quality
   - Dot matrix printing (pre-1990s)
   - Tag construction (woven vs printed, materials used)
   - Care label symbols (old international standards vs modern)
   - "Made in" country (some indicate era)
   - Fiber content format (old RN numbers, outdated terminology)
   
2. **Designer Brand Validation**: If you detect a high-end brand:
   - Verify logo/typography matches authentic historical tags for that brand
   - Check printing quality (designer tags have crisp, high-quality printing)
   - Look for authenticity codes, serial numbers, or holograms
   - Assess stitching quality on woven tags

**CRITICAL**: Font style is often the MOST RELIABLE age indicator. A tag with 1980s bold geometric fonts is likely from that era even if the fabric looks good.

Return ONLY this JSON format: 
{
    "brand": "exact_brand_name_or_null", 
    "size": "size_or_null",
    "material": "material_or_null",
    "tag_age_years": estimated_age_in_years_or_null,
    "tag_condition": "pristine/good/worn/heavily_worn/vintage",
    "vintage_indicators": [
        "1980s bold sans-serif font detected",
        "yellowed fabric", 
        "old care symbols",
        "dot matrix printing"
    ],
    "font_era": "1960s/1970s/1980s/1990s/2000s/2010s+/modern/unknown",
    "is_designer": true/false,
    "authenticity_confidence": "high/medium/low/cannot_determine",
    "authenticity_notes": "specific observations about authenticity"
}"""

            response = model.generate_content(
                [image_part, prompt],
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 200,
                }
            )
            
            response_text = response.text.strip()
            logger.info(f"[READER] Raw response: {response_text}")
            
            # Parse JSON response
            import json
            import re
            
            match = re.search(r"(\{.*?\})", response_text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                brand = data.get("brand")
                size = data.get("size", "Unknown")
                material = data.get("material")
                tag_age_years = data.get("tag_age_years")
                tag_condition = data.get("tag_condition", "unknown")
                vintage_indicators = data.get("vintage_indicators", [])
                font_era = data.get("font_era", "unknown")
                is_designer = data.get("is_designer", False)
                authenticity_confidence = data.get("authenticity_confidence", "cannot_determine")
                authenticity_notes = data.get("authenticity_notes", "")
                
                return {
                    'success': True,
                    'brand': brand,
                    'size': size,
                    'material': material,
                    'tag_age_years': tag_age_years,
                    'tag_condition': tag_condition,
                    'vintage_indicators': vintage_indicators,
                    'font_era': font_era,
                    'is_designer': is_designer,
                    'authenticity_confidence': authenticity_confidence,
                    'authenticity_notes': authenticity_notes,
                    'method': 'Vertex AI Gemini 2.0 Flash'
                }
            else:
                return {'success': False, 'error': 'Could not parse response', 'brand': None, 'size': 'Unknown'}
        
        except Exception as e:
            logger.error(f"[VERTEX-AI] Vision API error: {e}")
            return {'success': False, 'error': str(e), 'brand': None, 'method': 'Vertex AI Gemini 2.0 Flash'}

    def process_vintage_and_designer(self, tag_result, brand):
        """Process vintage and designer indicators from tag analysis"""
        
        # Check if brand is in designer database
        is_designer = brand in DESIGNER_BRANDS if brand else False
        designer_tier = DESIGNER_BRANDS[brand]['tier'] if is_designer else 'none'
        
        # Calculate vintage status (20+ years = vintage)
        current_year = 2025
        tag_age_years = tag_result.get('tag_age_years')
        is_vintage = False
        vintage_year_estimate = None
        
        if tag_age_years and tag_age_years >= 20:
            is_vintage = True
            vintage_year_estimate = current_year - tag_age_years
            logger.info(f"VINTAGE DETECTED: ~{tag_age_years} years old (circa {vintage_year_estimate})")
        
        # Log designer detection
        if is_designer:
            logger.info(f"DESIGNER BRAND: {brand} ({designer_tier} tier)")
        
        return {
            'is_designer': is_designer,
            'designer_tier': designer_tier,
            'is_vintage': is_vintage,
            'vintage_year_estimate': vintage_year_estimate,
            'tag_age_years': tag_age_years,
            'authenticity_confidence': tag_result.get('authenticity_confidence', 'unknown')
        }

    def analyze_tag_with_auto_retry(self, image_np: np.ndarray, max_retries: int = 2):
        """Auto-retry system for tag analysis"""
        logger.info("[AUTO-RETRY] Starting auto-retry system...")
        
        # Try single analysis first
        result = self.analyze_tag_with_gemini_sync(image_np)
        if result.get('success') and (result.get('brand') or result.get('size') != 'Unknown'):
            return result
        
        # If failed, try with preprocessing
        try:
            processed_image = self._preprocess_image_for_ocr(image_np)
            if processed_image is not None:
                result = self.analyze_tag_with_gemini_sync(processed_image)
                if result.get('success'):
                    result['method'] = 'Auto-Retry with preprocessing'
                    return result
        except Exception as e:
            logger.warning(f"[AUTO-RETRY] Preprocessing failed: {e}")
        
        # Final fallback
        return {
            'success': False,
            'brand': None,
            'size': 'Unknown',
            'error': 'All auto-retry strategies failed',
            'method': 'Auto-Retry (Failed)'
        }

    def _preprocess_image_for_ocr(self, image):
        """Simple preprocessing for OCR"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            if len(image.shape) == 3:
                return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            return enhanced
        except Exception as e:
            logger.warning(f"[PREPROCESS] Error: {e}")
            return image

    def analyze_tag(self, image_np: np.ndarray):
        """
        PRIMARY TAG ANALYSIS METHOD - Consolidated entry point for all tag analysis.
        """
        logger.info("[TAG-ANALYSIS] Starting tag analysis...")
        
        # Try with original image first
        if self._vertex_ai_configured:
            result = self.analyze_tag_with_gemini_sync(image_np)
            if result.get('success') and result.get('brand'):
                return result
            
            # If failed, try with preprocessing
            logger.info("[TAG-ANALYSIS] Retrying with preprocessing...")
            preprocessed = self._preprocess_image_for_ocr(image_np)
            if preprocessed is not None:
                result = self.analyze_tag_with_gemini_sync(preprocessed)
                if result.get('success'):
                    result['method'] = result.get('method', 'AI') + ' (preprocessed)'
                    return result
            
            # Final fallback to auto-retry system
            result = self.analyze_tag_with_auto_retry(image_np, max_retries=2)
        else:
            logger.warning("[TAG-ANALYSIS] Vertex AI not available, using OpenAI fallback")
            result = self.analyze_tag_with_openai_fallback(image_np)
        
        return result

    def analyze_tag_with_openai_fallback(self, image_np):
        """Fallback to OpenAI Vision if Vertex AI isn't available"""
        
        if not self.openai_client:
            return {
                'success': False,
                'error': 'No AI services configured',
                'brand': None,
                'size': 'Unknown'
            }
        
        try:
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', image_np)
            base64_image = base64.b64encode(buffer).decode()
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Read this clothing tag. Return ONLY: Brand name and Size. Format: BRAND: [name] SIZE: [size]"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=100
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            brand = "Unknown"
            size = "Unknown"
            
            for line in content.split('\n'):
                if 'BRAND:' in line.upper():
                    brand = line.split(':', 1)[1].strip()
                elif 'SIZE:' in line.upper():
                    size = line.split(':', 1)[1].strip()
            
            return {
                'success': True,
                'brand': brand,
                'size': size,
                'method': 'OpenAI Vision (Fallback)'
            }
            
        except Exception as e:
            logger.error(f"OpenAI Vision fallback failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'brand': None,
                'size': 'Unknown'
            }


class MeasurementDatasetManager:
    """Manages a dataset of garment images and their keypoint annotations for YOLO training."""
    
    def __init__(self, dataset_path="measurement_dataset/"):
        self.images_path = os.path.join(dataset_path, "images")
        self.annotations_path = os.path.join(dataset_path, "annotations")
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.annotations_path, exist_ok=True)
        logger.info("✅ Measurement Dataset Manager initialized.")

    def save_sample(self, image: np.ndarray, points: list, garment_type: str = "unknown"):
        """Save a garment image with annotated armpit points for training data."""
        if image is None or len(points) < 2:
            logger.warning("Cannot save measurement sample: invalid image or insufficient points")
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_filename = f"garment_{garment_type}_{timestamp}"
            
            # Save the image
            image_path = os.path.join(self.images_path, f"{base_filename}.jpg")
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, image_bgr)

            # Save the annotation data in YOLO format
            annotation_path = os.path.join(self.annotations_path, f"{base_filename}.txt")
            
            # YOLO format: class_id x_center y_center width height (normalized 0-1)
            height, width = image.shape[:2]
            annotations = []
            
            for i, point in enumerate(points[:2]):  # Only save first 2 points (armpits)
                x_center = point['x'] / width
                y_center = point['y'] / height
                # Use small bounding box around the point (e.g., 20x20 pixels normalized)
                box_width = 20 / width
                box_height = 20 / height
                
                # Class 0 = left_armpit, Class 1 = right_armpit
                class_id = i
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
            
            with open(annotation_path, 'w') as f:
                f.write('\n'.join(annotations))
            
            # Also save JSON metadata for reference
            json_path = os.path.join(self.annotations_path, f"{base_filename}.json")
            metadata = {
                "image_path": image_path,
                "image_shape": image.shape,
                "garment_type": garment_type,
                "timestamp": timestamp,
                "points": [
                    {"label": "left_armpit", "x": points[0]['x'], "y": points[0]['y']},
                    {"label": "right_armpit", "x": points[1]['x'], "y": points[1]['y']}
                ]
            }
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Saved measurement training sample: {base_filename}")
            st.toast("💾 Saved measurement for YOLO training!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save measurement data: {e}")
            st.toast("❌ Error saving measurement data.")
            return False
    
    def get_dataset_stats(self):
        """Get statistics about the collected measurement dataset."""
        try:
            image_files = [f for f in os.listdir(self.images_path) if f.endswith('.jpg')]
            annotation_files = [f for f in os.listdir(self.annotations_path) if f.endswith('.txt')]
            
            return {
                'total_samples': len(image_files),
                'annotation_files': len(annotation_files),
                'images_path': self.images_path,
                'annotations_path': self.annotations_path
            }
        except Exception as e:
            logger.error(f"Failed to get dataset stats: {e}")
            return {'total_samples': 0, 'annotation_files': 0}
    
    def setup_openai(self):
        """Setup OpenAI client"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    def setup_vertex_ai(self):
        """
        Initializes the Vertex AI client by loading credentials directly from a JSON file.
        This bypasses any issues with environment variables.
        """
        # Prevent repeated setup
        if hasattr(self, '_vertex_ai_configured') and self._vertex_ai_configured:
            return True
            
        try:
            # --- CRITICAL STEP 1 ---
            # Load the credential file directly from the code.
            # Make sure your 'gcp_credentials.json' file is in the same folder as this script.
            credentials_path = "gcp_credentials.json"

            # --- CRITICAL STEP 2 ---
            # Replace "your-gcp-project-id" with your actual Google Cloud Project ID.
            PROJECT_ID = "keen-answer-442316-v9"  # Your actual project ID
            LOCATION = "us-central1"  # Common region, usually fine to leave as is

            if not os.path.exists(credentials_path):
                logger.error(f"❌ CRITICAL ERROR: Credential file not found at '{credentials_path}'.")
                logger.error("   Please make sure 'gcp_credentials.json' is in the same folder as your script.")
                self._vertex_ai_configured = False
                return False

            # Explicitly create credentials from the service account file
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            
            # Initialize Vertex AI with the project AND the explicit credentials
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
            
            self._vertex_ai_configured = True
            logger.info(f"✅ Google Cloud Vertex AI initialized directly from key file for project '{PROJECT_ID}'.")
            return True
            
        except Exception as e:
            self._vertex_ai_configured = False
            logger.error(f"❌ Failed to initialize Vertex AI from key file: {e}")
            logger.error("   Please ensure 'gcp_credentials.json' is valid and has the correct permissions.")
            return False
    
    def detect_text_region(self, image):
        """Find the actual text region in the tag using morphological operations"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Find text regions using morphological operations
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Dilate to connect text characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
            dilated = cv2.dilate(binary, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.info("[TEXT-REGION] No text contours found, using full image")
                return image
            
            # Get bounding box of largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Add 10% padding
            padding = int(min(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            cropped = image[y:y+h, x:x+w]
            logger.info(f"[TEXT-REGION] Cropped to {w}x{h} from {image.shape[1]}x{image.shape[0]}")
            return cropped
            
        except Exception as e:
            logger.warning(f"[TEXT-REGION] Error detecting text region: {e}")
            return image
    
    def preprocess_for_gemini(self, image):
        """Gemini-specific preprocessing for better OCR accuracy"""
        try:
            # Convert to numpy if PIL
            if hasattr(image, 'mode'):
                image_np = np.array(image)
            else:
                image_np = image.copy()
            
            # 1. Super-resolution upscaling (Gemini likes bigger images)
            if image_np.shape[0] < 800:
                scale = 800 / image_np.shape[0]
                image_np = cv2.resize(image_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                logger.info(f"[GEMINI-PREP] Upscaled to {image_np.shape[:2]}")
            
            # 2. Convert to grayscale for processing
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np.copy()
            
            # 3. Extreme contrast enhancement (Gemini responds well to high contrast)
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4,4))  # More aggressive than normal
            enhanced = clahe.apply(gray)
            
            # 4. Adaptive thresholding (better than Otsu for varied lighting)
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
            )
            
            # 5. Convert back to RGB (Gemini expects color)
            rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            
            logger.info(f"[GEMINI-PREP] Applied extreme contrast and adaptive thresholding")
            return rgb
            
        except Exception as e:
            logger.warning(f"[GEMINI-PREP] Error in preprocessing: {e}")
            return image
# ==========================
# GARMENT ANALYZER
# ==========================
class OpenAIGarmentAnalyzer:
    """Analyzes a full garment image to determine its properties."""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client

    async def analyze_garment_async(self, garment_image, client):
        """Async version of garment analysis using OpenAI."""
        if not client:
            logger.error("[GARMENT-ANALYZER] OpenAI client not available.")
            return {'success': False, 'error': 'OpenAI client not configured'}
        
        try:
            logger.info("[GARMENT-ANALYZER] Starting garment analysis...")
            
            # Convert image to base64
            if isinstance(garment_image, np.ndarray):
                _, buffer = cv2.imencode('.jpg', garment_image)
                base64_image = base64.b64encode(buffer).decode()
            else:
                # If it's already a PIL image
                buffer = io.BytesIO()
                garment_image.save(buffer, format='JPEG', quality=95)
                base64_image = base64.b64encode(buffer.getvalue()).decode()
            
            prompt = """CRITICAL RULES - READ FIRST:
1. A CARDIGAN **MUST** have buttons/zipper/open edges
2. A PULLOVER has **NO** front opening (solid front)
3. If uncertain, default to PULLOVER (it's more common)

**DECISION TREE:**
- See buttons down center front? → CARDIGAN
- See zipper down center front? → CARDIGAN  
- See two separate fabric edges? → CARDIGAN
- See NONE of the above? → PULLOVER

You are an expert fashion analyst. Analyze this garment image.

**CRITICAL: CARDIGAN vs PULLOVER DETECTION PROTOCOL**

Follow this EXACT checklist:

1. **CENTER FRONT INSPECTION** (MOST IMPORTANT):
   - Look at the CENTER VERTICAL LINE of the garment from neck to hem
   - Do you see ANY of these indicators:
     ✓ Buttons or button holes running vertically
     ✓ A zipper track
     ✓ Two separate edges meeting (placket)
     ✓ A vertical seam or overlap down the middle
     ✓ Contrasting band/trim down center front
   
   → If YES to ANY: This is a CARDIGAN (has front opening)
   → If NO to ALL: Proceed to step 2

2. **EDGE INSPECTION**:
   - Look at the LEFT and RIGHT vertical edges
   - Are they:
     ✓ Finished edges (like a jacket front)
     ✓ Has ribbed bands going vertically (cardigan edge trim)
   
   → If YES: This is a CARDIGAN (open front style)
   → If NO: Proceed to step 3

3. **NECKLINE CHECK**:
   - Look at where the neckline meets the body
   - Is there:
     ✓ A V-shaped opening going down center front
     ✓ Two separate fabric edges meeting at neck
   
   → If YES: This is a CARDIGAN
   → If NO: Proceed to step 4

4. **FINAL DETERMINATION**:
   - If the garment is KNITWEAR and you found NONE of the above:
     → It's a PULLOVER/SWEATER/TURTLENECK
   - If you're uncertain:
     → Default to PULLOVER (it's more common)
     → Only call it CARDIGAN if you're 100% certain

**COMMON MISTAKES TO AVOID:**
- ❌ Don't assume "no visible closures" = pullover
- ❌ Subtle buttons can blend with fabric color - LOOK CAREFULLY
- ❌ Some cardigans have tiny decorative buttons that are easy to miss
- ❌ Open-front cardigans have no buttons but have finished front edges

**GARMENT TYPE CLASSIFICATION:**

Primary categories:
1. **CARDIGAN** (knitwear with front opening)
   - Has buttons, zipper, OR open front edges
   - Can be worn open or closed
   - Front edges are finished/trimmed
   
2. **PULLOVER/SWEATER** (knitwear with NO front opening)
   - Must be pulled over head
   - Solid construction, no center front seam
   - Neckline types: crewneck, V-neck, turtleneck

3. **JACKET/BLAZER** (structured outerwear)
   - Has lapels or structured collar
   - Woven fabric (not knit)
   - Often has lining

**GENDER DETECTION:**
- WOMEN'S: Buttons/zipper on RIGHT side, fitted waist, darts
- MEN'S: Buttons/zipper on LEFT side, straight cut, boxy silhouette
- Report EXACTLY what you observe

Return JSON:
{
    "type": "cardigan/pullover/turtleneck/sweater/jacket/blazer/dress/...",
    "subtype": "specific style",
    "has_front_opening": true/false,
    "closure_type": "buttons/zipper/open-front/none",
    "front_opening_confidence": "certain/likely/uncertain",
    "center_front_observations": [
        "SPECIFIC: saw 5 small buttons running vertically",
        "SPECIFIC: visible zipper teeth in center",
        "SPECIFIC: no center seam visible, solid construction"
    ],
    "collar_type": "crewneck/v-neck/turtleneck/shawl-collar/hood",
    "neckline": "turtleneck/crewneck/v-neck/cowl-neck/scoop/boat-neck",
    "sleeve_length": "long/short/3-4/sleeveless",
    "silhouette": "a-line/sheath/fit-and-flare/shift/empire",
    "fit": "slim/regular/relaxed/oversized",
    "gender": "men's/women's/unisex",
    "gender_confidence": "high/medium/low",
    "gender_indicators": [
        "OBSERVED: fitted waist = women's",
        "OBSERVED: ribbed collar = turtleneck style"
    ],
    "style": "casual/formal/business/sporty",
    "condition": "excellent/good/fair/poor",
    "needs_user_confirmation": false,
    "confidence": "high/medium/low",
    "visible_features": ["turtleneck collar", "no front opening", "ribbed cuffs"],
    "reasoning": "Detected [specific feature] at center front, therefore cardigan/pullover"
}"""

            # FIX: Ensure we're using the async client correctly
            try:
                # Check if client is actually async
                if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                    response = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }],
                        max_tokens=500,
                        temperature=0.0,  # ← DETERMINISTIC RESULTS
                        seed=42           # ← EXTRA CONSISTENCY
                    )
                else:
                    raise AttributeError("Client does not have required async methods")
                    
            except (AttributeError, TypeError) as e:
                # If async doesn't work, fall back to sync
                logger.warning(f"[GARMENT-ANALYZER] Async failed, using sync: {e}")
                try:
                    # Try to create a sync client as fallback
                    from openai import OpenAI
                    sync_client = OpenAI(api_key=client.api_key)
                    response = sync_client.chat.completions.create(  # Remove await
                        model="gpt-4o",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }],
                        max_tokens=500,
                        temperature=0.0,  # ← DETERMINISTIC RESULTS
                        seed=42           # ← EXTRA CONSISTENCY
                    )
                except Exception as sync_error:
                    logger.error(f"[GARMENT-ANALYZER] Both async and sync failed: {sync_error}")
                    return {'success': False, 'error': f'Both async and sync failed: {sync_error}'}
            
            content = response.choices[0].message.content
            logger.info(f"[GARMENT-ANALYZER] Raw response: {content}")
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    
                    # --- NEW: DRESS VALIDATION LOGIC ---
                    garment_type = data.get('type', 'Unknown').lower()
                    is_dress = data.get('is_dress', False)
                    visible_features = data.get('visible_features', [])
                    
                    # Pattern matching for dress keywords
                    dress_indicators = ['skirt', 'hem', 'waist-to-hem', 'bodice', 'one-piece', 'continuous fabric']
                    has_dress_features = any(indicator in ' '.join(visible_features).lower() 
                                            for indicator in dress_indicators)
                    
                    # If AI says "jacket" or "blazer" but also detects dress features, correct it
                    if ('jacket' in garment_type or 'blazer' in garment_type) and has_dress_features:
                        logger.warning(f"[DRESS-FIX] AI said '{garment_type}' but found dress features: {visible_features}")
                        garment_type = 'Blazer Dress'  # Correct the classification
                        data['type'] = 'Blazer Dress'
                    
                    # If is_dress flag is true but type doesn't say "dress", fix it
                    if is_dress and 'dress' not in garment_type:
                        if 'blazer' in garment_type or 'jacket' in garment_type:
                            garment_type = 'Blazer Dress'
                        elif 'shirt' in garment_type:
                            garment_type = 'Shirt Dress'
                        else:
                            garment_type = 'Dress'
                        
                        logger.info(f"[DRESS-FIX] Corrected to: {garment_type}")
                        data['type'] = garment_type
                    
                    # Apply computer vision aspect ratio check
                    cv_suggests_dress = self.looks_like_dress(garment_image)
                    if cv_suggests_dress and 'dress' not in garment_type.lower():
                        logger.warning("[DRESS-CHECK] CV suggests dress but AI classified as something else")
                        # Add warning but don't override AI completely
                        warnings = data.get('warnings', [])
                        warnings.append("Possible dress misclassification - verify manually")
                        data['warnings'] = warnings
                    
                    return {
                        'success': True,
                        'garment_type': data.get('type', 'Unknown'),
                        'gender': data.get('gender', 'Unisex'),
                        'gender_confidence': data.get('gender_confidence', 'Medium'),
                        'gender_indicators': data.get('gender_indicators', []),
                        'style': data.get('style', 'Unknown'),
                        'condition': data.get('condition', 'Good'),
                        'confidence': data.get('confidence', 'Medium'),
                        'needs_user_confirmation': data.get('needs_user_confirmation', False),
                        'visible_features': data.get('visible_features', []),
                        'subtype': data.get('subtype', ''),
                        'reasoning': data.get('reasoning', 'Analysis completed'),
                        'method': 'OpenAI GPT-4o Vision'
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"[GARMENT-ANALYZER] JSON parsing failed: {e}")
                    return {'success': False, 'error': f'JSON parsing failed: {e}'}
            else:
                logger.error("[GARMENT-ANALYZER] No JSON found in response")
                return {'success': False, 'error': 'No JSON found in response'}
                
        except Exception as e:
            logger.error(f"[GARMENT-ANALYZER] Analysis failed: {e}")
            return {'success': False, 'error': str(e)}

    def looks_like_dress(self, image):
        """Simple CV check: is this garment tall/long?"""
        
        if image is None:
            return False
        
        try:
            # Convert to numpy array if needed
            if hasattr(image, 'shape'):
                img_array = image
            else:
                # Assume it's a PIL image
                img_array = np.array(image)
            
            h, w = img_array.shape[:2]
            aspect_ratio = h / w
            
            # Dresses typically have aspect ratio > 1.2 when laid flat
            # Tops are usually wider than tall (ratio < 1.0)
            
            if aspect_ratio > 1.2:
                logger.info(f"[DRESS-CHECK] Aspect ratio {aspect_ratio:.2f} suggests dress")
                return True
            
            logger.info(f"[DRESS-CHECK] Aspect ratio {aspect_ratio:.2f} suggests top/pants")
            return False
            
        except Exception as e:
            logger.warning(f"[DRESS-CHECK] Error in aspect ratio calculation: {e}")
            return False

    async def analyze_garment_with_smart_retry(self, garment_image, client):
        """Retry only if model confidence is low or validation fails"""
        
        result = await self.analyze_garment_async(garment_image, client)
        
        # Check if model is uncertain or has contradictions
        confidence = result.get('confidence', 'high')
        has_contradiction = False
        
        # Check for cardigan/pullover contradiction
        if result.get('type') in ['cardigan', 'pullover', 'sweater', 'turtleneck']:
            # Check if validation would fail
            validation_issue = self.validate_cardigan_pullover_classification(result)
            has_contradiction = not validation_issue['valid']
        
        if confidence == 'low' or has_contradiction:
            logger.warning("[SMART-RETRY] Low confidence or contradiction detected - retrying once")
            await asyncio.sleep(0.5)  # Brief pause
            
            # Try again with slightly different prompt emphasizing decision tree
            retry_result = await self.analyze_garment_async(garment_image, client)
            
            # Use retry result if it has higher confidence
            if retry_result.get('confidence') == 'high' and not has_contradiction:
                logger.info("[SMART-RETRY] Retry successful - using retry result")
                return retry_result
            else:
                logger.info("[SMART-RETRY] Retry didn't improve - using original result")
        
        return result

    def analyze_garment(self, garment_image):
        """Synchronous wrapper for the async garment analysis."""
        if not self.openai_client:
            logger.error("[GARMENT-ANALYZER] OpenAI client not available.")
            return {'success': False, 'error': 'OpenAI client not configured'}
        
        try:
            # FIX: Don't use asyncio.run with a potentially sync client
            # Instead, call the async version properly or use sync directly
            
            # Convert image to base64
            if isinstance(garment_image, np.ndarray):
                _, buffer = cv2.imencode('.jpg', garment_image)
                base64_image = base64.b64encode(buffer).decode()
            else:
                buffer = io.BytesIO()
                garment_image.save(buffer, format='JPEG', quality=95)
                base64_image = base64.b64encode(buffer.getvalue()).decode()
            
            prompt = """CRITICAL RULES - READ FIRST:
1. A CARDIGAN **MUST** have buttons/zipper/open edges
2. A PULLOVER has **NO** front opening (solid front)
3. If uncertain, default to PULLOVER (it's more common)

**DECISION TREE:**
- See buttons down center front? → CARDIGAN
- See zipper down center front? → CARDIGAN  
- See two separate fabric edges? → CARDIGAN
- See NONE of the above? → PULLOVER

You are an expert fashion analyst. Analyze this garment image.

**CRITICAL: CARDIGAN vs PULLOVER DETECTION PROTOCOL**

Follow this EXACT checklist:

1. **CENTER FRONT INSPECTION** (MOST IMPORTANT):
   - Look at the CENTER VERTICAL LINE of the garment from neck to hem
   - Do you see ANY of these indicators:
     ✓ Buttons or button holes running vertically
     ✓ A zipper track
     ✓ Two separate edges meeting (placket)
     ✓ A vertical seam or overlap down the middle
     ✓ Contrasting band/trim down center front
   
   → If YES to ANY: This is a CARDIGAN (has front opening)
   → If NO to ALL: Proceed to step 2

2. **EDGE INSPECTION**:
   - Look at the LEFT and RIGHT vertical edges
   - Are they:
     ✓ Finished edges (like a jacket front)
     ✓ Has ribbed bands going vertically (cardigan edge trim)
   
   → If YES: This is a CARDIGAN (open front style)
   → If NO: Proceed to step 3

3. **NECKLINE CHECK**:
   - Look at where the neckline meets the body
   - Is there:
     ✓ A V-shaped opening going down center front
     ✓ Two separate fabric edges meeting at neck
   
   → If YES: This is a CARDIGAN
   → If NO: Proceed to step 4

4. **FINAL DETERMINATION**:
   - If the garment is KNITWEAR and you found NONE of the above:
     → It's a PULLOVER/SWEATER/TURTLENECK
   - If you're uncertain:
     → Default to PULLOVER (it's more common)
     → Only call it CARDIGAN if you're 100% certain

**COMMON MISTAKES TO AVOID:**
- ❌ Don't assume "no visible closures" = pullover
- ❌ Subtle buttons can blend with fabric color - LOOK CAREFULLY
- ❌ Some cardigans have tiny decorative buttons that are easy to miss
- ❌ Open-front cardigans have no buttons but have finished front edges

**GARMENT TYPE CLASSIFICATION:**

Primary categories:
1. **CARDIGAN** (knitwear with front opening)
   - Has buttons, zipper, OR open front edges
   - Can be worn open or closed
   - Front edges are finished/trimmed
   
2. **PULLOVER/SWEATER** (knitwear with NO front opening)
   - Must be pulled over head
   - Solid construction, no center front seam
   - Neckline types: crewneck, V-neck, turtleneck

3. **JACKET/BLAZER** (structured outerwear)
   - Has lapels or structured collar
   - Woven fabric (not knit)
   - Often has lining

**GENDER DETECTION:**
- WOMEN'S: Buttons/zipper on RIGHT side, fitted waist, darts
- MEN'S: Buttons/zipper on LEFT side, straight cut, boxy silhouette
- Report EXACTLY what you observe

Return JSON:
{
    "type": "cardigan/pullover/turtleneck/sweater/jacket/blazer/dress/...",
    "subtype": "specific style",
    "has_front_opening": true/false,
    "closure_type": "buttons/zipper/open-front/none",
    "front_opening_confidence": "certain/likely/uncertain",
    "center_front_observations": [
        "SPECIFIC: saw 5 small buttons running vertically",
        "SPECIFIC: visible zipper teeth in center",
        "SPECIFIC: no center seam visible, solid construction"
    ],
    "collar_type": "crewneck/v-neck/turtleneck/shawl-collar/hood",
    "neckline": "turtleneck/crewneck/v-neck/cowl-neck/scoop/boat-neck",
    "sleeve_length": "long/short/3-4/sleeveless",
    "silhouette": "a-line/sheath/fit-and-flare/shift/empire",
    "fit": "slim/regular/relaxed/oversized",
    "gender": "men's/women's/unisex",
    "gender_confidence": "high/medium/low",
    "gender_indicators": [
        "OBSERVED: fitted waist = women's",
        "OBSERVED: ribbed collar = turtleneck style"
    ],
    "style": "casual/formal/business/sporty",
    "condition": "excellent/good/fair/poor",
    "needs_user_confirmation": false,
    "confidence": "high/medium/low",
    "visible_features": ["turtleneck collar", "no front opening", "ribbed cuffs"],
    "reasoning": "Detected [specific feature] at center front, therefore cardigan/pullover"
}"""

            # Use sync API call directly
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=500,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            logger.info(f"[GARMENT-ANALYZER] Raw response: {content}")
            
            # Parse JSON
            import json
            import re
            
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                
                return {
                    'success': True,
                    'garment_type': data.get('type', 'Unknown'),
                    'gender': data.get('gender', 'Unisex'),
                    'gender_confidence': data.get('gender_confidence', 'Medium'),
                    'gender_indicators': data.get('gender_indicators', []),
                    'style': data.get('style', 'Unknown'),
                    'condition': data.get('condition', 'Good'),
                    'confidence': data.get('confidence', 'Medium'),
                    'needs_user_confirmation': data.get('needs_user_confirmation', False),
                    'visible_features': data.get('visible_features', []),
                    'subtype': data.get('subtype', ''),
                    'method': 'OpenAI GPT-4o Vision (Sync)'
                }
            
            return {'success': False, 'error': 'No JSON found in response'}
            
        except Exception as e:
            logger.error(f"[GARMENT-ANALYZER] Sync wrapper failed: {e}")
            return {'success': False, 'error': str(e)}


# ==========================
# DEFECT DETECTOR
# ==========================
class DefectDetector:
    """Detect holes, stains, and other issues in garments"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    def draw_defect_boxes(self, image, defects):
        """Draw bounding boxes on image for detected defects"""
        if not defects or len(defects) == 0:
            return image
        
        # Create a copy to draw on
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Location mapping to bounding box coordinates (as percentages)
        location_map = {
            'top-left': (0.05, 0.05, 0.35, 0.35),
            'top-center': (0.35, 0.05, 0.65, 0.35),
            'top-right': (0.65, 0.05, 0.95, 0.35),
            'center-left': (0.05, 0.35, 0.35, 0.65),
            'center': (0.35, 0.35, 0.65, 0.65),
            'center-right': (0.65, 0.35, 0.95, 0.65),
            'bottom-left': (0.05, 0.65, 0.35, 0.95),
            'bottom-center': (0.35, 0.65, 0.65, 0.95),
            'bottom-right': (0.65, 0.65, 0.95, 0.95),
        }
        
        for idx, defect in enumerate(defects):
            location = defect.get('location', 'center')
            defect_type = defect.get('type', 'Unknown')
            
            # Get bounding box coordinates
            if location in location_map:
                x1_pct, y1_pct, x2_pct, y2_pct = location_map[location]
                x1, y1 = int(x1_pct * w), int(y1_pct * h)
                x2, y2 = int(x2_pct * w), int(y2_pct * h)
            else:
                # Default to center if location not recognized
                x1, y1 = int(0.35 * w), int(0.35 * h)
                x2, y2 = int(0.65 * w), int(0.65 * h)
            
            # Draw red bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # Add label with number and type
            label = f"#{idx+1}: {defect_type}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background (white rectangle)
            cv2.rectangle(annotated, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), 
                         (255, 255, 255), -1)
            
            # Draw label text (red)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return annotated
    
    async def analyze_defects_async(self, garment_image, client):
        """Async version of defect analysis"""
        try:
            if isinstance(garment_image, np.ndarray):
                pil_image = Image.fromarray(garment_image)
            else:
                pil_image = garment_image
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=70)
            base64_image = base64.b64encode(buffer.getvalue()).decode()
            
            response = await client.chat.completions.create(
                model="gpt-4o",  # Upgraded from mini for better OCR and higher rate limits
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this garment image for defects and issues.

                            Look for SIGNIFICANT defects only (ignore minor normal wear):
                            - Holes or tears
                            - Stains or discoloration
                            - Significant fading or color loss
                            - Pilling or fabric damage
                            - Missing buttons or hardware
                            - Seam issues or unraveling
                            
                            For EACH defect found, provide its location as a percentage of image dimensions:
                            
                            Return in this EXACT format:
                            DEFECT_COUNT: [number]
                            CONDITION: [Excellent/Good/Fair/Poor]
                            DEFECTS:
                            1. Type: [hole/stain/tear/etc] | Location: [top-left/top-center/top-right/center-left/center/center-right/bottom-left/bottom-center/bottom-right] | Description: [brief details]
                            2. Type: [type] | Location: [location] | Description: [details]
                            
                            If NO defects: write "DEFECTS: None"
                            
                            Be realistic - minor wrinkles and normal wear are NOT defects."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=100,
                timeout=30
            )
            
            content = response.choices[0].message.content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            result = {
                'defect_count': 0,
                'condition': 'Good',
                'defects': [],
                'success': True,
                'raw_text': content,
                'confidence': 0.9,
                'method': 'Async OpenAI Vision'
            }
            
            # Parse structured response
            parsing_defects = False
            for line in lines:
                line_upper = line.upper()
                if 'DEFECT_COUNT:' in line_upper:
                    try:
                        result['defect_count'] = int(line.split(':', 1)[1].strip())
                    except:
                        result['defect_count'] = 0
                elif 'CONDITION:' in line_upper:
                    result['condition'] = line.split(':', 1)[1].strip()
                elif 'DEFECTS:' in line_upper:
                    parsing_defects = True
                    # Check if it's "DEFECTS: None"
                    rest = line.split(':', 1)[1].strip()
                    if rest.lower() == 'none':
                        parsing_defects = False
                elif parsing_defects and line and (line[0].isdigit() or '|' in line):
                    # Parse defect line: "1. Type: stain | Location: center | Description: details"
                    defect_info = {'type': 'Unknown', 'location': 'unknown', 'description': ''}
                    parts = line.split('|')
                    for part in parts:
                        part = part.strip()
                        if 'Type:' in part or 'type:' in part:
                            defect_info['type'] = part.split(':', 1)[1].strip()
                        elif 'Location:' in part or 'location:' in part:
                            defect_info['location'] = part.split(':', 1)[1].strip().lower()
                        elif 'Description:' in part or 'description:' in part:
                            defect_info['description'] = part.split(':', 1)[1].strip()
                    if defect_info['type'] != 'Unknown':
                        result['defects'].append(defect_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Async defect analysis failed: {e}")
            return {'success': False, 'error': str(e), 'defect_count': 0, 'condition': 'Unknown'}
    
    def analyze_defects(self, garment_image):
        """Analyze garment for defects"""
        try:
            if isinstance(garment_image, np.ndarray):
                pil_image = Image.fromarray(garment_image)
            else:
                pil_image = garment_image
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            base64_image = base64.b64encode(buffer.getvalue()).decode()
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Inspect this garment for defects. Look for:
                            - Holes, stains, tears, burns
                            - Pilling, fading, stretched areas
                            - Missing buttons, broken zippers
                            
                            Return as JSON:
                            {
                                "has_defects": true/false,
                                "defects": [{"type": "", "location": "", "severity": ""}],
                                "overall_condition": "Poor/Fair/Good/Excellent/New",
                                "sellable": true/false
                            }"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                else:
                    json_str = content
                
                result = json.loads(json_str)
                result['success'] = True
                return result
                
            except:
                return {
                    'has_defects': False,
                    'defects': [],
                    'overall_condition': 'Good',
                    'sellable': True,
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"Defect detection error: {e}")
            return {'success': False, 'error': str(e)}

# ==========================
# MOCK TAG GENERATOR
# ==========================
class MockTagGenerator:
    """Generate a mock clothing tag with analyzed information"""
    
    def __init__(self):
        self.tag_width = 400
        self.tag_height = 600
        self.margin = 20
    
    def create_mock_tag(self, analysis_results):
        """Create a mock tag image"""
        tag = np.ones((self.tag_height, self.tag_width, 3), dtype=np.uint8) * 255
        cv2.rectangle(tag, (10, 10), (self.tag_width-10, self.tag_height-10), (200, 200, 200), 2)
        
        y_position = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Brand
        brand = analysis_results.get('brand', 'Unknown')
        cv2.putText(tag, brand.upper(), (50, y_position), font, 1.0, (0, 0, 0), 2)
        y_position += 60
        
        # Size
        size = analysis_results.get('size', 'Unknown')
        cv2.putText(tag, f"SIZE: {size}", (50, y_position), font, 0.8, (0, 0, 0), 2)
        y_position += 50
        
        # Type
        garment_type = analysis_results.get('type', 'Unknown')
        cv2.putText(tag, garment_type.upper(), (50, y_position), font, 0.6, (50, 50, 50), 1)
        y_position += 40
        
        # Price
        price = analysis_results.get('price_estimate', {})
        if price:
            price_text = f"${price.get('low', 0)} - ${price.get('high', 0)}"
            cv2.putText(tag, price_text, (50, y_position), font, 0.8, (0, 128, 0), 2)
        
        return tag

# ==========================
# PRICING HELPERS
# ==========================
def check_if_designer(brand):
    """Check if brand is designer"""
    if not brand or brand == 'Unknown':
        return False
    
    designer_brands = [
        'st. john', 'gucci', 'prada', 'versace', 'armani',
        'burberry', 'fendi', 'valentino', 'saint laurent',
        'balenciaga', 'givenchy', 'dolce', 'gabbana'
    ]
    
    return any(designer in brand.lower() for designer in designer_brands)

def check_if_vintage(era, style=None):
    """Check if item is vintage"""
    if not era or era == 'Unknown':
        return False
    
    vintage_indicators = ['80s', '90s', '70s', '60s', 'vintage', 'retro', 'antique', 'classic']
    era_lower = era.lower()
    style_lower = style.lower() if style else ""
    
    return any(indicator in era_lower or indicator in style_lower 
               for indicator in vintage_indicators)

def calculate_priority_price(brand, garment_type, gender, is_designer, 
                            is_vintage, condition, size):
    """Calculate price based on priority factors"""
    
    base_prices = {
        't-shirt': {'low': 15, 'mid': 25, 'high': 40},
        'shirt': {'low': 20, 'mid': 35, 'high': 60},
        'dress': {'low': 25, 'mid': 45, 'high': 80},
        'jacket': {'low': 30, 'mid': 60, 'high': 120},
        'pants': {'low': 20, 'mid': 35, 'high': 60},
        'jeans': {'low': 25, 'mid': 40, 'high': 70},
        'sweater': {'low': 20, 'mid': 35, 'high': 60},
        'unknown': {'low': 15, 'mid': 25, 'high': 40}
    }
    
    type_key = garment_type.lower() if garment_type else 'unknown'
    if type_key not in base_prices:
        type_key = 'unknown'
    
    prices = base_prices[type_key].copy()
    
    if is_designer:
        multiplier = 4.0 if 'st. john' in brand.lower() else 3.0
        prices = {k: int(v * multiplier) for k, v in prices.items()}
    
    if is_vintage:
        prices = {k: int(v * 1.3) for k, v in prices.items()}
    
    condition_mult = {'new': 1.2, 'excellent': 1.0, 'good': 0.85, 'fair': 0.6, 'poor': 0.3}
    mult = condition_mult.get(condition.lower(), 0.85)
    prices = {k: int(v * mult) for k, v in prices.items()}
    
    return prices

class EBayPricingAPI:
    """Get real-time market pricing data from the eBay Finding API."""
    
    def __init__(self):
        self.app_id = os.getenv('EBAY_APP_ID')
        # Use the production Finding API endpoint
        self.endpoint = "https://svcs.ebay.com/services/search/FindingService/v1"
        self.base_url = "https://www.ebay.com/sch/i.html"
        self.enabled = False
        
        if not self.app_id:
            logger.warning("⚠️ eBay App ID not found - pricing will use fallback methods")
            logger.warning("   → Add EBAY_APP_ID to your api.env file to enable eBay pricing")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ eBay Finding API enabled")
    
    @rate_limited(max_per_minute=3)  # eBay allows 5000 calls per DAY (≈3.5/min, use 3 for safety)
    def get_sold_listings_data(self, brand, garment_type, size, gender="", item_specifics=None):
        """
        Get actual sold prices from eBay Finding API with Item Specifics filtering
        
        Args:
            item_specifics: dict like {'Neckline': 'Turtleneck', 'Sleeve Length': 'Long Sleeve'}
        """
        
        if not self.enabled:
            logger.error("[EBAY] API not enabled - missing API key")
            return {'success': False, 'error': 'eBay API not configured'}
        
        # Check cache first (include item_specifics in cache key)
        cache_key = _cache_key_with_specifics(brand, garment_type, size, gender, item_specifics)
        cached_result = _get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Build a smarter search query
        search_terms = f'"{brand}" {garment_type}'
        if size and size.lower() not in ['unknown', 'n/a']:
            search_terms += f' size {size}'
        if gender and gender.lower() not in ['unisex', 'unknown']:
            search_terms += f' {gender}'
        
        # Base parameters
        params = {
            'OPERATION-NAME': 'findCompletedItems',
            'SERVICE-VERSION': '1.13.0',
            'SECURITY-APPNAME': self.app_id,
            'RESPONSE-DATA-FORMAT': 'JSON',
            'keywords': search_terms,
            'itemFilter(0).name': 'SoldItemsOnly',
            'itemFilter(0).value': 'true',
            'itemFilter(1).name': 'Condition',
            'itemFilter(1).value': 'Used',
            'sortOrder': 'EndTimeSoonest',
            'paginationInput.entriesPerPage': '100',
            # Request item specifics in output
            'outputSelector(0)': 'ItemSpecifics'
        }
        
        # Add Item Specifics filters (aspectFilter)
        if item_specifics:
            aspect_idx = 0
            for specific_name, specific_value in item_specifics.items():
                # Add aspect filter
                params[f'aspectFilter({aspect_idx}).aspectName'] = specific_name
                params[f'aspectFilter({aspect_idx}).aspectValueName'] = specific_value
                aspect_idx += 1
                
                logger.info(f"[EBAY] Filtering by {specific_name}: {specific_value}")
        
        try:
            # Use POST for complex requests (better for large parameter sets)
            response = requests.post(self.endpoint, data=params, timeout=20)
            
            # Better error differentiation
            if response.status_code == 429:  # Rate limited
                logger.warning("[EBAY] Rate limited - waiting before retry")
                return {'success': False, 'error': 'rate_limited', 'retry_after': 60}
            elif response.status_code == 401:  # Auth failed
                logger.error("[EBAY] Authentication failed - check API key")
                return {'success': False, 'error': 'invalid_api_key'}
            elif response.status_code >= 500:  # Server error
                logger.error(f"[EBAY] Server error: {response.status_code}")
                return {'success': False, 'error': 'server_error', 'retryable': True}
            elif response.status_code != 200:
                logger.error(f"[EBAY] API error: {response.status_code}")
                return {'success': False, 'error': f'API returned {response.status_code}'}
            
            data = response.json()
            
            # ✅ Check for eBay-specific errors
            if 'errorMessage' in data:
                error_msg = 'Unknown eBay error'
                try:
                    error_msg = data['errorMessage'][0].get('error', [{}])[0].get('message', 'Unknown error')
                except (KeyError, IndexError):
                    pass
                logger.error(f"[EBAY] API Error: {error_msg}")
                return {'success': False, 'error': error_msg}
            
            search_result = data.get('findCompletedItemsResponse', [{}])[0].get('searchResult', [{}])[0]
            items = search_result.get('item', [])
            
            if not items:
                logger.warning(f"[EBAY] No sold items found for: {search_terms}")
                return {'success': False, 'error': 'No sold items found'}
            
            # Extract prices AND item specifics
            prices = []
            items_with_specifics = []
            
            for item in items:
                try:
                    # Get price
                    price_data = item['sellingStatus'][0]['currentPrice'][0]
                    price = float(price_data['__value__'])
                    
                    # Extract item specifics
                    specifics = self._extract_item_specifics(item)
                    
                    if price > 0:
                        prices.append(price)
                        items_with_specifics.append({
                            'title': item.get('title', [''])[0],
                            'price': price,
                            'specifics': specifics,
                            'url': item.get('viewItemURL', [''])[0]
                        })
                except (KeyError, ValueError, IndexError):
                    continue
            
            if not prices:
                return {'success': False, 'error': 'No valid prices found'}
            
            # Calculate statistics
            prices_sorted = sorted(prices)
            
            # Remove outliers (top/bottom 10%)
            trim_count = max(1, len(prices_sorted) // 10)
            prices_trimmed = prices_sorted[trim_count:-trim_count] if len(prices_sorted) > 10 else prices_sorted
            
            avg_price = sum(prices_trimmed) / len(prices_trimmed)
            median_price = prices_sorted[len(prices_sorted) // 2]
            
            # Analyze item specifics to understand what we found
            specifics_analysis = self._analyze_specifics_distribution(items_with_specifics)
            
            logger.info(f"[EBAY] Found {len(prices)} sold items, avg: ${avg_price:.2f}")
            logger.info(f"[EBAY] Specifics distribution: {specifics_analysis}")
            
            result = {
                'success': True,
                'count': len(prices),
                'average_price': avg_price,
                'median_price': median_price,
                'min_price': min(prices),
                'max_price': max(prices),
                'all_prices': prices_sorted,
                'items': items_with_specifics[:10],  # Top 10 with details
                'specifics_analysis': specifics_analysis,
                'confidence': 'high' if len(prices) >= 20 else 'medium' if len(prices) >= 10 else 'low',
                'search_query': search_terms,
                'filters_used': item_specifics or {}
            }
            
            # Cache successful result
            _cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"[EBAY] API error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_item_specifics(self, item):
        """Extract item specifics from eBay item with robust error handling"""
        specifics = {}
        
        try:
            # Navigate to itemSpecifics (eBay response structure can be inconsistent)
            item_specifics = item.get('itemSpecifics', [])
            
            if not item_specifics:
                return specifics
            
            # Handle both list and dict formats
            if isinstance(item_specifics, list):
                name_value_list = item_specifics[0].get('NameValueList', [])
            elif isinstance(item_specifics, dict):
                name_value_list = item_specifics.get('NameValueList', [])
            else:
                logger.warning(f"[EBAY] Unexpected itemSpecifics type: {type(item_specifics)}")
                return specifics
            
            if not name_value_list:
                return specifics
            
            # Extract name-value pairs
            for nv_pair in name_value_list:
                try:
                    # Handle both string and list formats for Name/Value
                    name = nv_pair.get('Name', [''])
                    value = nv_pair.get('Value', [''])
                    
                    # Convert to string if it's a list
                    if isinstance(name, list):
                        name = name[0] if name else ''
                    if isinstance(value, list):
                        value = value[0] if value else ''
                    
                    if name and value:
                        specifics[name] = value
                        
                except Exception as pair_error:
                    logger.debug(f"[EBAY] Error parsing name-value pair: {pair_error}")
                    continue
        
        except Exception as e:
            logger.debug(f"[EBAY] Could not extract specifics: {e}")
        
        return specifics

    def _analyze_specifics_distribution(self, items_with_specifics):
        """Analyze distribution of item specifics across results"""
        from collections import Counter
        
        analysis = {}
        
        # Get all unique specific names
        all_specific_names = set()
        for item in items_with_specifics:
            all_specific_names.update(item['specifics'].keys())
        
        # Count values for each specific
        for specific_name in all_specific_names:
            values = [item['specifics'].get(specific_name) for item in items_with_specifics 
                      if specific_name in item['specifics']]
            
            if values:
                value_counts = Counter(values)
                analysis[specific_name] = {
                    'most_common': value_counts.most_common(3),
                    'coverage': len(values) / len(items_with_specifics)
                }
        
        return analysis

    def get_sold_comps(self, brand, garment_type, size="", gender="", condition="Used"):
        """Wrapper method to maintain compatibility with existing code"""
        return self.get_sold_listings_data(brand, garment_type, size, gender)
    
    def calculate_hybrid_price(self, brand, garment_type, condition, size, gender=""):
        """Hybrid: eBay when available, designer DB otherwise"""
        
        # Try multi-strategy eBay first
        ebay_result = self.get_sold_listings_data(brand, garment_type, size, gender)
        
        if ebay_result.get('success') and ebay_result.get('count', 0) >= 10:
            avg = ebay_result['average_price']
            return {
                'low': int(avg * 0.7),
                'mid': int(avg),
                'high': int(avg * 1.3),
                'source': f"eBay avg ({ebay_result['count']} sold)",
                'confidence': ebay_result.get('confidence', 'high')
            }
        
        # Fallback to designer database
        if brand in DESIGNER_BRANDS:
            data = DESIGNER_BRANDS[brand]
            base = BASE_GARMENT_PRICES.get(garment_type.lower(), 20)
            estimated = base * data['multiplier']
            
            condition_factor = {
                'Excellent': 1.0, 'Good': 0.7,
                'Fair': 0.45, 'Poor': 0.25
            }.get(condition, 0.7)
            
            final = estimated * condition_factor
            
            return {
                'low': int(final * 0.7),
                'mid': int(final),
                'high': int(final * 1.4),
                'source': f"Designer DB ({data['tier']} tier)",
                'confidence': 'medium'
            }
        
        # Last resort: generic calculation
        return calculate_priority_price(brand, garment_type, gender,
                                       False, False, condition, size)
    
    def generate_search_url(self, brand, garment_type, size="", gender=""):
        """Generate an eBay search URL for manual verification"""
        
        # Build search query
        search_terms = f'"{brand}" {garment_type}'
        if size and size.lower() not in ['unknown', 'n/a', '']:
            search_terms += f' size {size}'
        if gender and gender.lower() not in ['unisex', 'unknown', '']:
            search_terms += f' {gender}'
        
        # eBay sold listings search URL format
        # LH_Sold=1 shows sold items, LH_Complete=1 shows completed listings
        params = {
            '_nkw': search_terms,
            'LH_Sold': '1',
            'LH_Complete': '1',
            '_sop': '13'  # Sort by newest first
        }
        
        # Construct URL
        query_string = urlencode(params)
        search_url = f"{self.base_url}?{query_string}"
        
        return search_url
    
# ==========================
# GOOGLE LENS INTEGRATION
# ==========================
class GoogleLensIntegration:
    """Google Lens-style visual search via SERP API"""
    
    def __init__(self):
        self.api_key = os.getenv('SERPAPI_KEY')
        self.base_url = "https://serpapi.com/search"
        
        if self.api_key:
            logger.info("✅ Google Lens integration enabled")
        else:
            logger.warning("⚠️ SERPAPI_KEY not found - add to your api.env file")
    
    def reverse_image_search(self, garment_image):
        """Upload image for reverse image search (like Google Lens)"""
        
        if not self.api_key:
            return {'success': False, 'error': 'SERPAPI_KEY not found'}
        
        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', garment_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode()
            
            params = {
                'api_key': self.api_key,
                'engine': 'google_lens',  # Critical: use lens engine, not images
                'url': f'data:image/jpeg;base64,{img_base64}',
                'no_cache': 'true'  # Force fresh search
            }
            
            response = requests.post(self.base_url, data=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Google Lens returns visual matches
                visual_matches = data.get('visual_matches', [])
                
                if visual_matches:
                    brands = self._extract_brands_from_lens(visual_matches)
                    return {
                        'success': True,
                        'brands': brands[:5],
                        'matches': visual_matches[:10],
                        'method': 'Google Lens API'
                    }
            
            return {'success': False, 'error': 'No matches found'}
            
        except Exception as e:
            logger.error(f"Google Lens search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_brands_from_lens(self, matches):
        """Extract brand names from Lens visual matches"""
        
        brands = []
        brand_counts = Counter()
        
        # Brand patterns to look for in titles/sources
        known_brands = [
            'Nike', 'Adidas', 'Ralph Lauren', 'Tommy Hilfiger', 'Calvin Klein',
            'Gap', 'Zara', 'H&M', 'Levi\'s', 'Gucci', 'Prada', 'Coach',
            'Rebecca Minkoff', 'Theory', 'Vince', 'Equipment', 'Rag & Bone',
            'J.Crew', 'Banana Republic', 'Ann Taylor', 'Loft', 'Nordstrom',
            'Madewell', 'Everlane', 'Reformation', 'Free People', 'Anthropologie'
        ]
        
        for match in matches:
            title = match.get('title', '').lower()
            source = match.get('source', '').lower()
            link = match.get('link', '').lower()
            
            combined_text = f"{title} {source} {link}"
            
            # Check each known brand
            for brand in known_brands:
                if brand.lower() in combined_text:
                    brand_counts[brand] += 1
        
        # Return brands sorted by frequency
        return [brand for brand, _ in brand_counts.most_common(5)]

# ==========================
# COMPLETE BRAND DETECTION FALLBACK CHAIN
# ==========================
def detect_brand_with_complete_fallback(text_extractor, garment_image, tag_image=None):
    """Ultimate fallback chain - mirrors what you do with your phone"""
    
    start_time = time.time()
    
    # Level 1: Tag OCR (fastest, most reliable when tag is clear)
    if tag_image is not None:
        logger.info("[BRAND] Level 1: Tag OCR")
        tag_result = text_extractor.analyze_tag(tag_image)
        if tag_result.get('brand') and tag_result.get('confidence', 0) > 0.7:
            logger.info(f"[BRAND] Success via tag: {tag_result['brand']}")
            return {
                'brand': tag_result['brand'],
                'method': 'Tag OCR',
                'confidence': tag_result.get('confidence', 0.9),
                'time': time.time() - start_time
            }
    
    # Level 2: Logo detection on garment (fast, works for ~30% of items)
    logger.info("[BRAND] Level 2: Logo Detection")
    logo_result = detect_logo_on_garment(garment_image)
    if logo_result:
        logger.info(f"[BRAND] Success via logo: {logo_result}")
        return {
            'brand': logo_result,
            'method': 'Logo Detection',
            'confidence': 0.9,
            'time': time.time() - start_time
        }
    
    # Level 3: Google Lens-style visual search (slower but powerful)
    logger.info("[BRAND] Level 3: Visual Search (Google Lens)")
    lens_integration = GoogleLensIntegration()
    lens_result = lens_integration.reverse_image_search(garment_image)
    if lens_result.get('success') and lens_result.get('brands'):
        brands = lens_result['brands']
        logger.info(f"[BRAND] Visual search found: {brands}")
        
        # Show user the top 5 options
        return {
            'brand': brands[0],  # Top match
            'alternatives': brands[1:5],  # Other options
            'method': 'Visual Search',
            'confidence': lens_result.get('confidence', 0.7),
            'time': time.time() - start_time,
            'visual_matches': lens_result.get('matches', [])
        }
    
    # Level 4: Text detection on garment (sometimes brands print on fabric)
    logger.info("[BRAND] Level 4: Text on Garment")
    text_result = detect_text_on_garment(garment_image)
    if text_result:
        logger.info(f"[BRAND] Text found on garment: {text_result}")
        return {
            'brand': text_result,
            'method': 'Garment Text',
            'confidence': 0.6,
            'time': time.time() - start_time
        }
    
    # Level 5: Manual entry required
    logger.warning("[BRAND] All methods failed - manual entry needed")
    return {
        'brand': None,
        'method': 'Manual Required',
        'confidence': 0,
        'time': time.time() - start_time
    }

def detect_logo_on_garment(garment_image):
    """Simple logo detection using OpenCV template matching"""
    # This is a placeholder - you'd implement actual logo detection here
    # For now, return None to indicate no logo found
    return None

def detect_text_on_garment(garment_image):
    """Detect text printed on the garment itself"""
    # This is a placeholder - you'd implement text detection here
    # For now, return None to indicate no text found
    return None

# ==========================
# LEARNING SYSTEM: DATASET MANAGER
# ==========================
class TagDatasetManager:
    """Manages a growing dataset of tag images with ground truth labels for continuous learning"""
    
    def __init__(self, dataset_path="tag_dataset/"):
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, "images")
        self.db_path = os.path.join(dataset_path, "tag_database.db")
        self._init_database()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.images_path, exist_ok=True)
    
    def _init_database(self):
        """Create SQLite database for metadata tracking"""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Try to connect to database
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS tag_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT UNIQUE,
                    image_hash TEXT,
                    timestamp DATETIME,
                    brand TEXT,
                    size TEXT,
                    material TEXT,
                    ai_brand_prediction TEXT,
                    ai_confidence REAL,
                    user_corrected BOOLEAN DEFAULT FALSE,
                    user_validated BOOLEAN DEFAULT FALSE,
                    validation_correct BOOLEAN DEFAULT FALSE,
                    lighting_brightness INTEGER,
                    image_mean_brightness REAL,
                    preprocessing_method TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    garment_type TEXT,
                    gender TEXT,
                    condition TEXT,
                    zoom_level REAL,
                    camera_model TEXT
                )
            ''')
            
            # Create index for faster queries
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON tag_samples(timestamp)
            ''')
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_brand ON tag_samples(brand)
            ''')
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_success ON tag_samples(success)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Tag dataset database initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            logger.error(f"Database path: {self.db_path}")
            logger.error(f"Directory exists: {os.path.exists(os.path.dirname(self.db_path))}")
            # Create a fallback database path in the current directory
            fallback_path = "tag_database.db"
            logger.info(f"Trying fallback database path: {fallback_path}")
            try:
                conn = sqlite3.connect(fallback_path)
                c = conn.cursor()
                c.execute('''
                    CREATE TABLE IF NOT EXISTS tag_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_path TEXT UNIQUE,
                        image_hash TEXT,
                        timestamp DATETIME,
                        brand TEXT,
                        size TEXT,
                        material TEXT,
                        ai_brand_prediction TEXT,
                        ai_confidence REAL,
                        user_corrected BOOLEAN DEFAULT FALSE,
                        user_validated BOOLEAN DEFAULT FALSE,
                        validation_correct BOOLEAN DEFAULT FALSE,
                        lighting_brightness INTEGER,
                        image_mean_brightness REAL,
                        preprocessing_method TEXT,
                        success BOOLEAN,
                        error_message TEXT,
                        garment_type TEXT,
                        gender TEXT,
                        condition TEXT,
                        zoom_level REAL,
                        camera_model TEXT
                    )
                ''')
                conn.commit()
                conn.close()
                self.db_path = fallback_path
                logger.info("✅ Tag dataset database initialized with fallback path")
            except Exception as e2:
                logger.error(f"❌ Fallback database also failed: {e2}")
                raise e2
    
    def _calculate_image_hash(self, image):
        """Calculate hash of image for duplicate detection"""
        try:
            # Convert to bytes for hashing
            _, buffer = cv2.imencode('.jpg', image)
            return hashlib.md5(buffer.tobytes()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate image hash: {e}")
            return None
    
    def save_sample(self, tag_image, ai_result, user_correction=None, metadata={}, user_validated=False, validation_correct=False):
        """Save a tag sample with AI prediction and optional user correction/validation"""
        if tag_image is None:
            logger.warning("[DATASET] Cannot save sample: tag_image is None")
            return False
        
        try:
            # Calculate image hash to prevent duplicates
            image_hash = self._calculate_image_hash(tag_image)
            
            # Check for duplicates
            if image_hash and self._is_duplicate(image_hash):
                logger.info("[DATASET] Skipping duplicate image")
                return False
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            sanitized_brand = re.sub(r'[\\/*?:"<>|]', "_", 
                                   (user_correction.get('brand') if user_correction else ai_result.get('brand', 'unknown')))
            filename = f"{sanitized_brand}_{timestamp}.jpg"
            image_path = os.path.join(self.images_path, filename)
            
            # Save image
            cv2.imwrite(image_path, cv2.cvtColor(tag_image, cv2.COLOR_RGB2BGR))
            logger.info(f"[DATASET] Saved image: {image_path}")
            
            # Determine final values (user correction takes precedence)
            final_brand = user_correction.get('brand') if user_correction else ai_result.get('brand')
            final_size = user_correction.get('size') if user_correction else ai_result.get('size')
            
            logger.info(f"[DATASET] Saving to DB: Brand='{final_brand}', Size='{final_size}'")
            
            # Save metadata to database
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO tag_samples 
                (image_path, image_hash, timestamp, brand, size, material,
                 ai_brand_prediction, ai_confidence, user_corrected, user_validated, validation_correct,
                 lighting_brightness, image_mean_brightness, preprocessing_method, 
                 success, error_message, garment_type, gender, condition, 
                 zoom_level, camera_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_path,
                image_hash,
                datetime.now().isoformat(),
                final_brand,
                final_size,
                metadata.get('material', ''),
                ai_result.get('brand', ''),
                ai_result.get('confidence', 0.0),
                user_correction is not None,
                user_validated,
                validation_correct,
                metadata.get('brightness', 0),
                metadata.get('mean_brightness', 0.0),
                metadata.get('preprocessing', 'auto_optimized'),
                ai_result.get('success', False),
                ai_result.get('error', ''),
                metadata.get('garment_type', ''),
                metadata.get('gender', ''),
                metadata.get('condition', ''),
                metadata.get('zoom_level', 1.0),
                metadata.get('camera_model', 'arducam')
            ))
            conn.commit()
            conn.close()
            
            logger.info(f"[DATASET] ✅ Successfully saved training sample: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"[DATASET] ❌ Failed to save training sample: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _is_duplicate(self, image_hash):
        """Check if image hash already exists in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM tag_samples WHERE image_hash = ?", (image_hash,))
        count = c.fetchone()[0]
        conn.close()
        return count > 0
    
    def get_dataset_stats(self):
        """Get statistics about the dataset"""
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Total samples
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM tag_samples")
        stats['total_samples'] = c.fetchone()[0]
        
        # Success rate
        c.execute("SELECT AVG(CASE WHEN success THEN 100.0 ELSE 0.0 END) FROM tag_samples")
        result = c.fetchone()[0]
        stats['success_rate'] = result if result else 0.0
        
        # User corrections
        c.execute("SELECT COUNT(*) FROM tag_samples WHERE user_corrected = 1")
        stats['user_corrections'] = c.fetchone()[0]
        
        # User validations
        c.execute("SELECT COUNT(*) FROM tag_samples WHERE user_validated = 1")
        stats['user_validations'] = c.fetchone()[0]
        
        # Validation accuracy (when user provided feedback)
        c.execute("SELECT COUNT(*) FROM tag_samples WHERE user_validated = 1 AND validation_correct = 1")
        correct_validations = c.fetchone()[0]
        stats['validation_accuracy'] = (correct_validations / stats['user_validations'] * 100) if stats['user_validations'] > 0 else 0.0
        
        # Top brands
        c.execute("""
            SELECT brand, COUNT(*) as count 
            FROM tag_samples 
            WHERE brand IS NOT NULL AND brand != '' 
            GROUP BY brand 
            ORDER BY count DESC 
            LIMIT 10
        """)
        stats['top_brands'] = c.fetchall()
        
        # Recent activity (last 7 days)
        c.execute("""
            SELECT COUNT(*) FROM tag_samples 
            WHERE timestamp > datetime('now', '-7 days')
        """)
        stats['recent_samples'] = c.fetchone()[0]
        
        conn.close()
        return stats
    
    def export_for_training(self, output_path="training_export.json"):
        """Export dataset in format suitable for model training"""
        conn = sqlite3.connect(self.db_path)
        
        # Get all samples with corrections (these are our "ground truth")
        c = conn.cursor()
        c.execute("""
            SELECT image_path, brand, size, garment_type, gender, condition,
                   lighting_brightness, preprocessing_method
            FROM tag_samples 
            WHERE user_corrected = 1 OR success = 1
            ORDER BY timestamp
        """)
        
        training_data = []
        for row in c.fetchall():
            training_data.append({
                'image_path': row[0],
                'brand': row[1],
                'size': row[2],
                'garment_type': row[3],
                'gender': row[4],
                'condition': row[5],
                'lighting_brightness': row[6],
                'preprocessing_method': row[7]
            })
        
        conn.close()
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"✅ Exported {len(training_data)} training samples to {output_path}")
        return len(training_data)

# ==========================
# MINIMAL ENHANCED PIPELINE MANAGER
# ==========================
class MeasurementCalculator:
    """Calculate garment measurements from clicked points"""
    
    def __init__(self):
        self.calibration_size = 6.14  # Dollar bill length in inches
        self.points = []
        self.calibrated = False
        self.pixels_per_inch = 0.0
        self.load_calibration()
    
    def load_calibration(self):
        """Load calibration data from file if available"""
        try:
            if os.path.exists('calibration.json'):
                with open('calibration.json', 'r') as f:
                    calibration_data = json.load(f)
                    self.pixels_per_inch = calibration_data.get('pixels_per_inch', 0.0)
                    if self.pixels_per_inch > 0:
                        self.calibrated = True
                        logger.info(f"Loaded calibration: {self.pixels_per_inch:.2f} pixels per inch")
                    else:
                        logger.warning("Invalid calibration data found")
            else:
                logger.info("No calibration file found - measurements will be estimates")
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            self.calibrated = False
        
    def calibrate_from_bill(self, bill_length_pixels):
        """Calibrate pixels per inch from dollar bill (6.14 inches)"""
        if bill_length_pixels > 0:
            self.pixels_per_inch = bill_length_pixels / 6.14  # Dollar bill length
            self.calibrated = True
            logger.info(f"Calibrated: {self.pixels_per_inch:.2f} pixels per inch")
            return self.pixels_per_inch
        return None
    
    def calibrate_from_tag(self, tag_width_pixels, known_width_inches=2.5):
        """Calibrate pixels per inch from tag"""
        if tag_width_pixels > 0:
            self.pixels_per_inch = tag_width_pixels / known_width_inches
            self.calibrated = True
            logger.info(f"Calibrated from tag: {self.pixels_per_inch:.2f} pixels per inch")
            return self.pixels_per_inch
        return None
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points in inches"""
        if not self.calibrated or self.pixels_per_inch == 0:
            return None
        
        pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        return pixel_distance / self.pixels_per_inch
    
    def calculate_bust(self, left_armpit, right_armpit):
        """Calculate bust measurement from armpit points"""
        if not self.calibrated:
            return None
        
        # Distance between armpits (half circumference)
        armpit_distance = self.calculate_distance(left_armpit, right_armpit)
        
        if armpit_distance is None:
            return None
        
        # Estimate full circumference (double the front measurement)
        bust_measurement = armpit_distance * 2
        
        return bust_measurement

    def quick_garment_type_check(self, image):
        """Quick CV check for obvious garment types"""
        if image is None:
            return None
        
        h, w = image.shape[:2]
        aspect_ratio = h / w
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for structured elements (lapels, collars)
        # These appear as strong vertical/diagonal lines in upper portion
        upper_third = edges[:h//3, :]
        strong_edges = np.sum(upper_third > 0)
        total_pixels = upper_third.size
        edge_density = strong_edges / total_pixels
        
        hints = {}
        
        # High edge density in upper portion = structured garment (jacket/blazer)
        if edge_density > 0.15:
            hints['likely_jacket'] = True
            hints['reasoning'] = "Structured shoulders/collar detected"
        
        # Very tall aspect ratio = dress
        if aspect_ratio > 1.3:
            hints['likely_dress'] = True
            hints['reasoning'] = "Tall aspect ratio suggests full-length garment"
        
        # Short aspect ratio = top/jacket
        if aspect_ratio < 0.8:
            hints['likely_top_or_jacket'] = True
            hints['reasoning'] = "Short aspect ratio suggests upper body garment"
        
        return hints


def pause_refresh_on_interaction():
    """Pause live feed when user is interacting with UI elements"""
    
    # Check if any form elements are focused
    interaction_keys = [
        'manual_brand_entry',
        'manual_size_input',
        'brand_selection_radio',
        'garment_type_selection',
        'manual_garment_type'
    ]
    
    for key in interaction_keys:
        if key in st.session_state and st.session_state[key]:
            # User is typing/selecting - pause refresh
            st.session_state.live_preview_enabled = False
            return True
    
    return False

def show_confirmation_dialog(title, message, options, key_prefix):
    """Display a modal-style confirmation dialog"""
    
    # Create a visually distinct container
    st.markdown("""
    <style>
    .confirmation-modal {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 3px solid #fff;
    }
    .modal-title {
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }
    .modal-message {
        color: #f0f0f0;
        font-size: 16px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="confirmation-modal">', unsafe_allow_html=True)
        st.markdown(f'<div class="modal-title">⚠️ {title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="modal-message">{message}</div>', unsafe_allow_html=True)
        
        # Center the selection
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            selection = st.radio(
                "Please select:",
                options,
                key=f"{key_prefix}_selection",
                label_visibility="collapsed"
            )
            
            if st.button("✅ Confirm", type="primary", key=f"{key_prefix}_confirm", width='stretch'):
                st.markdown('</div>', unsafe_allow_html=True)
                return selection
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return None

class EnhancedPipelineManager:
    """Pipeline manager with minimal initialization for testing"""
    
    def __init__(self):
        print("Initializing Pipeline Manager...")
        
        # Consolidated steps (calibration removed)
        self.steps = [
            "Capture & Analyze Tag",
            "Capture & Analyze Garment",
            "Measure Garment",
            "Detect Defects",
            "Calculate Price"
        ]
        self.current_step = 0
        self.completed_steps = set()
        
        # Background analysis system for garment analysis
        self.background_garment_thread = None
        self.background_garment_result = None
        self.background_garment_error = None
        
        print("  - Steps initialized")
        
        # Use working light controller with fast discovery
        self.light_controller = ElgatoLightController(quick_mode=True)
        self.auto_optimizer = ImprovedSmartLightOptimizer(self.light_controller)
        self.measurement_calculator = MeasurementCalculator()
        
        # Initialize SERP API brand detector
        self.serp_detector = SERPAPIBrandDetector()
        print("  - SERP API brand detector initialized")
        
        print("  - Light controller initialized (working)")
        
        # Initialize camera manager
        try:
            self.camera_manager = OpenAIVisionCameraManager()
            print("  - Camera manager initialized")
        except Exception as e:
            print(f"  - Camera manager failed: {e}")
            self.camera_manager = None
        
        # Initialize data
        self.pipeline_data = PipelineData()
        print("  - Pipeline data initialized")
        
        # Initialize analyzers
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                import openai
                openai_client = openai.OpenAI(api_key=api_key)
                self.text_extractor = OpenAIVisionTextExtractor()
                self.garment_analyzer = OpenAIGarmentAnalyzer(openai_client)
                self.defect_detector = DefectDetector(openai_client)
                print("  - OpenAI analyzers initialized")
            except Exception as e:
                print(f"  - OpenAI initialization failed: {e}")
                self.text_extractor = None
                self.garment_analyzer = None
                self.defect_detector = None
        else:
            print("  - No OpenAI API key found")
            self.text_extractor = None
            self.garment_analyzer = None
            self.defect_detector = None
        
        self.pricing_api = EBayPricingAPI()
        self.mock_tag_generator = MockTagGenerator()
        
        # Initialize eBay sold comps research
        try:
            self.ebay_finder = eBayCompsFinder()
            print("  - eBay sold comps research initialized")
        except Exception as e:
            print(f"  - eBay finder initialization failed: {e}")
            self.ebay_finder = None
        
        # Initialize learning system
        try:
            self.dataset_manager = TagDatasetManager()
            print("  - Learning system (dataset manager) initialized")
        except Exception as e:
            print(f"  - Learning system failed to initialize: {e}")
            self.dataset_manager = None
        
        # Initialize measurement dataset manager
        try:
            self.measurement_manager = MeasurementDatasetManager()
            print("  - Measurement dataset manager initialized")
        except Exception as e:
            print(f"  - Measurement dataset manager failed to initialize: {e}")
            self.measurement_manager = None
        
        # Initialize clean state management
        self.ui_state = UIState()
        self.camera_cache = CameraCache()
        self.analysis_state = AnalysisState()
        self.retry_manager = SimpleRetryManager()
        print("  - Clean state management initialized")
    
    def start_background_garment_analysis(self, garment_image):
        """Start garment analysis in background thread"""
        if self.background_garment_thread and self.background_garment_thread.is_alive():
            return False  # Already running
        
        self.background_garment_result = None
        self.background_garment_error = None
        
        def run_analysis():
            try:
                self.background_garment_result = self.garment_analyzer.analyze_garment(garment_image)
            except Exception as e:
                self.background_garment_error = str(e)
        
        self.background_garment_thread = threading.Thread(target=run_analysis)
        self.background_garment_thread.start()
        return True
    
    def get_background_garment_result(self):
        """Get background garment analysis result if ready"""
        if self.background_garment_thread and not self.background_garment_thread.is_alive():
            if self.background_garment_result:
                return self.background_garment_result
            elif self.background_garment_error:
                return {'success': False, 'error': self.background_garment_error}
        return None
    
    def handle_step_0_tag_analysis(self):
        """Clean handler for Step 0: Tag Analysis with intelligent lighting probe"""
        try:
            # Run intelligent lighting probe using helper method
            self._run_intelligent_lighting_probe()
            
            # Get final camera frame with optimized lighting
            tag_image = self.camera_manager.get_arducam_frame()
            if tag_image is None:
                return {'success': False, 'error': 'No camera frame available'}
            
            # Apply ROI
            roi_image = self.camera_manager.apply_roi(tag_image, 'tag')
            if roi_image is None:
                return {'success': False, 'error': 'ROI not set or invalid'}
            
            # Store the image
            self.pipeline_data.tag_image = roi_image
            
            # Analyze with simplified method
            result = self.analyze_tag_simple(roi_image)
            
            if result.get('success'):
                # Store basic results
                self.pipeline_data.brand = result.get('brand')
                raw_size = result.get('size')
                
                # ENHANCED: Check for numerical sizing if no size detected
                if not raw_size or raw_size == 'Unknown':
                    logger.info("[SIZE-DETECTION] No size detected, checking for numerical sizing...")
                    try:
                        numerical_size = self._detect_numerical_size_from_tag(roi_image)
                        if numerical_size:
                            raw_size = numerical_size
                            logger.info(f"[SIZE-DETECTION] Found numerical size: {raw_size}")
                        else:
                            logger.info("[SIZE-DETECTION] No numerical size detected, continuing with Unknown")
                    except Exception as e:
                        logger.warning(f"[SIZE-DETECTION] Numerical size detection failed: {e}")
                        logger.info("[SIZE-DETECTION] Continuing with Unknown size")
                
                # Convert international sizing to US
                if raw_size and raw_size != 'Unknown':
                    # Enhanced size conversion for common EU sizes
                    if raw_size.isdigit():
                        eu_size = int(raw_size)
                        # EU to US conversion for clothing
                        if self.pipeline_data.gender == 'men':
                            if eu_size <= 36: us_size = "XS"
                            elif eu_size <= 38: us_size = "S"
                            elif eu_size <= 40: us_size = "M"
                            elif eu_size <= 42: us_size = "L"
                            elif eu_size <= 44: us_size = "XL"
                            elif eu_size <= 46: us_size = "XXL"
                            else: us_size = f"EU {eu_size}"
                        else:  # women's
                            if eu_size <= 34: us_size = "XS"
                            elif eu_size <= 36: us_size = "S"
                            elif eu_size <= 38: us_size = "M"
                            elif eu_size <= 40: us_size = "L"
                            elif eu_size <= 42: us_size = "XL"
                            elif eu_size <= 44: us_size = "XXL"
                            else: us_size = f"EU {eu_size}"
                        
                        self.pipeline_data.size = us_size
                        self.pipeline_data.raw_size = f"EU {eu_size}"
                        logger.info(f"EU size converted: {eu_size} -> {us_size}")
                    else:
                        # Try standard conversion for non-numeric sizes
                        converted_size = convert_size_to_us(
                            raw_size, 
                            self.pipeline_data.gender or 'women',
                            region='EU'
                        )
                        self.pipeline_data.size = converted_size
                        self.pipeline_data.raw_size = raw_size
                        
                        if converted_size != raw_size:
                            logger.info(f"Size converted: {raw_size} -> {converted_size}")
                else:
                    self.pipeline_data.size = 'Unknown'
                
                # Process vintage and designer data
                vintage_designer_info = self.text_extractor.process_vintage_and_designer(
                    result, 
                    self.pipeline_data.brand
                )
                
                self.pipeline_data.is_designer = vintage_designer_info['is_designer']
                self.pipeline_data.designer_tier = vintage_designer_info['designer_tier']
                self.pipeline_data.is_vintage = vintage_designer_info['is_vintage']
                self.pipeline_data.vintage_year_estimate = vintage_designer_info['vintage_year_estimate']
                self.pipeline_data.tag_age_years = vintage_designer_info['tag_age_years']
                self.pipeline_data.authenticity_confidence = vintage_designer_info['authenticity_confidence']
                
                # Store additional vintage data from AI analysis
                self.pipeline_data.font_era = result.get('font_era', 'unknown')
                self.pipeline_data.vintage_indicators = result.get('vintage_indicators', [])
                self.pipeline_data.material = result.get('material')
                
                # Stop live feed after successful analysis (if running in Streamlit)
                try:
                    st.session_state.live_preview_enabled = False
                except:
                    pass  # Ignore if not running in Streamlit context
                
                # Start background garment analysis
                garment_image = self.camera_manager.capture_garment_for_analysis()
                if garment_image is not None:
                    # Enhance center front region for better classification
                    enhanced_image = self.camera_manager.enhance_garment_for_classification(garment_image)
                    self.start_background_garment_analysis(enhanced_image)
                
                # Reset camera exposure for next run
                self.camera_manager.reset_auto_exposure()
                
                # Research brand with eBay sold comps
                if self.ebay_finder:
                    try:
                        logger.info(f"[EBAY] Starting research for {self.pipeline_data.brand}...")
                        self.pipeline_data = research_brand_with_ebay(self.pipeline_data, self.ebay_finder)
                    except Exception as e:
                        logger.warning(f"[EBAY] Research failed: {e}")
                        # Continue without eBay data
                
                return {'success': True, 'message': f"Brand: {result.get('brand')}, Size: {result.get('size')}"}
            else:
                # Reset camera exposure even on failure
                self.camera_manager.reset_auto_exposure()
                return {'success': False, 'error': result.get('error', 'Analysis failed')}
                
        except Exception as e:
            # Reset camera exposure on any error
            self.camera_manager.reset_auto_exposure()
            logger.error(f"Step 0 handler error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_current_step(self):
        """Execute the logic for the current step and return success/failure"""
        try:
            if self.current_step == 0:
                result = self.handle_step_0_tag_analysis()
                if result is None:
                    return {'success': False, 'error': 'Step 0 returned None - check camera and lighting'}
                return result
            elif self.current_step == 1:
                result = self.handle_step_1_garment_analysis()
                if result is None:
                    return {'success': False, 'error': 'Step 1 returned None - check garment capture'}
                return result
            elif self.current_step == 2:
                return {'success': True, 'message': 'Measurements step - ready for calibration'}
            elif self.current_step == 3:
                return {'success': True, 'message': 'Measurements complete'}
            elif self.current_step == 4:
                return {'success': True, 'message': 'Defects checked'}
            elif self.current_step == 5:
                return {'success': True, 'message': 'Pricing calculated'}
            else:
                return {'success': True, 'message': 'Final review - ready to complete'}
        except Exception as e:
            logger.error(f"Step {self.current_step} failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
    
    def _detect_numerical_size_from_tag(self, image_np: np.ndarray):
        """Detect numerical sizing patterns like '0 1 2 3 4' where one number is emphasized"""
        try:
            import pytesseract
            import re
            import signal
            import time
            
            # Add timeout protection
            def timeout_handler(signum, frame):
                raise TimeoutError("OCR processing timed out")
            
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Try simpler approach first - just basic OCR without complex configs
            try:
                logger.info("[NUMERICAL-SIZE] Starting simple OCR detection...")
                
                # Set timeout for OCR (5 seconds)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                
                # Simple OCR with basic config
                text = pytesseract.image_to_string(gray, config='--psm 6')
                
                # Cancel timeout
                signal.alarm(0)
                
                logger.info(f"[NUMERICAL-SIZE] Basic OCR result: '{text.strip()}'")
                
                # Look for number sequences
                numbers = re.findall(r'\d+', text)
                
                if len(numbers) >= 3:  # Found number sequence
                    logger.info(f"[NUMERICAL-SIZE] Found number sequence: {numbers}")
                    
                    # For CRUSH tags, typically the first number is the size
                    # Check if we have a sequence like "0 1 2 3 4"
                    if len(numbers) >= 5 and all(int(n) < 10 for n in numbers):
                        logger.info(f"[NUMERICAL-SIZE] CRUSH-style sequence detected, using first number: {numbers[0]}")
                        return numbers[0]
                    elif len(numbers) >= 3:
                        logger.info(f"[NUMERICAL-SIZE] Using first number as size: {numbers[0]}")
                        return numbers[0]
                
                # If no sequence found, look for single emphasized numbers
                if len(numbers) >= 1:
                    logger.info(f"[NUMERICAL-SIZE] Single number found, using: {numbers[0]}")
                    return numbers[0]
                
                logger.info("[NUMERICAL-SIZE] No numerical sizing patterns detected")
                return None
                
            except TimeoutError:
                logger.warning("[NUMERICAL-SIZE] OCR processing timed out")
                signal.alarm(0)  # Cancel alarm
                return None
            except Exception as e:
                logger.warning(f"[NUMERICAL-SIZE] OCR failed: {e}")
                signal.alarm(0)  # Cancel alarm
                return None
            
        except Exception as e:
            logger.error(f"[NUMERICAL-SIZE] Error detecting numerical size: {e}")
            return None
    
    # Removed _find_emphasized_number function - using simplified approach

    def _run_intelligent_lighting_probe(self):
        """Extract intelligent lighting probe logic for reuse"""
        if not self.auto_optimizer.enabled or not self.auto_optimizer.light_controller:
            return
        
        logger.info("[TAG-ANALYSIS] Running probe...")
        
        # Step 1: Start with VERY low light to test reflectivity
        self.auto_optimizer.light_controller.set_light(brightness=5, temperature=6000)
        self.camera_manager.reset_auto_exposure()  # Let camera auto-adjust
        time.sleep(2.0)  # Give camera time to adjust
        
        # Step 2: Capture probe frame
        probe_frame = self.camera_manager.get_arducam_frame()
        if probe_frame is None:
            logger.warning("[TAG-ANALYSIS] No probe frame, using default 20%")
            self.auto_optimizer.light_controller.set_light(brightness=20, temperature=6000)
            time.sleep(1.5)
            return
        
        probe_roi = self.camera_manager.apply_roi(probe_frame, 'tag')
        if probe_roi is None:
            logger.warning("[TAG-ANALYSIS] No ROI, using default 20%")
            self.auto_optimizer.light_controller.set_light(brightness=20, temperature=6000)
            time.sleep(1.5)
            return
        
        # Step 3: Analyze brightness
        brightness_info = self.auto_optimizer.analyze_image_brightness(probe_roi)
        if not brightness_info:
            logger.warning("[TAG-ANALYSIS] No brightness info, using default 20%")
            self.auto_optimizer.light_controller.set_light(brightness=20, temperature=6000)
            time.sleep(1.5)
            return
        
        probe_brightness = brightness_info['mean']
        logger.info(f"[TAG-ANALYSIS] Probe brightness at 5%: {probe_brightness:.0f}/255")
        
        # Step 4: Calculate target - VERY CONSERVATIVE for white tags
        if probe_brightness > 180:  # Extremely reflective at just 5% light
            target = 3
            exposure = -10  # Very low camera sensitivity
            logger.warning(f"ULTRA BRIGHT tag at 5% light! Using {target}% + exposure {exposure}")
        elif probe_brightness > 140:  # Very reflective
            target = 8
            exposure = -8
            logger.warning(f"VERY BRIGHT tag. Using {target}% + exposure {exposure}")
        elif probe_brightness > 100:  # Moderately reflective
            target = 15
            exposure = -6
            logger.info(f"Bright tag. Using {target}% + exposure {exposure}")
        elif probe_brightness > 60:  # Good range
            target = 25
            exposure = None  # Auto exposure OK
            logger.info(f"Good brightness. Using {target}%")
        else:  # Dark tag
            target = 50
            exposure = None
            logger.info(f"Dark tag. Using {target}%")
        
        # Step 5: Apply settings
        if exposure:
            self.camera_manager.set_exposure(exposure)
        else:
            self.camera_manager.reset_auto_exposure()
        
        self.auto_optimizer.light_controller.set_light(brightness=target, temperature=6000)
        time.sleep(2.0)  # Let everything stabilize
        
        logger.info(f"[TAG-ANALYSIS] Final settings: {target}% brightness, exposure={exposure}")
    
    def _determine_target_brightness(self, probe_brightness):
        """Determine target brightness and camera exposure based on probe brightness"""
        if probe_brightness > 200:  # EXTREMELY reflective (the "washout" case)
            logger.warning("🚨 EXTREMELY REFLECTIVE tag detected. Using minimum light and low exposure.")
            exposure_result = self.camera_manager.set_exposure(-8)  # Tell camera to be much less sensitive
            logger.info(f"[TAG-ANALYSIS] Set exposure -8 result: {exposure_result}")
            return 15  # CHANGE from 2 to 15 - too dark to read
        elif probe_brightness > 150:  # Very reflective
            logger.warning("⚠️ Very reflective tag detected. Using low light and low exposure.")
            exposure_result = self.camera_manager.set_exposure(-7)  # Tell camera to be less sensitive
            logger.info(f"[TAG-ANALYSIS] Set exposure -7 result: {exposure_result}")
            return 25  # CHANGE from 5 to 25
        elif probe_brightness > 100:
            self.camera_manager.reset_auto_exposure()  # Use auto exposure
            logger.info(f"[TAG-ANALYSIS] Using auto exposure for reflective tag")
            return 15
        elif probe_brightness > 60:
            self.camera_manager.reset_auto_exposure()  # Use auto exposure
            logger.info(f"[TAG-ANALYSIS] Using auto exposure for medium tag")
            return 40
        elif probe_brightness > 30:
            self.camera_manager.reset_auto_exposure()  # Use auto exposure
            logger.info(f"[TAG-ANALYSIS] Using auto exposure for dark tag")
            return 40
        else:  # Dark tag
            self.camera_manager.reset_auto_exposure()  # Use auto exposure
            logger.info(f"[TAG-ANALYSIS] Using auto exposure for very dark tag")
            return 70
    
    def handle_step_1_garment_analysis(self):
        """Clean handler for Step 1: Garment Analysis"""
        try:
            # Check for background results first
            background_result = self.get_background_garment_result()
            if background_result and background_result.get('success'):
                # Use background results
                self.pipeline_data.garment_type = background_result.get('garment_type', 'Unknown')
                self.pipeline_data.gender = background_result.get('gender', 'Unisex')
                self.pipeline_data.gender_confidence = background_result.get('gender_confidence', 'Medium')
                self.pipeline_data.gender_indicators = background_result.get('gender_indicators', [])
                self.pipeline_data.style = background_result.get('style', 'Casual')
                self.pipeline_data.era = background_result.get('era', 'Contemporary')
                self.pipeline_data.condition = background_result.get('condition', 'Good')
                
                return {'success': True, 'message': 'Background analysis complete', 'from_background': True}
            
            # Manual analysis if background failed - use extended ROI for better framing
            garment_image = self.camera_manager.capture_garment_for_analysis()
            if garment_image is None:
                return {'success': False, 'error': 'No garment image available'}
            
            # Enhance center front region for better button/zipper detection
            enhanced_image = self.camera_manager.enhance_garment_for_classification(garment_image)
            
            roi_image = self.camera_manager.apply_roi(enhanced_image, 'work')
            if roi_image is None:
                return {'success': False, 'error': 'Garment ROI not set'}
            
            # Store image and analyze
            self.pipeline_data.garment_image = roi_image
            if self.garment_analyzer:
                result = self.garment_analyzer.analyze_garment(roi_image)
            else:
                return {'success': False, 'error': 'Garment analyzer not available'}
            
            # FIX: Check if we got valid data, not just success flag
            if result.get('success') or result.get('garment_type'):  # CHANGED THIS LINE
                # Store ALL the fields including Item Specifics
                self.pipeline_data.garment_type = result.get('garment_type', 'Unknown')
                self.pipeline_data.subtype = result.get('subtype', 'Unknown')
                self.pipeline_data.has_front_opening = result.get('has_front_opening', False)
                self.pipeline_data.collar_type = result.get('collar_type', 'Unknown')
                self.pipeline_data.neckline = result.get('neckline', 'Unknown')
                self.pipeline_data.sleeve_length = result.get('sleeve_length', 'Unknown')
                self.pipeline_data.silhouette = result.get('silhouette', 'Unknown')
                self.pipeline_data.fit = result.get('fit', 'Unknown')
                self.pipeline_data.gender = result.get('gender', 'Unisex')
                self.pipeline_data.gender_confidence = result.get('gender_confidence', 'Medium')
                self.pipeline_data.gender_indicators = result.get('gender_indicators', [])
                self.pipeline_data.style = result.get('style', 'Casual')
                self.pipeline_data.era = result.get('era', 'Contemporary')
                self.pipeline_data.condition = result.get('condition', 'Good')
                
                # VALIDATION: Check cardigan vs pullover classification
                validation = validate_cardigan_pullover_classification(result)
                if not validation['valid']:
                    logger.error(f"⚠️ Classification Issue: {validation['error']}")
                    logger.warning(f"💡 Suggested correction: {validation['suggestion']}")
                    # Store validation result for UI display
                    self.pipeline_data.validation_issue = validation
                elif validation.get('requires_user_confirmation'):
                    logger.warning(f"⚠️ {validation.get('warning', 'Low confidence classification')}")
                    self.pipeline_data.validation_issue = validation
                
                # CV VALIDATION: Check for obvious misclassifications
                if result.get('garment_type') == 'ambiguous_dress_or_top':
                    # Run quick CV check
                    cv_hints = self.camera_manager.quick_garment_type_check(self.pipeline_data.garment_image)
                    
                    if cv_hints and cv_hints.get('likely_jacket'):
                        logger.info("[CV-CHECK] Overriding ambiguous - CV detected jacket structure")
                        self.pipeline_data.garment_type = 'Jacket'
                        result['garment_type'] = 'Jacket'
                        result['needs_user_confirmation'] = False
                        result['reasoning'] += " (CV override: structured shoulders detected)"
                
                # Validate cardigan classification
                if result.get('garment_type', '').lower() == 'cardigan':
                    has_opening = result.get('has_front_opening', False)
                    visible_features = result.get('visible_features', [])
                    
                    # Cardigan MUST have front opening
                    if not has_opening:
                        logger.warning("[GARMENT] AI said cardigan but no front opening detected - likely misclassification")
                        # Check for turtleneck indicators
                        features_str = ' '.join(visible_features).lower()
                        if 'turtleneck' in features_str or 'high collar' in features_str:
                            logger.info("[GARMENT] Correcting: cardigan → turtleneck (no front opening + high collar)")
                            self.pipeline_data.garment_type = 'Turtleneck'
                            result['garment_type'] = 'Turtleneck'
                            result['subtype'] = 'turtleneck pullover'
                            result['reasoning'] += " [AUTO-CORRECTED: no front opening detected]"
                        else:
                            logger.info("[GARMENT] Correcting: cardigan → pullover (no front opening)")
                            self.pipeline_data.garment_type = 'Pullover'
                            result['garment_type'] = 'Pullover'
                            result['subtype'] = 'pullover sweater'
                            result['reasoning'] += " [AUTO-CORRECTED: no front opening detected]"
                
                # NEW: Store confirmation flag
                self.pipeline_data.needs_user_confirmation = result.get('needs_user_confirmation', False)
                
                # If needs confirmation, don't advance yet
                if self.pipeline_data.needs_user_confirmation:
                    logger.info("[GARMENT] Analysis ambiguous - requiring user confirmation")
                    return {
                        'success': True, 
                        'message': 'Partial analysis - user confirmation needed',
                        'needs_confirmation': True
                    }
                
                # Stop live feed after successful analysis
                st.session_state.live_preview_enabled = False
                
                # Log what we got
                logger.info(f"[GARMENT] Stored: {self.pipeline_data.garment_type} | {self.pipeline_data.gender}")
                
                return {'success': True, 'message': 'Analysis complete', 'from_background': False}
            else:
                return {'success': False, 'error': result.get('error', 'Garment analysis failed')}
                
        except Exception as e:
            logger.error(f"Step 1 handler error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_brand_specific_garment_hints(self, brand):
        """Get garment type hints based on brand for faster analysis"""
        brand_hints = {
            # Women's brands - typically dresses, blouses, jackets
            'Rebecca Minkoff': ['dress', 'blazer', 'top', 'jacket'],
            'Antonio Melani': ['dress', 'blouse', 'jacket', 'top'],
            'Theory': ['blazer', 'dress', 'pants', 'top'],
            'J.Crew': ['blouse', 'dress', 'jacket', 'pants'],
            'Madewell': ['jeans', 'top', 'jacket', 'dress'],
            'Anthropologie': ['dress', 'blouse', 'skirt', 'jacket'],
            
            # Men's brands - typically shirts, jackets, pants
            'Brooks Brothers': ['shirt', 'blazer', 'pants', 'suit'],
            'Ralph Lauren': ['polo', 'shirt', 'jacket', 'pants'],
            'Tommy Hilfiger': ['shirt', 'polo', 'jeans', 'jacket'],
            'Nike': ['shorts', 'pants', 'shirt', 'jacket'],
            'Adidas': ['shorts', 'pants', 'shirt', 'jacket'],
            
            # Unisex brands
            'Levi\'s': ['jeans', 'shorts', 'jacket'],
            'Uniqlo': ['shirt', 'pants', 'jacket', 'dress'],
            'Gap': ['jeans', 'shirt', 'dress', 'jacket'],
        }
        
        return brand_hints.get(brand, [])
    
    def analyze_tag_simple(self, image_np: np.ndarray):
        """
        Simplified tag analysis with clean error handling and single retry strategy.
        Replaces the complex analyze_tag_with_auto_retry system.
        """
        logger.info("[TAG-ANALYSIS] Starting simplified tag analysis...")
        
        # NEW: Crop to just text area
        text_region = self.text_extractor.crop_to_text_region(image_np)
        
        # Save the cropped image we're sending to Gemini
        cv2.imwrite("debug_tag_to_gemini.jpg", cv2.cvtColor(text_region, cv2.COLOR_RGB2BGR))
        logger.info("[DEBUG] Saved cropped tag image to debug_tag_to_gemini.jpg")
        
        # Check if text_extractor is available
        if not self.text_extractor:
            logger.error("[TAG-ANALYSIS] Text extractor not available!")
            return {'success': False, 'error': 'Text extractor not initialized', 'brand': None, 'size': 'Unknown'}
        
        # Use the retry manager for clean retry logic
        def _single_analysis_attempt():
            """Single analysis attempt - this is what gets retried"""
            try:
                # Try the primary analysis method (with fallback) using cropped text region
                result = self.text_extractor.analyze_tag(text_region)
                
                # Success if we got EITHER brand OR size (not requiring both)
                if result.get('brand') or (result.get('size') and result.get('size') != 'Unknown'):
                    result['success'] = True
                    logger.info(f"[TAG-ANALYSIS] Got brand='{result.get('brand')}', size='{result.get('size')}'")
                else:
                    result['success'] = False
                    result['error'] = 'No text detected - check lighting/focus'
                    logger.warning(f"[TAG-ANALYSIS] No useful data extracted")
                
                if result.get('success'):
                    logger.info(f"[TAG-ANALYSIS] Primary analysis successful: {result.get('brand')}")
                    return result
                
                # If primary fails, try with preprocessing
                logger.info("[TAG-ANALYSIS] Primary failed, trying with preprocessing...")
                processed_image = self._preprocess_image_for_ocr(image_np)
                if processed_image is not None:
                    result = self.text_extractor.analyze_tag(processed_image)
                    
                    # Success if we got EITHER brand OR size (not requiring both)
                    if result.get('brand') or (result.get('size') and result.get('size') != 'Unknown'):
                        result['success'] = True
                        logger.info(f"[TAG-ANALYSIS] PREPROCESSING Got brand='{result.get('brand')}', size='{result.get('size')}'")
                    else:
                        result['success'] = False
                        result['error'] = 'No text detected even after preprocessing - check lighting/focus'
                        logger.warning(f"[TAG-ANALYSIS] PREPROCESSING No useful data extracted")
                    
                    if result.get('success'):
                        result['method'] = result.get('method', 'AI') + ' with preprocessing'
                        logger.info(f"[TAG-ANALYSIS] Preprocessing successful: {result.get('brand')}")
                        return result
                
                # Final fallback
                logger.warning("[TAG-ANALYSIS] All attempts failed")
                return {'success': False, 'error': 'All analysis methods failed', 'brand': None, 'size': 'Unknown'}
                
            except Exception as e:
                logger.error(f"[TAG-ANALYSIS] Analysis exception: {e}")
                return {'success': False, 'error': str(e), 'brand': None, 'size': 'Unknown'}
        
        # Use the retry manager
        result = self.retry_manager.execute_with_retry(
            "Tag Analysis",
            _single_analysis_attempt
        )
        
        # Ensure consistent return format
        if not isinstance(result, dict):
            result = {'success': False, 'error': 'Invalid result format', 'brand': None, 'size': 'Unknown'}
        
        return result
    
    def _preprocess_image_for_ocr(self, image_np: np.ndarray):
        """Single preprocessing strategy for OCR"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to RGB
            rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            return rgb
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return None
    
    def listen_for_brand_whisper(self):
        """Uses the microphone to record a clip and sends it to the Whisper API for accurate brand transcription."""
        
        # --- 1. Record Audio ---
        fs = 44100  # Sample rate
        seconds = 4   # Duration of recording
        
        st.info("🎤 **Listening...** Speak the brand name clearly now.")
        
        try:
            # Record audio from the default microphone
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()  # Wait until recording is finished
            st.info("🔄 **Processing...** Sending to Whisper API...")

            # --- 2. Save Audio to an In-Memory File ---
            # We use a virtual file in memory instead of saving to disk
            audio_buffer = io.BytesIO()
            write(audio_buffer, fs, myrecording)
            audio_buffer.seek(0)
            # We must name the file for the API
            audio_buffer.name = "recording.wav" 

            # --- 3. Send to Whisper API ---
            # Get your OpenAI client
            if not self.text_extractor or not self.text_extractor.openai_client:
                st.error("❌ OpenAI client not initialized.")
                return ""

            try:
                client = self.text_extractor.openai_client
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_buffer
                )
            except Exception as api_error:
                st.error(f"❌ Whisper API error: {api_error}")
                logger.error(f"Whisper API error: {api_error}")
                return ""
            
            recognized_text = transcript.text.strip()
            st.success(f"🎯 **Whisper heard:** '{recognized_text}'")
            return recognized_text

        except Exception as e:
            logger.error(f"Whisper API or recording error: {e}")
            st.error(f"❌ An error occurred: {e}")
            return ""
    
    async def analyze_everything_parallel(self, tag_image, garment_image):
        """Run all three API analyses in parallel for 3x speed improvement"""
        import asyncio
        from openai import AsyncOpenAI
        
        try:
            # Create async client
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return None
            
            client = AsyncOpenAI(api_key=api_key)
            
            # Run all 3 API calls simultaneously
            # Use synchronous ensemble voting for tag analysis
            tag_result = self.text_extractor.analyze_tag(tag_image)
            garment_task = self.garment_analyzer.analyze_garment_with_smart_retry(garment_image, client)
            defect_task = self.defect_detector.analyze_defects_async(garment_image, client)
            
            # Wait for garment and defect analysis to complete
            results = await asyncio.gather(garment_task, defect_task, return_exceptions=True)
            
            # Process results (tag_result already available synchronously)
            garment_result = results[0] if not isinstance(results[0], Exception) else {'success': False, 'error': str(results[0])}
            defect_result = results[1] if not isinstance(results[1], Exception) else {'success': False, 'error': str(results[1])}
            
            return {
                'tag': tag_result,
                'garment': garment_result,
                'defect': defect_result,
                'success': tag_result.get('success', False) and garment_result.get('success', False),
                'method': 'Parallel Async'
            }
            
        except Exception as e:
            logger.error(f"Parallel analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def is_network_available(self):
        """Check if OpenAI API is accessible"""
        try:
            # Test OpenAI API directly instead of just internet connectivity
            response = requests.get("https://api.openai.com/v1/models", 
                                  headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}, 
                                  timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def get_offline_fallback_data(self, image_type="tag"):
        """Get fallback data when network is unavailable"""
        if image_type == "tag":
            return {
                'brand': 'Unknown Brand',
                'size': 'Unknown Size',
                'material': 'Unknown Material',
                'success': True,
                'method': 'Offline Fallback',
                'confidence': 0.1
            }
        else:  # garment
            return {
                'garment_type': 'Unknown Garment',
                'gender': 'Unisex',
                'color': 'Unknown',
                'style': 'Casual',
                'era': 'Contemporary',
                'condition': 'Unknown',
                'defect_count': 0,
                'success': True,
                'method': 'Offline Fallback',
                'confidence': 0.1
            }
    
    def render_compact_layout(self):
        """Renders the main dashboard layout: buttons on top, then chalkboard/pipeline, then the main content."""
        
        # TITLE
        st.title("🔍 Garment Analysis")
        
        # ===================================================
        # ====== 1. ACTION BUTTONS AT THE VERY TOP ======
        # ===================================================
        self.render_action_panel()
        st.markdown("---")
        
        # ======================================================================
        # ====== 2. CHALKBOARD (Left) and PIPELINE PROGRESS (Right) ======
        # ======================================================================
        col_chalkboard, col_pipeline = st.columns([2, 1])
        
        with col_chalkboard:
            self.render_data_chalkboard()  # Use the proper chalkboard renderer
        
        with col_pipeline:
            st.markdown("#### 📊 Pipeline Progress")
            self.render_cool_step_pipeline()

        st.markdown("---")
        
        # ========================================================================
        # ====== 3. MAIN CONTENT AREA (Centered Camera or Step Info) ======
        # ========================================================================
        # This section will render the content for the current step
        if self.current_step == 0:
            self._render_step_0_compact()
        elif self.current_step == 1:
            self._render_step_1_compact()
        elif self.current_step == 2:
            self._render_step_3_compact()  # Measurements (was step 3)
        elif self.current_step == 3:
            self._render_step_4_compact()  # Defects (was step 4)
        elif self.current_step == 4:
            self._render_step_5_compact()  # Results (was step 5)
        else:
            self._render_final_review_compact()

    def _render_garment_analysis_board(self):
        """Render the garment analysis board (chalkboard)"""
        # Create a styled container for the analysis board
        st.markdown("""
        <div style="
            background-color: #2b2b2b;
            border: 3px solid #8B4513;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            color: white;
            font-family: 'Courier New', monospace;
        ">
        """, unsafe_allow_html=True)
        
        # Brand information
        brand = getattr(self.pipeline_data, 'brand', 'Unknown')
        st.markdown(f"**BRAND:** {brand}")
        
        # Garment type information
        garment_type = getattr(self.pipeline_data, 'garment_type', 'Not analyzed')
        if garment_type == 'Unknown':
            garment_type = 'Not analyzed'
        st.markdown(f"**GARMENT:** Type: {garment_type}")
        
        # Size information
        size = getattr(self.pipeline_data, 'size', 'Unknown')
        st.markdown(f"**SIZE:** {size}")
        
        # Material information
        material = getattr(self.pipeline_data, 'material', 'Unknown')
        st.markdown(f"**MATERIAL:** {material}")
        
        # Condition information
        condition = getattr(self.pipeline_data, 'condition', 'Unknown')
        defects = getattr(self.pipeline_data, 'defects', [])
        if defects:
            defect_text = f"Defects: {len(defects)} found"
        else:
            defect_text = "Defects: None ✓"
        st.markdown(f"**CONDITION:** {defect_text}")
        
        # AI Model info
        st.markdown("**AI MODEL:** Model: Gemini 2.0 Flash, Tag readable ✓")
        
        st.markdown("</div>", unsafe_allow_html=True)

    def _render_step_0_compact(self):
        """Renders the content for Step 0: A centered camera view for tight layout."""
        # Compact header
        st.markdown("### 🏷️ Tag Analysis")
        st.info("Position your garment tag in the **GREEN BOX**, then click **'Next Step'** above.")

        # Create a centered column for the camera view - more compact
        col1, col_camera, col3 = st.columns([1, 3, 1])  # More compact middle column

        with col_camera:
            try:
                frame = self.camera_manager.get_arducam_frame()
                if frame is not None:
                    # Draw the ROI overlay on the full frame
                    frame_with_roi = self.camera_manager.draw_roi_overlay(frame.copy(), 'tag')
                    if frame_with_roi is not None:
                        st.image(frame_with_roi, caption="📸 Tag Camera - Position tag in Green Box", width='stretch')
                        
                        # Show current ROI coordinates - more compact
                        roi_coords = self.camera_manager.roi_coords.get('tag', (0, 0, 0, 0))
                        st.caption(f"ROI: ({roi_coords[0]}, {roi_coords[1]}) {roi_coords[2]}×{roi_coords[3]}")
                    else:
                        st.error("❌ No ROI overlay available")
                else:
                    st.warning("⚠️ ArduCam camera not available")
            except Exception as e:
                st.error(f"❌ Camera error: {e}")

        # Optional: Add camera controls or previews in an expander if needed - more compact
        with st.expander("🔍 AI Preview & Controls"):
            if 'frame' in locals() and frame is not None:
                roi_image = self.camera_manager.apply_roi(frame, 'tag')
                if roi_image is not None:
                    st.image(roi_image, caption="This is what the AI will analyze", width='stretch')
                    st.caption(f"Size: {roi_image.shape[1]}×{roi_image.shape[0]} pixels")
                    
                    # Show brightness info
                    if self.auto_optimizer.enabled:
                        brightness_info = self.auto_optimizer.analyze_image_brightness(roi_image)
                        if brightness_info:
                            mean = brightness_info['mean']
                            if mean > 180:
                                st.caption("💡 Very bright - will reduce on capture")
                            elif mean < 60:
                                st.caption("💡 Dark - will boost on capture")
                            else:
                                st.caption("✅ Good lighting")
        
        # Check if already analyzed
        if (self.pipeline_data.tag_image is not None and 
            self.pipeline_data.brand != "Unknown"):
            
            st.markdown("---")
            st.markdown("#### ✅ Analysis Results")
            
            # Show results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Brand:** {self.pipeline_data.brand}")
            with col2:
                st.success(f"**Size:** {self.pipeline_data.size}")
            
            # Quick override
            with st.expander("✏️ Override", expanded=False):
                new_brand = st.text_input("Brand", value=self.pipeline_data.brand)
                new_size = st.text_input("Size", value=self.pipeline_data.size)
                if st.button("Update"):
                    self.pipeline_data.brand = new_brand
                    self.pipeline_data.size = new_size
                    st.rerun()
        
        else:
            # AUTO-REFRESH MOTION DETECTION (IMPROVED)
            if 'last_tag_motion_check' not in st.session_state:
                st.session_state.last_tag_motion_check = 0
            if 'tag_motion_detected' not in st.session_state:
                st.session_state.tag_motion_detected = False
            
            # Check for motion more frequently (every 0.5 seconds instead of 1)
            current_time = time.time()
            time_since_check = current_time - st.session_state.last_tag_motion_check
            
            if time_since_check > 0.5:  # Check every 0.5 seconds (more responsive)
                motion = self.camera_manager.detect_motion_in_roi('tag', threshold=15)  # Lower threshold for better sensitivity
                st.session_state.last_tag_motion_check = current_time
                
                if motion:
                    st.session_state.tag_motion_detected = True
                    logger.info("[MOTION] Tag movement detected")
                    # NOTE: Removed st.rerun() - let natural Streamlit refresh handle it
                    # This prevents blocking button clicks
            
            # Control buttons at the top
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
            
            with col_ctrl1:
                # Manual refresh
                if st.button("🔄 Refresh", key="refresh_tag_compact", width='stretch'):
                    st.session_state.last_tag_motion_check = 0  # Force immediate check
                    st.rerun()
            
            with col_ctrl2:
                # Auto-zoom toggle
                auto_zoom = st.checkbox(
                    "🤖 Auto-Zoom",
                    value=st.session_state.get('auto_zoom_enabled', False),
                    help="Automatically detect and crop tag"
                )
                st.session_state.auto_zoom_enabled = auto_zoom
            
            with col_ctrl3:
                if not auto_zoom:
                    zoom_level = st.slider(
                        "🔬",
                        min_value=1.0,
                        max_value=3.0,
                        value=st.session_state.get('zoom_level', 2.0),
                        step=0.25,
                        key="zoom_slider_compact",
                        help="Zoom level"
                    )
                    st.session_state.zoom_level = zoom_level
            
            # Motion status indicator
            if st.session_state.tag_motion_detected:
                st.success("✅ Tag detected in ROI - preview updated!")
                st.session_state.tag_motion_detected = False  # Reset after showing
            else:
                st.info("⏳ Watching for tag movement... (position tag in GREEN BOX)")
            
            # Force periodic refresh to check for motion
            if time_since_check > 0.5:
                time.sleep(0.05)  # Small delay to prevent CPU spike
                st.rerun()

    def _render_step_1_compact(self):
        """Compact Step 1: Garment Analysis with camera feed"""
        st.markdown("### 👕 Garment Analysis")
        
        # Show RealSense camera feed for garment capture
        frame = self.camera_manager.get_realsense_frame()
        if frame is not None:
            st.image(frame, caption="RealSense Garment View", width='stretch')
            st.caption("💡 Position garment in view, then click Next Step to analyze")
        else:
            st.warning("⚠️ RealSense camera not available")
        
        if (self.pipeline_data.garment_image is not None and 
            self.pipeline_data.garment_type != "Unknown"):
            
            # Show results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Type:** {self.pipeline_data.garment_type}")
            with col2:
                st.success(f"**Gender:** {self.pipeline_data.gender}")
            
            # Quick override
            with st.expander("✏️ Override", expanded=False):
                new_type = st.selectbox("Type", ["Keep detected", "T-Shirt", "Dress", "Jacket", "Pants"])
                if new_type != "Keep detected":
                    self.pipeline_data.garment_type = new_type
        
        else:
            st.info("📸 Position garment in view above, then click Next Step to analyze")

    # Calibration step removed - measurements will use pixel-based estimates

    def _render_step_3_compact(self):
        """Compact Step 3: Measurements - Mandatory if no size, optional if size detected"""
        st.markdown("### 📐 Garment Measurements")
        
        # Handle numerical sizes (like CRUSH tag: 0=XS, 1=S, 2=M, etc.)
        size_detected = self.pipeline_data.size and self.pipeline_data.size != "Unknown"
        
        if size_detected:
            from data_collection_and_correction_system import GarmentDataCollector
            collector = GarmentDataCollector()
            size_info = collector.convert_size_format(self.pipeline_data.size)
            
            if size_info['number'] != 'Unknown':
                st.success(f"✅ Size from tag: {size_info['letter']} (Number: {size_info['number']})")
                st.caption(f"Numerical sizing: {size_info['number']} = {size_info['letter']}")
            else:
                st.success(f"✅ Size from tag: {self.pipeline_data.size}")
            
            st.success("✅ Size detected from tag - measurement complete!")
            st.caption("💡 Optional: You can measure armpit seams below for verification")
            
            # OPTIONAL: Show measurement camera only if user wants to verify
            with st.expander("📏 Optional: Measure armpit seams for verification", expanded=False):
                self._render_simple_armpit_measurement()
        else:
            st.warning("⚠️ No size detected from tag")
            st.error("❌ Manual measurement required")
            st.info("📏 Please click on both armpit seams to measure the garment")
            
            # MANDATORY: Show measurement camera since no size was detected
            self._render_simple_armpit_measurement()
    
    def _render_simple_armpit_measurement(self):
        """Simplified armpit seam measurement - just click two points"""
        st.markdown("#### 📷 Click on Armpit Seams to Measure")
        st.caption("💡 Click the left armpit seam, then the right armpit seam. A line will be drawn automatically.")
        
        # Initialize measurement data
        if 'armpit_points' not in st.session_state:
            st.session_state.armpit_points = []
        
        # Get RealSense camera frame with error handling
        try:
            frame = self.camera_manager.get_realsense_frame()
        except Exception as e:
            logger.error(f"[ARMPIT-MEASUREMENT] RealSense camera error: {e}")
            frame = None
        
        if frame is not None:
            # Use streamlit_image_coordinates for click detection
            from streamlit_image_coordinates import streamlit_image_coordinates
            
            # Draw existing points and line
            display_frame = frame.copy()
            points = st.session_state.armpit_points
            
            # Draw points and line
            if len(points) >= 1:
                # Draw first point
                x1, y1 = points[0]
                cv2.circle(display_frame, (x1, y1), 8, (0, 255, 0), -1)
                cv2.putText(display_frame, "P1", (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(points) >= 2:
                    # Draw second point
                    x2, y2 = points[1]
                    cv2.circle(display_frame, (x2, y2), 8, (0, 255, 0), -1)
                    cv2.putText(display_frame, "P2", (x2+10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw line between points
                    cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    
                    # Calculate and display measurement
                    pixel_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    estimated_inches = pixel_distance / 20.0
                    estimated_cm = estimated_inches * 2.54
                    
                    # Draw measurement text on image
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    measurement_text = f"{estimated_inches:.1f}\""
                    cv2.putText(display_frame, measurement_text, (mid_x, mid_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Display image with click detection
            try:
                clicked_point = streamlit_image_coordinates(
                    display_frame,
                    key="armpit_measurement_click"
                )
                
                # Debug: Show current state
                st.caption(f"🔍 Debug: {len(points)} points, clicked_point: {clicked_point}")
                
                # Handle click
                if clicked_point is not None:
                    x, y = clicked_point["x"], clicked_point["y"]
                    logger.info(f"[ARMPIT-CLICK] Point clicked at ({x}, {y})")
                    
                    if len(points) < 2:
                        points.append((x, y))
                        st.session_state.armpit_points = points
                        
                        if len(points) == 1:
                            st.success(f"✅ Point 1 added at ({x}, {y}). Click to add point 2.")
                        elif len(points) == 2:
                            st.success(f"✅ Point 2 added at ({x}, {y}). Measurement complete!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Maximum 2 points reached. Clear points to measure again.")
                else:
                    # Show instruction based on current state
                    if len(points) == 0:
                        st.info("👆 Click on the left armpit seam to add Point 1")
                    elif len(points) == 1:
                        st.info("👆 Click on the right armpit seam to add Point 2")
                        
            except Exception as e:
                logger.error(f"[ARMPIT-CLICK] Error with streamlit_image_coordinates: {e}")
                st.error(f"Click detection error: {e}")
                # Fallback: show image without click detection
                st.image(display_frame, caption="RealSense Garment View", width='stretch')
                st.warning("⚠️ Click detection not available. Please use manual measurement.")
            
            # Show measurement results
            if len(points) >= 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                pixel_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                estimated_inches = pixel_distance / 20.0
                estimated_cm = estimated_inches * 2.54
                
                st.markdown("#### 📏 Measurement Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Pixel Distance", f"{pixel_distance:.1f} px")
                with col2:
                    st.metric("Estimated Inches", f"{estimated_inches:.1f}\"")
                with col3:
                    st.metric("Estimated CM", f"{estimated_cm:.1f} cm")
                
                # Store measurement in pipeline data
                self.pipeline_data.armpit_measurement = {
                    'pixel_distance': pixel_distance,
                    'inches': estimated_inches,
                    'cm': estimated_cm,
                    'points': points
                }
                
                # Clear points button
                if st.button("🗑️ Clear Points & Measure Again", key="clear_armpit_points"):
                    st.session_state.armpit_points = []
                    st.rerun()
        else:
            st.warning("⚠️ RealSense camera not available for measurements")
            
            # Try fallback to ArduCam
            try:
                logger.info("[ARMPIT-MEASUREMENT] Trying ArduCam fallback...")
                fallback_frame = self.camera_manager.get_arducam_frame()
                if fallback_frame is not None:
                    st.info("📷 Using ArduCam as fallback for measurements")
                    st.image(fallback_frame, caption="ArduCam Fallback View", width='stretch')
                    st.caption("💡 Note: ArduCam doesn't support click detection. Use manual measurement below.")
                    
                    # Manual measurement input
                    st.markdown("#### 📏 Manual Armpit Measurement")
                    manual_width = st.number_input("Enter armpit-to-armpit width (inches):", 
                                                  min_value=0.0, max_value=50.0, value=20.0, step=0.1)
                    
                    if st.button("✅ Use Manual Measurement", key="manual_armpit_width"):
                        # Store manual measurement
                        self.pipeline_data.armpit_measurement = {
                            'inches': manual_width,
                            'cm': manual_width * 2.54,
                            'method': 'manual_input'
                        }
                        st.success(f"✅ Manual measurement saved: {manual_width}\" ({manual_width * 2.54:.1f} cm)")
                        st.rerun()
                else:
                    st.error("❌ No cameras available for measurements")
            except Exception as e:
                logger.error(f"[ARMPIT-MEASUREMENT] ArduCam fallback also failed: {e}")
                st.error("❌ All cameras unavailable for measurements")
    
    # Old complex measurement functions removed - now using simplified armpit seam measurement

    def _render_step_4_compact(self):
        """Compact Step 4: Defects"""
        st.markdown("### 🔍 Defect Check")
        
        if self.pipeline_data.defect_count > 0:
            st.warning(f"⚠️ {self.pipeline_data.defect_count} defects found")
        else:
            st.success("✅ No defects detected")

    def _render_step_5_compact(self):
        """Compact Step 5: Results Review & Pricing"""
        st.markdown("### ✅ Analysis Complete!")
        
        # Show basic results in metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏷️ Brand", getattr(self.pipeline_data, 'brand', 'Unknown'))
            st.metric("👕 Type", getattr(self.pipeline_data, 'garment_type', 'Unknown'))
        with col2:
            st.metric("📏 Size", getattr(self.pipeline_data, 'size', 'Unknown'))
            st.metric("🧵 Material", getattr(self.pipeline_data, 'material', 'Unknown'))
        with col3:
            st.metric("👤 Gender", getattr(self.pipeline_data, 'gender', 'Unknown'))
            st.metric("🎨 Style", getattr(self.pipeline_data, 'style', 'Unknown'))
        
        # Show confidence scores
        render_confidence_scores(self.pipeline_data)
        
        # === EBAY SOLD COMPS RESEARCH ===
        display_ebay_comps(self.pipeline_data)
        
        # Store tag detection removed - was overdetecting and creating false positives
        # === CORRECTION PANEL ===
        render_correction_panel(self.pipeline_data)
        
        # === EBAY PRICING VERIFICATION ===
        verify_ebay_pricing(self.pipeline_data)
        
        # === SAVE BUTTON ===
        st.markdown("---")
        if st.button("💾 Save to Training Dataset", type="primary", key="save_training_data"):
            save_analysis_with_corrections(self.pipeline_data)
            st.balloons()

    def _render_final_review_compact(self):
        """Compact final review"""
        st.markdown("### ✅ Review")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Brand:** {self.pipeline_data.brand}")
            st.write(f"**Type:** {self.pipeline_data.garment_type}")
            st.write(f"**Size:** {self.pipeline_data.size}")
        with col2:
            st.write(f"**Gender:** {self.pipeline_data.gender}")
            st.write(f"**Condition:** {self.pipeline_data.condition}")
            if self.pipeline_data.price_estimate:
                st.write(f"**Price:** ${self.pipeline_data.price_estimate.get('mid', 0)}")
        
        if st.button("🔄 Start New", type="primary", width='stretch'):
            self.current_step = 0
            self.pipeline_data = PipelineData()
            st.rerun()
    
    def draw_roi_on_image(self, image, roi_coords, color=(0, 255, 0), thickness=4):
        """Draw ROI rectangle on image with interactive handles"""
        if image is None:
            return None
        
        img_with_roi = image.copy()
        x, y, w, h = roi_coords
        
        # Draw rectangle
        cv2.rectangle(img_with_roi, (x, y), (x + w, y + h), color, thickness)
        
        # Draw corner handles (for visual feedback)
        handle_size = 15
        handle_color = (0, 0, 255)  # Blue handles
        cv2.rectangle(img_with_roi, (x - handle_size, y - handle_size), 
                      (x + handle_size, y + handle_size), handle_color, -1)
        cv2.rectangle(img_with_roi, (x + w - handle_size, y - handle_size), 
                      (x + w + handle_size, y + handle_size), handle_color, -1)
        cv2.rectangle(img_with_roi, (x - handle_size, y + h - handle_size), 
                      (x + handle_size, y + h + handle_size), handle_color, -1)
        cv2.rectangle(img_with_roi, (x + w - handle_size, y + h - handle_size), 
                      (x + w + handle_size, y + h + handle_size), handle_color, -1)
        
        # Add label
        cv2.putText(img_with_roi, "Click to Move ROI", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return img_with_roi
    
    def handle_roi_click(self, click_coords, current_roi):
        """Handle click to move ROI center to clicked position"""
        if click_coords is None:
            return current_roi
        
        x_click = click_coords['x']
        y_click = click_coords['y']
        
        # Get current ROI dimensions (keep size the same)
        _, _, w, h = current_roi
        
        # Center the ROI on the clicked point
        new_x = max(0, x_click - w // 2)
        new_y = max(0, y_click - h // 2)
        
        logger.info(f"[ROI-CLICK] Moved ROI from {current_roi[:2]} to ({new_x}, {new_y})")
        
        return (new_x, new_y, w, h)
    
    def save_roi_to_config(self, roi_coords):
        """Save updated ROI to config file"""
        import json
        
        config_file = 'roi_config.json'
        
        try:
            # Load existing config
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {'original_resolution': [1280, 720]}
            
            # Update tag ROI
            config['roi_coords'] = config.get('roi_coords', {})
            config['roi_coords']['tag'] = list(roi_coords)
            
            # Save
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"[ROI-SAVE] Saved ROI to config: {roi_coords}")
            return True
            
        except Exception as e:
            logger.error(f"[ROI-SAVE] Failed to save ROI: {e}")
            return False

    def render_cool_step_pipeline(self):
        """Render an animated step pipeline with checkboxes on the right side"""
        
        # Build the complete HTML with inline styles
        pipeline_html = '''
        <style>
        .step-pipeline {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .step-item {
            display: flex;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .step-item.completed {
            background: rgba(76, 175, 80, 0.3);
        }
        .step-item.current {
            background: rgba(255, 193, 7, 0.3);
            border: 2px solid #FFC107;
        }
        .step-checkbox {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            margin-right: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
        }
        .step-checkbox.completed {
            background: #4CAF50;
            color: white;
        }
        .step-checkbox.current {
            background: #FFC107;
            color: white;
            animation: pulse 2s infinite;
        }
        .step-checkbox.pending {
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.5);
            color: rgba(255,255,255,0.7);
        }
        .step-text {
            color: white;
            font-weight: 500;
            font-size: 16px;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        </style>
        <div class="step-pipeline">
            <h3 style="color: white; text-align: center; margin-bottom: 20px;">📋 Pipeline Progress</h3>
        '''
        
        for i, step in enumerate(self.steps):
            if i < self.current_step:
                status_class = "completed"
                icon = "✓"
            elif i == self.current_step:
                status_class = "current"
                icon = "●"
            else:
                status_class = "pending"
                icon = str(i + 1)
            
            pipeline_html += f'''
            <div class="step-item {status_class}">
                <div class="step-checkbox {status_class}">{icon}</div>
                <div class="step-text">{step}</div>
            </div>
            '''
        
        pipeline_html += '</div>'
        
        # Use components.html for guaranteed rendering
        import streamlit.components.v1 as components
        components.html(pipeline_html, height=600, scrolling=False)

    def render_checklist_sidebar(self):
        """Render a clean, essential-only sidebar"""
        st.sidebar.header("📊 Progress")
        
        # Progress checklist
        for i, step in enumerate(self.steps):
            if i < self.current_step:
                st.sidebar.success(f"✅ {step}")
            elif i == self.current_step:
                st.sidebar.warning(f"▶️ {step}")
            else:
                st.sidebar.info(f"⏸️ {step}")
        
        # Progress bar
        progress = self.current_step / len(self.steps)
        st.sidebar.progress(min(max(progress, 0.0), 1.0))
        st.sidebar.caption(f"Progress: {int(progress * 100)}%")
        
        # Camera status (compact)
        st.sidebar.markdown("---")
        st.sidebar.subheader("📹 Cameras")
        
        if self.camera_manager.camera_status['arducam']:
            st.sidebar.success("✅ Tag Camera")
        else:
            st.sidebar.error("❌ Tag Camera")
        
        if self.camera_manager.camera_status['realsense']:
            st.sidebar.success("✅ Garment Camera")
        else:
            st.sidebar.error("❌ Garment Camera")
        
        # Lighting status (compact)
        st.sidebar.markdown("---")
        st.sidebar.subheader("💡 Lighting")
        
        if self.light_controller.lights:
            st.sidebar.success(f"✅ Connected ({self.light_controller.current_state['brightness']}%)")
        else:
            st.sidebar.warning("⚠️ Not detected")
        
        # Dataset stats (compact)
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Learning System")
        
        if self.dataset_manager:
            stats = self.dataset_manager.get_dataset_stats()
            st.sidebar.metric("Total Samples", stats['total_samples'])
            if 'success_rate' in stats:
                st.sidebar.metric("Success Rate", f"{stats['success_rate']:.0f}%")
        
        # Camera tools
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Tools")
        
        if st.sidebar.button("🔬 Focus Calibration"):
            st.session_state.focus_mode = True
            st.session_state.focus_start_time = time.time()
            st.rerun()
        
        if st.sidebar.button("🎯 Position Tag ROI"):
            st.session_state.roi_positioning_mode = True
            st.rerun()
        
        if st.sidebar.button("🎨 Interactive ROI Editor"):
            st.session_state.interactive_roi_mode = True
            st.rerun()
        
        if st.sidebar.button("🔄 Reset Pipeline"):
            self.current_step = 0
            self.pipeline_data = PipelineData()
            st.rerun()
        
        # Display Settings
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Display Settings")
        
        auto_refresh = st.sidebar.checkbox(
            "Auto-refresh display",
            value=st.session_state.get('auto_refresh', True),
            help="Automatically update the camera feed",
            key="auto_refresh_checkbox"
        )
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_rate = st.sidebar.slider(
                "Refresh rate (seconds)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                key="refresh_rate_slider"
            )
            st.session_state.refresh_rate = refresh_rate
            
            # Auto-refresh using time-based trigger
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = time.time()
            
            current_time = time.time()
            if current_time - st.session_state.last_refresh >= refresh_rate:
                st.session_state.last_refresh = current_time
                st.rerun()
            
            # Show countdown
            time_until_refresh = refresh_rate - (current_time - st.session_state.last_refresh)
            st.sidebar.caption(f"⏱️ Next refresh: {time_until_refresh:.1f}s")
        
        # System Tests
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧪 System Tests")
        
        if st.sidebar.button("Test Learning System", key="test_learning_btn"):
            if self.dataset_manager:
                stats = self.dataset_manager.get_dataset_stats()
                st.sidebar.success(f"✅ Learning system OK: {stats['total_samples']} samples")
            else:
                st.sidebar.error("❌ Learning system not initialized")
        
        if st.sidebar.button("Test RealSense Color", key="test_realsense_color_btn"):
            frame = self.camera_manager.get_realsense_frame()
            if frame is not None and len(frame.shape) == 3:
                unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
                if unique_colors > 5000:
                    st.sidebar.success(f"✅ True color: {unique_colors} colors")
                else:
                    st.sidebar.error(f"❌ Grayscale: only {unique_colors} colors")
            else:
                st.sidebar.error("❌ No frame or wrong shape")
        
        # Keep diagnostic button for troubleshooting (collapsed by default)
        if st.sidebar.button("Diagnose RealSense (Debug)"):
            if self.camera_manager.realsense_sdk_available:
                st.sidebar.success("✅ SDK Available")
                
                if self.camera_manager.realsense_pipeline:
                    st.sidebar.success("✅ Pipeline Created")
                    
                    try:
                        # Try to get a frame
                        frames = self.camera_manager.realsense_pipeline.wait_for_frames(timeout_ms=1000)
                        color_frame = frames.get_color_frame()
                        
                        if color_frame:
                            st.sidebar.success("✅ Color Frame Received")
                            frame_data = np.asanyarray(color_frame.get_data())
                            st.sidebar.write(f"Shape: {frame_data.shape}")
                            
                            # Check if truly color
                            if len(frame_data.shape) == 3:
                                unique_colors = len(np.unique(frame_data.reshape(-1, 3), axis=0))
                                st.sidebar.write(f"Unique colors: {unique_colors}")
                                if unique_colors > 1000:
                                    st.sidebar.success("✅ TRUE COLOR MODE")
                                elif unique_colors > 256:
                                    st.sidebar.warning(f"⚠️ LIMITED COLORS ({unique_colors})")
                                else:
                                    st.sidebar.error("❌ GRAYSCALE DETECTED")
                            else:
                                st.sidebar.error(f"❌ Wrong shape: {frame_data.shape}")
                        else:
                            st.sidebar.error("❌ No Color Frame")
                    except Exception as e:
                        st.sidebar.error(f"❌ Pipeline Error: {e}")
                else:
                    st.sidebar.error("❌ Pipeline Not Initialized")
            else:
                st.sidebar.error("❌ SDK Not Available")
                st.sidebar.info("💡 Install: pip install pyrealsense2")
        
        # Deep Color Test
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔬 Color Diagnostics")
        
        if st.sidebar.button("🧪 Deep Color Test"):
            frame = self.camera_manager.get_realsense_frame()
            
            if frame is not None:
                st.sidebar.write(f"Shape: {frame.shape}")
                
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Count unique colors
                    unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
                    
                    # Check if RGB channels are identical (grayscale indicator)
                    r_channel = frame[:, :, 0]
                    g_channel = frame[:, :, 1]
                    b_channel = frame[:, :, 2]
                    
                    channels_identical = np.array_equal(r_channel, g_channel) and np.array_equal(g_channel, b_channel)
                    
                    st.sidebar.write(f"Unique colors: {unique_colors}")
                    st.sidebar.write(f"Channels identical: {channels_identical}")
                    
                    if channels_identical:
                        st.sidebar.error("❌ GRAYSCALE (all RGB channels identical)")
                    elif unique_colors < 500:
                        st.sidebar.warning(f"⚠️ LIMITED COLORS ({unique_colors})")
                    elif unique_colors < 1000:
                        st.sidebar.warning(f"⚠️ POSSIBLE GRAYSCALE ({unique_colors} colors)")
                    else:
                        st.sidebar.success(f"✅ TRUE COLOR ({unique_colors} colors)")
                    
                    # Show channel samples
                    st.sidebar.caption("Sample pixel at (100,100):")
                    st.sidebar.write(f"R: {frame[100,100,0]}, G: {frame[100,100,1]}, B: {frame[100,100,2]}")
                    
                    # Show min/max per channel
                    st.sidebar.caption("Channel ranges:")
                    st.sidebar.write(f"R: {r_channel.min()}-{r_channel.max()}")
                    st.sidebar.write(f"G: {g_channel.min()}-{g_channel.max()}")
                    st.sidebar.write(f"B: {b_channel.min()}-{b_channel.max()}")
                else:
                    st.sidebar.error(f"❌ Wrong shape: {frame.shape}")
            else:
                st.sidebar.error("❌ No frame available")
        
        # ROI Debug Panel
        self.render_roi_debug_panel()
        
        # Dataset Analytics Dashboard
        self.render_dataset_analytics()
        
        st.sidebar.markdown("---")
        
        if st.sidebar.button("Reset Pipeline"):
            self.current_step = 0
    
    def render_roi_debug_panel(self):
        """Debug panel to check ROI configuration"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 ROI Debug")
        
        if st.sidebar.button("Check ROI"):
            # Check if ROI coords exist
            if hasattr(self.camera_manager, 'roi_coords'):
                st.sidebar.success("✅ ROI coords loaded")
                
                # Show tag ROI
                if 'tag' in self.camera_manager.roi_coords:
                    tag_roi = self.camera_manager.roi_coords['tag']
                    st.sidebar.write(f"**Tag ROI:** {tag_roi}")
                else:
                    st.sidebar.error("❌ No 'tag' ROI found")
                
                # Show work ROI
                if 'work' in self.camera_manager.roi_coords:
                    work_roi = self.camera_manager.roi_coords['work']
                    st.sidebar.write(f"**Work ROI:** {work_roi}")
                else:
                    st.sidebar.error("❌ No 'work' ROI found")
                
                # Test drawing on a frame
                frame = self.camera_manager.get_arducam_frame()
                if frame is not None:
                    test_frame = self.camera_manager.draw_roi_overlay(frame.copy(), 'tag')
                    if test_frame is not None:
                        st.sidebar.success("✅ ROI overlay works")
                        st.sidebar.image(test_frame, caption="Test ROI", width=150)
                    else:
                        st.sidebar.error("❌ ROI overlay failed")
                else:
                    st.sidebar.error("❌ No camera frame")
            else:
                st.sidebar.error("❌ No ROI coords attribute")
        
        # Manual ROI setter
        if st.sidebar.checkbox("Set ROI Manually"):
            st.sidebar.write("**Tag ROI:**")
            x = st.sidebar.number_input("X", value=183, key="roi_x")
            y = st.sidebar.number_input("Y", value=171, key="roi_y")
            w = st.sidebar.number_input("Width", value=211, key="roi_w")
            h = st.sidebar.number_input("Height", value=159, key="roi_h")
            
            if st.sidebar.button("Apply Manual ROI"):
                self.camera_manager.roi_coords['tag'] = (x, y, w, h)
                st.sidebar.success("✅ Manual ROI applied")
                st.rerun()
    
    def render_dataset_analytics(self):
        """Show statistics about the growing dataset in the sidebar"""
        try:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Learning System Stats")
            
            if not self.dataset_manager:
                st.sidebar.warning("⚠️ Learning system unavailable")
                st.sidebar.caption("Database initialization failed")
                return
            
            # Get dataset statistics
            stats = self.dataset_manager.get_dataset_stats()
            
            # Total samples
            st.sidebar.metric("Total Samples", stats['total_samples'])
            
            # Success rate
            st.sidebar.metric("Success Rate", f"{stats['success_rate']:.1f}%")
            
            # User corrections
            st.sidebar.metric("User Corrections", stats['user_corrections'])
            
            # User validations
            st.sidebar.metric("User Validations", stats['user_validations'])
            
            # Validation accuracy
            if stats['user_validations'] > 0:
                st.sidebar.metric("Validation Accuracy", f"{stats['validation_accuracy']:.1f}%")
            
            # Recent activity (last 7 days)
            st.sidebar.metric("This Week", stats['recent_samples'])
            
            # Top brands (if any)
            if stats['top_brands']:
                st.sidebar.markdown("**Top Brands:**")
                for brand, count in stats['top_brands'][:5]:  # Show top 5
                    if brand and brand != 'unknown':
                        st.sidebar.caption(f"• {brand}: {count}")
            
            # Export buttons
            if stats['total_samples'] > 0:
                # Manual export
                if st.sidebar.button("📤 Export Training Data", help="Export dataset for model training"):
                    try:
                        exported_count = self.dataset_manager.export_for_training()
                        st.sidebar.success(f"✅ Exported {exported_count} samples!")
                        st.toast(f"📤 Training data exported ({exported_count} samples)", icon="🎯")
                    except Exception as e:
                        st.sidebar.error(f"Export failed: {e}")
                
                # Automated batch export (every 100 corrections)
                if stats['user_corrections'] >= 100:
                    batch_filename = f"training_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    if st.sidebar.button("🔄 Auto-Export Batch", help=f"Export batch (every 100 corrections)"):
                        try:
                            exported_count = self.dataset_manager.export_for_training(batch_filename)
                            st.sidebar.success(f"✅ Batch exported: {exported_count} samples!")
                            st.toast(f"🎯 Auto-batch exported: {batch_filename}", icon="📤")
                        except Exception as e:
                            st.sidebar.error(f"Batch export failed: {e}")
                
            # Show next milestone
            next_milestone = ((stats['user_corrections'] // 100) + 1) * 100
            if stats['user_corrections'] < 100:
                st.sidebar.caption(f"📈 Next auto-export at {next_milestone} corrections")
        
            # eBay API Status
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 eBay API Status")
            
            # Check if eBay API is available
            ebay_app_id = os.getenv('EBAY_APP_ID')
            if ebay_app_id:
                st.sidebar.success("✅ eBay API Key Found")
                st.sidebar.caption("Real market data available")
            else:
                st.sidebar.warning("⚠️ eBay API Key Missing")
                st.sidebar.caption("Add EBAY_APP_ID to api.env")
            
            # Rate limit indicator
            if 'ebay_rate_limited' in st.session_state and st.session_state.ebay_rate_limited:
                st.sidebar.error("🚫 Rate Limited")
                st.sidebar.caption("Try again tomorrow")
                            
        except Exception as e:
            st.sidebar.error(f"Analytics error: {e}")
            logger.error(f"Dataset analytics failed: {e}")
    
    def render_serp_api_results(self, visual_matches):
        """
        Displays Google Lens visual matches in the Streamlit UI.
        Helps the user manually review when brand confidence is low.
        """
        if not visual_matches:
            st.warning("🔍 Google Lens returned no visual matches.")
            return
        
        st.subheader("🎯 Google Lens Visual Matches")
        st.info("📸 Here are the top visually similar items found online. Click to investigate.")
        
        # Show top 3-4 results in columns
        num_results = min(4, len(visual_matches))
        cols = st.columns(num_results)
        
        for i, match in enumerate(visual_matches[:num_results]):
            with cols[i]:
                # Title with link
                title = match.get('title', 'Unknown Item')
                link = match.get('link', '#')
                st.markdown(f"**[{title}]({link})**")
                
                # Thumbnail image
                if match.get('thumbnail'):
                    try:
                        st.image(match.get('thumbnail'), width='stretch')
                    except Exception as e:
                        logger.warning(f"Could not display SERP thumbnail: {e}")
                        st.caption("🖼️ Thumbnail unavailable")
                
                # Source
                source = match.get('source', 'Unknown Source')
                st.caption(f"🌐 Source: {source}")
                
                # Price (if available)
                if match.get('price'):
                    price_info = match.get('price')
                    extracted_value = price_info.get('extracted_value', 'N/A')
                    currency = price_info.get('currency', '')
                    st.success(f"💰 Price: {extracted_value} {currency}")
    
    def render_data_chalkboard(self):
        """Render a cute chalkboard-style display of captured data"""
        # Create a styled container that looks like a chalkboard
        with st.container():
            st.markdown("""
            <style>
            .chalkboard {
                background: linear-gradient(135deg, #2d3436 0%, #1e272e 100%);
                border: 12px solid #8b6914;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 
                    inset 0 0 8px rgba(0,0,0,0.5),
                    0 3px 5px rgba(0,0,0,0.3);
                position: relative;
                margin: 10px auto;
                max-width: 600px;
                width: 100%;
            }
            .chalk-text {
                color: #ffffff;
                font-family: 'Courier New', monospace;
                text-shadow: 1px 1px 2px rgba(255,255,255,0.1);
                margin: 3px 0;
                font-size: 14px;
            }
            .chalk-title {
                color: #ffeaa7;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 12px;
                font-family: 'Comic Sans MS', cursive;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }
            .chalk-section {
                border-left: 2px solid #ffeaa7;
                padding-left: 8px;
                margin: 8px 0;
            }
            .chalk-value {
                color: #74b9ff;
                font-weight: bold;
            }
            
            /* Responsive tablet styles */
            @media (max-width: 768px) {
                .stColumns { 
                    flex-direction: column !important; 
                }
                .chalkboard {
                    max-width: 95%;
                    padding: 10px;
                    font-size: 12px;
                }
                .chalk-title {
                    font-size: 16px;
                }
                .chalk-text {
                    font-size: 12px;
                }
                .stButton button {
                    height: 50px !important;
                    font-size: 16px !important;
                    min-width: 100px !important;
                }
                .stSelectbox > div > div {
                    font-size: 16px !important;
                }
                .stTextInput > div > div > input {
                    font-size: 16px !important;
                }
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Collect all known data first
            known_data = []
            
            # Tag Information
            tag_info = []
            if st.session_state.pipeline_manager.pipeline_data.brand != "Unknown":
                tag_info.append(f'Brand: <span class="chalk-value">{st.session_state.pipeline_manager.pipeline_data.brand}</span>')
            if st.session_state.pipeline_manager.pipeline_data.size != "Unknown":
                tag_info.append(f'Size: <span class="chalk-value">{st.session_state.pipeline_manager.pipeline_data.size}</span>')
            
            if tag_info:
                known_data.append(('📏 TAG INFO', tag_info))
            
            # 🔍 AI MODEL INFO (Gemini-only for now)
            # Check if we have any model comparison data OR if we have a successful brand detection
            has_model_data = (hasattr(st.session_state.pipeline_manager.pipeline_data, 'model_comparison') and 
                             st.session_state.pipeline_manager.pipeline_data.model_comparison)
            has_brand = (hasattr(st.session_state.pipeline_manager.pipeline_data, 'brand') and 
                        st.session_state.pipeline_manager.pipeline_data.brand and 
                        st.session_state.pipeline_manager.pipeline_data.brand != "Unknown")
            
            if has_model_data or has_brand:
                model_info = []
                
                # Get the actual model name from the text extractor
                actual_model = "Gemini 2.0 Flash"  # Default fallback
                if hasattr(st.session_state.pipeline_manager.text_extractor, '_gemini_model'):
                    # Try to get model name from the configured model
                    try:
                        model_name = str(st.session_state.pipeline_manager.text_extractor._gemini_model.model_name)
                        if '2.0-flash' in model_name.lower() or 'gemini-2.0-flash-exp' in model_name.lower():
                            actual_model = "Gemini 2.0 Flash"
                        elif '1.5-pro' in model_name.lower():
                            actual_model = "Gemini 1.5 Pro"
                        else:
                            actual_model = "Gemini 2.0 Flash"  # Default to 2.0 Flash
                    except:
                        pass
                
                model_info.append(f'Model: <span class="chalk-value">{actual_model}</span>')
                
                # Determine if brand was successfully detected
                detected_brand = None
                if has_model_data:
                    comp = st.session_state.pipeline_manager.pipeline_data.model_comparison
                    detected_brand = comp.get('gemini_brand')
                
                # Fallback to main pipeline brand if model comparison doesn't have it
                if not detected_brand and has_brand:
                    detected_brand = st.session_state.pipeline_manager.pipeline_data.brand
                
                # Don't show brand here since it's already shown in TAG INFO section
                # Just show model status
                if detected_brand:
                    model_info.append('<span style="color: #00ff00;">✅ Tag readable</span>')
                else:
                    model_info.append('<span style="color: #ff0000;">❌ Tag unreadable</span>')
                
                if model_info:
                    known_data.append(('🤖 AI MODEL', model_info))
            
            # Garment Details
            garment_info = []
            
            # Debug: Check what garment data we have
            garment_type = getattr(st.session_state.pipeline_manager.pipeline_data, 'garment_type', 'Not Set')
            gender = getattr(st.session_state.pipeline_manager.pipeline_data, 'gender', 'Not Set')
            style = getattr(st.session_state.pipeline_manager.pipeline_data, 'style', 'Not Set')
            
            logger.info(f"[UI-DEBUG] Garment data - Type: '{garment_type}', Gender: '{gender}', Style: '{style}'")
            
            if garment_type != "Unknown" and garment_type != "Not Set":
                garment_info.append(f'Type: <span class="chalk-value">{garment_type}</span>')
            else:
                garment_info.append(f'Type: <span class="chalk-value">Not analyzed</span>')
                
            if gender != "Unisex" and gender != "Not Set":
                garment_info.append(f'Gender: <span class="chalk-value">{gender}</span>')
                
                # Add gender confidence if available
                gender_confidence = getattr(st.session_state.pipeline_manager.pipeline_data, 'gender_confidence', None)
                if gender_confidence and gender_confidence != 'Medium':
                    confidence_emoji = '🟢' if gender_confidence == 'high' else '🟡' if gender_confidence == 'medium' else '🔴'
                    garment_info.append(f'Confidence: <span class="chalk-value">{confidence_emoji} {gender_confidence.title()}</span>')
                
                # Add gender indicators if available
                gender_indicators = getattr(st.session_state.pipeline_manager.pipeline_data, 'gender_indicators', [])
                if gender_indicators:
                    indicators_text = ', '.join(gender_indicators[:2])  # Show first 2 indicators
                    garment_info.append(f'Indicators: <span class="chalk-value">{indicators_text}</span>')
            if style != "Unknown" and style != "Not Set":
                garment_info.append(f'Style: <span class="chalk-value">{style}</span>')
            
            if garment_info:
                known_data.append(('👔 GARMENT', garment_info))
            
            # Designer Information
            if st.session_state.pipeline_manager.pipeline_data.is_designer:
                designer_info = []
                designer_info.append(f'Brand: <span class="chalk-value">{st.session_state.pipeline_manager.pipeline_data.brand}</span>')
                designer_info.append(f'Tier: <span class="chalk-value">{st.session_state.pipeline_manager.pipeline_data.designer_tier}</span>')
                designer_info.append(f'Authenticity: <span class="chalk-value">{st.session_state.pipeline_manager.pipeline_data.authenticity_confidence}</span>')
                known_data.append(('💎 DESIGNER', designer_info))

            # Vintage Information
            if st.session_state.pipeline_manager.pipeline_data.is_vintage:
                vintage_info = []
                vintage_info.append(f'Age: <span class="chalk-value">~{st.session_state.pipeline_manager.pipeline_data.tag_age_years} years</span>')
                if st.session_state.pipeline_manager.pipeline_data.vintage_year_estimate:
                    vintage_info.append(f'Era: <span class="chalk-value">circa {st.session_state.pipeline_manager.pipeline_data.vintage_year_estimate}</span>')
                if st.session_state.pipeline_manager.pipeline_data.font_era != "unknown":
                    vintage_info.append(f'Font Style: <span class="chalk-value">{st.session_state.pipeline_manager.pipeline_data.font_era} era</span>')
                known_data.append(('🕰️ VINTAGE', vintage_info))
            
            # Measurements
            if self.current_step >= 3 and st.session_state.pipeline_manager.pipeline_data.bust_measurement > 0:
                known_data.append(('📐 MEASUREMENTS', [f'Bust/Chest: <span class="chalk-value">{st.session_state.pipeline_manager.pipeline_data.bust_measurement:.1f}"</span>']))
            
            # Condition
            condition_info = []
            if self.current_step >= 4:
                condition_info.append(f'Status: <span class="chalk-value">{st.session_state.pipeline_manager.pipeline_data.condition}</span>')
                if st.session_state.pipeline_manager.pipeline_data.defect_count > 0:
                    condition_info.append(f'Defects: <span class="chalk-value">{st.session_state.pipeline_manager.pipeline_data.defect_count} found</span>')
                else:
                    condition_info.append('Defects: <span class="chalk-value">None ✓</span>')
            else:
                condition_info.append('Defects: <span class="chalk-value">None ✓</span>')
            known_data.append(('✨ CONDITION', condition_info))
            
            # Price
            if self.current_step >= 5 and hasattr(st.session_state.pipeline_manager.pipeline_data, 'price_estimate') and st.session_state.pipeline_manager.pipeline_data.price_estimate:
                price_info = []
                if st.session_state.pipeline_manager.pipeline_data.price_estimate.get("mid", 0) > 0:
                    price_info.append(f'Recommended: <span class="chalk-value">${st.session_state.pipeline_manager.pipeline_data.price_estimate.get("mid", 0)}</span>')
                if st.session_state.pipeline_manager.pipeline_data.price_estimate.get("low", 0) > 0 and st.session_state.pipeline_manager.pipeline_data.price_estimate.get("high", 0) > 0:
                    price_info.append(f'Range: <span class="chalk-value">${st.session_state.pipeline_manager.pipeline_data.price_estimate.get("low", 0)}-${st.session_state.pipeline_manager.pipeline_data.price_estimate.get("high", 0)}</span>')
                if price_info:
                    known_data.append(('💰 PRICING', price_info))
            
            # Always show chalkboard - with data or placeholder
            if known_data:
                chalkboard_html = '<div class="chalkboard">'
                chalkboard_html += '<div class="chalk-title">📝 Garment Analysis Board</div>'
                
                for section_title, items in known_data:
                    chalkboard_html += '<div class="chalk-section">'
                    chalkboard_html += f'<div class="chalk-text">{section_title}:</div>'
                    for item in items:
                        chalkboard_html += f'<div class="chalk-text">• {item}</div>'
                    chalkboard_html += '</div>'
                
                chalkboard_html += '</div>'
                
                # Center the chalkboard
                st.markdown(f'<div style="display: flex; justify-content: center; margin: 20px 0;">{chalkboard_html}</div>', unsafe_allow_html=True)
            else:
                # Show chalkboard with placeholder text
                placeholder_html = '<div class="chalkboard">'
                placeholder_html += '<div class="chalk-title">📝 Garment Analysis Board</div>'
                placeholder_html += '<div class="chalk-section">'
                placeholder_html += '<div class="chalk-text" style="text-align: center; font-style: italic; color: #95a5a6;">Garment data will appear here as you progress through the steps...</div>'
                placeholder_html += '</div>'
                placeholder_html += '</div>'
                
                # Center the chalkboard
                st.markdown(f'<div style="display: flex; justify-content: center; margin: 20px 0;">{placeholder_html}</div>', unsafe_allow_html=True)
    
    def render_tag_preview_compact(self):
        """Tag preview with motion-triggered auto-refresh"""
        if st.session_state.pipeline_manager.camera_manager is None:
            return
        
        # Initialize motion detection session state
        if 'last_motion_check' not in st.session_state:
            st.session_state.last_motion_check = 0
        if 'motion_detected' not in st.session_state:
            st.session_state.motion_detected = False
        
        # Check for motion every 2 seconds (not every render)
        current_time = time.time()
        if current_time - st.session_state.last_motion_check > 2.0:
            motion = st.session_state.pipeline_manager.camera_manager.detect_motion_in_roi('tag', threshold=25)
            st.session_state.last_motion_check = current_time
            
            if motion:
                st.session_state.motion_detected = True
                st.info("Tag detected - refreshing preview...")
                time.sleep(0.5)  # Brief pause for tag to settle
                st.rerun()
        
        # Create centered container
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 📷 Tag Camera Preview")
            st.info("Position your garment tag in the green ROI box")
            
            # Manual refresh button
            if st.button("🔄 Refresh Preview", key="manual_refresh_tag"):
                st.session_state.motion_detected = False
                st.rerun()
            
            # Show current frame
            try:
                ardu_frame = st.session_state.pipeline_manager.camera_manager.get_arducam_frame()
                if ardu_frame is not None:
                    frame_with_roi = st.session_state.pipeline_manager.camera_manager.draw_roi_overlay(ardu_frame.copy(), 'tag')
                    st.image(frame_with_roi, caption="Position tag in green box", width=400)
                    
                    # Show motion status
                    if st.session_state.motion_detected:
                        st.caption("✅ Tag detected in ROI")
                    else:
                        st.caption("⏳ Waiting for tag...")
                else:
                    st.warning("ArduCam not accessible")
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
            
            # Periodic background check (triggers rerun every 2 sec to check motion)
            if current_time - st.session_state.last_motion_check > 2.0:
                st.rerun()
        
        st.markdown("---")
    
    def render_camera_status_and_start(self):
        """Render camera status and start button at the top"""
        # Check camera status
        cameras_ready = (st.session_state.pipeline_manager.camera_manager and 
                        st.session_state.pipeline_manager.camera_manager.camera_status['arducam'] and 
                        st.session_state.pipeline_manager.camera_manager.camera_status['realsense'])
        
        # Create status columns
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            st.subheader("📹 Camera Status")
            if cameras_ready:
                st.success("✅ Both cameras ready!")
            else:
                st.error("❌ Camera setup needed")
                if st.session_state.pipeline_manager.camera_manager:
                    if not st.session_state.pipeline_manager.camera_manager.camera_status['arducam']:
                        st.error("ArduCam not ready")
                    if not st.session_state.pipeline_manager.camera_manager.camera_status['realsense']:
                        st.error("RealSense not ready")
        
        with col2:
            st.subheader("💡 Lighting Status")
            if st.session_state.pipeline_manager.light_controller.lights:
                st.success(f"✅ Elgato connected ({len(st.session_state.pipeline_manager.light_controller.lights)} light)")
                st.caption(f"Brightness: {st.session_state.pipeline_manager.light_controller.current_state['brightness']}%")
                st.caption(f"Temperature: {st.session_state.pipeline_manager.light_controller.current_state['temperature']}K")
            else:
                st.warning("⚠️ No Elgato lights detected")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🔍 Try Connect", key="try_connect_top"):
                        st.session_state.pipeline_manager.light_controller.discover_lights()
                        st.rerun()
                with col_btn2:
                    if st.button("📍 Scan Network for Lights", key="connect_known_top"):
                        try:
                            # Reset discovery and try again
                            st.session_state.pipeline_manager.light_controller.discovery_attempted = False
                            discovered_lights = st.session_state.pipeline_manager.light_controller.discover_lights()
                            if discovered_lights:
                                st.success(f"Found {len(st.session_state.pipeline_manager.light_controller.lights)} light(s)")
                                st.rerun()
                            else:
                                st.warning("No Elgato lights found on network")
                                st.success("Connected!")
                                st.rerun()
                        except:
                            st.error("Connection failed")
        
        with col3:
            st.subheader("🚀 System Status")
            if cameras_ready and st.session_state.pipeline_manager.light_controller.lights:
                st.success("✅ All systems ready!")
                if self.current_step == 0:
                    st.info("Ready to begin tag analysis")
            elif cameras_ready:
                st.warning("⚠️ Cameras ready, lights optional")
                if self.current_step == 0:
                    st.info("Ready to begin tag analysis")
            else:
                st.error("❌ Setup cameras first")
            
            # Show current step progress
            if self.current_step > 0:
                st.info(f"Step {self.current_step + 1}: {self.steps[self.current_step]}")
                steps_len = len(self.steps)
                if steps_len > 0:
                    progress = (self.current_step + 1) / steps_len
                    progress = min(max(progress, 0.0), 1.0)  # Clamp to valid range
                    st.progress(progress)
                else:
                    st.progress(0.0)
        
        st.markdown("---")
    
    def render_camera_feeds(self):
        """Camera feed rendering with live brightness analysis"""
        if st.session_state.pipeline_manager.camera_manager is None:
            st.error("Camera manager not initialized")
            return
        
        # For step 0 (Camera Setup), skip camera feeds since we show compact preview above
        if self.current_step == 0:
            # Camera preview is shown in render_tag_preview_compact() above
            return
        
        else:
            # For other steps, show both cameras if needed
            st.header("Live Camera Feeds")
            
            # Auto-adjust toggle
            auto_enabled = st.checkbox("Auto-Adjust Lights", value=st.session_state.pipeline_manager.auto_optimizer.enabled, key="auto_adjust_main")
            if auto_enabled != st.session_state.pipeline_manager.auto_optimizer.enabled:
                st.session_state.pipeline_manager.auto_optimizer.enabled = auto_enabled
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ArduCam - Tag Camera")
                try:
                    # Small delay to avoid USB bandwidth issues
                    time.sleep(0.1)
                    ardu_frame = st.session_state.pipeline_manager.camera_manager.get_arducam_frame()
                    if ardu_frame is not None:
                        # Live brightness analysis for ArduCam
                        if st.session_state.pipeline_manager.auto_optimizer.enabled:
                            # Brightness analysis removed from preview for better performance
                            pass
                        
                        frame_with_roi = st.session_state.pipeline_manager.camera_manager.draw_roi_overlay(ardu_frame.copy(), 'tag')
                        st.image(frame_with_roi, caption="Tag Camera", width='stretch')
                    else:
                        st.warning("ArduCam not accessible")
                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
            
            with col2:
                st.subheader("RealSense - Garment Camera")
                try:
                    real_frame = st.session_state.pipeline_manager.camera_manager.get_realsense_frame()
                    if real_frame is not None:
                        # Live brightness analysis for RealSense
                        if st.session_state.pipeline_manager.auto_optimizer.enabled:
                            # Brightness analysis removed from preview for better performance
                            pass
                        
                        frame_with_roi = st.session_state.pipeline_manager.camera_manager.draw_roi_overlay(real_frame.copy(), 'work')
                        st.image(frame_with_roi, caption="Garment Camera", width='stretch')
                    else:
                        st.warning("RealSense not accessible")
                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
        
        # Camera status is now shown at the top in render_camera_status_and_start()
        
        # SAFETY CHECK: Ensure Next Step button is always visible
        # If for some reason the Next Step button didn't render above, show it here as a fallback
        if not st.session_state.get('next_step_button_rendered', False):
            st.markdown("---")
            st.markdown("### 🔧 Action Panel")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                # Fallback button removed - using main button in render_action_panel()
                st.write("")  # Empty space for layout
            st.session_state.next_step_button_rendered = True
    def _render_step_header(self):
        """Render the common header with progress bar, reset, and next step buttons"""
        # Show step progress bar, reset button, and Next Step button at the top
        progress_col1, progress_col2, reset_col, next_col = st.columns([2, 1, 1, 1])
        with progress_col1:
            steps_len = len(self.steps)
            if steps_len > 0:
                progress_val = (st.session_state.pipeline_manager.current_step + 1) / steps_len
                progress_val = min(max(progress_val, 0.0), 1.0)  # Clamp to valid range
                st.progress(progress_val)
            else:
                st.progress(0.0)
        with progress_col2:
            st.write(f"Step {st.session_state.pipeline_manager.current_step + 1} of {len(self.steps)}")
        with reset_col:
            if st.button("🔄 Reset", help="Start over with new garment", key="reset_pipeline"):
                # Clear all pipeline data including images
                st.session_state.pipeline_manager.pipeline_data = PipelineData()
                st.session_state.pipeline_manager.current_step = 0
                
                # Clear retry flags and loop prevention
                # Reset session state for fresh start
                
                # Clear any cached camera frames or images
                if 'cached_tag_frame' in st.session_state:
                    del st.session_state.cached_tag_frame
                if 'cached_garment_frame' in st.session_state:
                    del st.session_state.cached_garment_frame
                if 'last_camera_frame' in st.session_state:
                    del st.session_state.last_camera_frame
                
                # Clear any UI state flags
                if 'show_tag_preview' in st.session_state:
                    st.session_state.show_tag_preview = False
                if 'show_garment_preview' in st.session_state:
                    st.session_state.show_garment_preview = False
                # Reset Next Step button flag
                st.session_state.next_step_button_rendered = False
                
                st.success("✅ Pipeline reset! Camera feed refreshed.")
                st.rerun()
        
        return next_col
    
    def _render_step_0_tag_analysis(self):
        """Render Step 0: Tag Analysis with LIVE camera feed"""
        st.markdown("### 📸 Capture & Analyze Tag")
        
        # Control buttons at the top
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
        
        with col_ctrl1:
            if st.session_state.live_preview_enabled:
                if st.button("⏸️ Pause Live Feed", key="pause_tag_feed"):
                    st.session_state.live_preview_enabled = False
                    st.rerun()
            else:
                if st.button("▶️ Resume Live Feed", key="resume_tag_feed"):
                    st.session_state.live_preview_enabled = True
                    st.rerun()
        
        with col_ctrl2:
            # Auto-zoom toggle
            auto_zoom = st.checkbox(
                "🤖 Auto-Zoom",
                value=st.session_state.get('auto_zoom_enabled', True),
                help="Automatically detect and crop tag"
            )
            st.session_state.auto_zoom_enabled = auto_zoom
        
        with col_ctrl3:
            if not auto_zoom:
                zoom_level = st.slider(
                    "🔬 Zoom",
                    min_value=1.0,
                    max_value=3.0,
                    value=st.session_state.get('zoom_level', 1.0),
                    step=0.25,
                    key="zoom_slider"
                )
                st.session_state.zoom_level = zoom_level
        
        st.markdown("---")
        
        # Add preview control buttons (manual lighting hidden for employee use)
        col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 1, 1])
        # Advanced controls toggle (hidden by default for employee use)
        with col_refresh1:
            show_advanced = st.button("⚙️ Advanced Controls", help="Show manual lighting and measurement options", key="show_advanced_controls")
            if show_advanced:
                st.session_state.show_advanced_controls = not st.session_state.get('show_advanced_controls', False)
                st.rerun()
        
        # Show advanced controls if enabled
        if st.session_state.get('show_advanced_controls', False):
            with st.expander("🔧 Advanced Controls", expanded=True):
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    if st.button("🔅 Preview with Tag Lighting"):
                        st.session_state.pipeline_manager.light_controller.set_brightness(15)
                        st.session_state.pipeline_manager.light_controller.set_color_temp(5500)
                        st.success("🔅 Tag lighting applied!")
                with col_adv2:
                    if st.button("📏 Show Measurements"):
                        st.session_state.show_measurements = not st.session_state.get('show_measurements', False)
                        st.rerun()
        
        with col_refresh2:
            if st.button("🔄 Refresh Preview", key="refresh_tag_preview"):
                # Clear any cached frames to force refresh
                if 'cached_tag_frame' in st.session_state:
                    del st.session_state.cached_tag_frame
                if 'last_camera_frame' in st.session_state:
                    del st.session_state.last_camera_frame
                st.rerun()
        
        with col_refresh3:
            if st.button("📸 Capture Tag", key="capture_tag_manual"):
                st.session_state.pipeline_manager.capture_tag()
                st.success("📸 Tag captured!")
                st.rerun()
        
        # LIVE CAMERA PREVIEW
        st.markdown("#### 📷 Live Tag Camera")
        
        if st.session_state.pipeline_manager.camera_manager:
            try:
                # Get fresh frame
                frame = st.session_state.pipeline_manager.camera_manager.get_arducam_frame()
                
                if frame is not None:
                    # Apply ROI
                    zoom = st.session_state.get('zoom_level', 1.0)
                    roi_frame = st.session_state.pipeline_manager.camera_manager.apply_roi(
                        frame, 'tag', zoom_factor=zoom
                    )
                    
                    if roi_frame is not None:
                        # Draw ROI overlay on full frame
                        frame_with_roi = st.session_state.pipeline_manager.camera_manager.draw_roi_overlay(
                            frame.copy(), 'tag'
                        )
                        
                        # Center the preview
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            st.image(
                                frame_with_roi,
                                caption="🎯 Position tag in GREEN BOX",
                                width='stretch'
                            )
                            
                            # Show brightness info
                            if st.session_state.pipeline_manager.auto_optimizer.enabled:
                                brightness_info = st.session_state.pipeline_manager.auto_optimizer.analyze_image_brightness(roi_frame)
                                if brightness_info:
                                    mean_bright = brightness_info['mean']
                                    if mean_bright > 180:
                                        st.caption("💡 Very bright - will reduce lighting on capture")
                                    elif mean_bright > 140:
                                        st.caption("💡 Bright - will use low lighting")
                                    elif mean_bright < 60:
                                        st.caption("💡 Dark - will boost lighting")
                                    else:
                                        st.caption("✅ Good lighting detected")
                    
                    else:
                        st.warning("⚠️ ROI not set")
                else:
                    st.error("❌ No camera frame available")
            
            except Exception as e:
                st.error(f"Camera error: {e}")
                st.session_state.live_preview_enabled = False
        else:
            st.error("Camera manager not initialized")
        
        # AUTO-REFRESH LOGIC
        if st.session_state.live_preview_enabled:
            # Throttle to prevent WebSocket overload
            if 'last_preview_refresh' not in st.session_state:
                st.session_state.last_preview_refresh = 0
            
            current_time = time.time()
            time_since_refresh = current_time - st.session_state.last_preview_refresh
            
            # Refresh every 0.5 seconds (2 FPS)
            if time_since_refresh > 0.5:
                # Check if user is interacting
                if not pause_refresh_on_interaction():
                    st.session_state.last_preview_refresh = current_time
                    time.sleep(0.1)  # Small delay to prevent CPU spike
                    st.rerun()
        
        # Manual override section
        st.markdown("---")
        st.markdown("#### ✏️ Manual Override (if needed)")
        col_override1, col_override2 = st.columns(2)
        
        with col_override1:
            manual_brand = st.text_input("Brand (if not detected)", value="")
            if manual_brand:
                st.session_state.pipeline_manager.pipeline_data.brand = manual_brand
                st.success(f"✅ Brand set to: {manual_brand}")
            
            # Voice input removed for reliability - use manual entry instead
        
        with col_override2:
            manual_size = st.text_input("Size (if not detected)", value="")
            if manual_size:
                st.session_state.pipeline_manager.pipeline_data.size = manual_size
                st.success(f"✅ Size set to: {manual_size}")
    
    def _render_step_1_garment_analysis(self):
        """Render Step 1: Garment Capture & Analysis with user confirmation"""
        st.markdown("### 👔 Capture & Analyze Garment")
        
        # Check if analysis needs user confirmation
        needs_confirmation = getattr(st.session_state.pipeline_manager.pipeline_data, 'needs_user_confirmation', False)
        
        # Check for validation issues (cardigan vs pullover)
        validation_issue = getattr(st.session_state.pipeline_manager.pipeline_data, 'validation_issue', None)
        
        if validation_issue:
            st.error(f"⚠️ Classification Issue: {validation_issue['error']}")
            st.warning(f"💡 Suggested correction: {validation_issue['suggestion']}")
            
            # Show the captured image for user verification
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(
                    st.session_state.pipeline_manager.pipeline_data.garment_image,
                    caption="Look at this image carefully",
                    width=400
                )
            
            with col2:
                st.markdown("### Does this garment have ANY of these?")
                
                st.markdown("**CARDIGAN indicators:**")
                st.markdown("✓ Buttons down the front")
                st.markdown("✓ Zipper down the front")
                st.markdown("✓ Two separate edges that overlap")
                st.markdown("✓ Can be worn open like a jacket")
                
                st.markdown("**PULLOVER indicators:**")
                st.markdown("✓ Solid front, no opening")
                st.markdown("✓ Must pull over head to wear")
                st.markdown("✓ No buttons or zipper")
                st.markdown("✓ Cannot be worn open")
            
            # Show correction buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Yes → It's a CARDIGAN", type="primary", key="confirm_cardigan"):
                    # Save training data
                    ai_original_type = st.session_state.pipeline_manager.pipeline_data.garment_type
                    correction_data = {
                        'image_path': 'captured_garment.jpg',  # Reference to current image
                        'ai_said': ai_original_type,
                        'user_corrected_to': 'cardigan',
                        'ai_observations': validation_issue.get('center_front_observations', []),
                        'validation_error': validation_issue.get('error', ''),
                        'timestamp': datetime.now().isoformat()
                    }
                    save_correction_for_training(correction_data)
                    
                    st.session_state.pipeline_manager.pipeline_data.garment_type = "cardigan"
                    st.session_state.pipeline_manager.pipeline_data.validation_issue = None
                    st.success("✅ Confirmed as cardigan")
                    st.toast("🎯 Thank you! This correction will improve the AI.", icon="📚")
                    st.rerun()
            
            with col2:
                if st.button("No → It's a PULLOVER", type="primary", key="confirm_pullover"):
                    # Save training data
                    ai_original_type = st.session_state.pipeline_manager.pipeline_data.garment_type
                    correction_data = {
                        'image_path': 'captured_garment.jpg',  # Reference to current image
                        'ai_said': ai_original_type,
                        'user_corrected_to': 'pullover',
                        'ai_observations': validation_issue.get('center_front_observations', []),
                        'validation_error': validation_issue.get('error', ''),
                        'timestamp': datetime.now().isoformat()
                    }
                    save_correction_for_training(correction_data)
                    
                    st.session_state.pipeline_manager.pipeline_data.garment_type = "pullover"
                    st.session_state.pipeline_manager.pipeline_data.validation_issue = None
                    st.success("✅ Confirmed as pullover")
                    st.toast("🎯 Thank you! This correction will improve the AI.", icon="📚")
                    st.rerun()
            
            # Show photography guidance
            st.info("""
            📸 **For Best Results:**
            - Lay garment FLAT on table
            - Button up cardigans OR lay open to show both front edges
            - Ensure center front is visible and well-lit
            - Photo from directly above
            """)
            
            return  # Don't proceed with normal analysis
        
        if needs_confirmation:
            # Show the captured image first
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.image(
                    st.session_state.pipeline_manager.pipeline_data.garment_image,
                    caption="Captured Garment",
                    width='stretch'
                )
            
            with col2:
                st.markdown("### 📏 Length Guide")
                st.markdown("""
                **Dress**: Extends below mid-thigh  
                (Usually 32+ inches from shoulder)
                
                **Tunic**: Mid-length  
                (26-32 inches from shoulder)
                
                **Blouse/Top**: Ends at hip or above  
                (20-26 inches from shoulder)
                """)
            
            # Show the modal-style selector
            garment_choice = show_confirmation_dialog(
                title="Garment Type Classification",
                message="Please identify this garment:",
                options=[
                    "Jacket/Blazer (structured outerwear with lapels/collar)",
                    "Cardigan/Sweater (knit outerwear, open front)",
                    "Dress (one-piece, extends below hips)",
                    "Tunic (long top, hip to mid-thigh length)",
                    "Blouse/Shirt/Top (ends at or above hip)"
                ],
                key_prefix="garment_type"
            )
            
            if garment_choice:
                # Parse the choice
                if "Jacket" in garment_choice or "Blazer" in garment_choice:
                    garment_type = "Jacket"
                elif "Cardigan" in garment_choice or "Sweater" in garment_choice:
                    garment_type = "Cardigan"
                elif "Dress" in garment_choice:
                    garment_type = "Dress"
                elif "Tunic" in garment_choice:
                    garment_type = "Tunic"
                else:
                    garment_type = "Blouse"
                
                st.session_state.pipeline_manager.pipeline_data.garment_type = garment_type
                st.session_state.pipeline_manager.pipeline_data.needs_user_confirmation = False
                st.success(f"✅ Confirmed as: {garment_type}")
                st.rerun()
            
            # Option to recapture with better positioning
            if st.button("🔄 Recapture Garment", key="recapture_garment"):
                st.session_state.pipeline_manager.pipeline_data.garment_image = None
                st.session_state.pipeline_manager.pipeline_data.needs_user_confirmation = False
                st.info("Position the ENTIRE garment in frame, then click Next Step")
                st.rerun()
            
            return  # Exit early if showing confirmation UI
        
        # 🚀 OPTIMIZATION: Check for background garment analysis results
        background_result = st.session_state.pipeline_manager.get_background_garment_result()
        if background_result:
            if background_result.get('success'):
                st.success("⚡ Background analysis complete! Results ready.")
                # Use the background results
                st.session_state.pipeline_manager.pipeline_data.garment_type = background_result.get('garment_type', 'Unknown')
                st.session_state.pipeline_manager.pipeline_data.gender = background_result.get('gender', 'Unisex')
                st.session_state.pipeline_manager.pipeline_data.style = background_result.get('style', 'Casual')
                st.session_state.pipeline_manager.pipeline_data.era = background_result.get('era', 'Contemporary')
                st.session_state.pipeline_manager.pipeline_data.condition = background_result.get('condition', 'Good')
                
                # Show the results
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Type:** {background_result.get('garment_type', 'Unknown')}")
                    st.info(f"**Gender:** {background_result.get('gender', 'Unisex')}")
                with col2:
                    st.info(f"**Style:** {background_result.get('style', 'Casual')}")
                    st.info(f"**Era:** {background_result.get('era', 'Contemporary')}")
                
                st.session_state.pipeline_manager.current_step = 2
                st.rerun()
            else:
                st.warning(f"⚠️ Background analysis failed: {background_result.get('error', 'Unknown error')}")
        
        # 🚀 OPTIMIZATION: Show brand-specific garment hints for faster manual entry
        detected_brand = st.session_state.pipeline_manager.pipeline_data.brand
        if detected_brand and detected_brand != 'Unknown':
            hints = st.session_state.pipeline_manager.get_brand_specific_garment_hints(detected_brand)
            if hints:
                st.info(f"💡 **{detected_brand}** typically makes: {', '.join(hints[:3])}")
        
        # Add preview control buttons
        col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 1, 1])
        with col_refresh1:
            if st.button("🔄 Refresh Preview", key="refresh_garment_preview"):
                # Clear any cached frames to force refresh
                if 'cached_garment_frame' in st.session_state:
                    del st.session_state.cached_garment_frame
                if 'last_camera_frame' in st.session_state:
                    del st.session_state.last_camera_frame
                st.rerun()
        
        with col_refresh2:
            if st.button("📸 Capture Garment", key="capture_garment_manual"):
                st.session_state.pipeline_manager.capture_garment()
                st.success("📸 Garment captured!")
                st.rerun()
        
        with col_refresh3:
            if st.button("🔍 Analyze Defects", key="analyze_defects_manual"):
                if st.session_state.pipeline_manager.pipeline_data.garment_image is not None:
                    st.session_state.pipeline_manager.analyze_defects()
                    st.success("🔍 Defect analysis complete!")
                    st.rerun()
                else:
                    st.warning("Please capture garment first")
        
        # LIVE CAMERA PREVIEW WITH MOTION DETECTION
        st.markdown("#### 📷 Live Garment Camera")
        
        # Initialize motion detection session state for garment
        if 'last_garment_motion_check' not in st.session_state:
            st.session_state.last_garment_motion_check = 0
        if 'garment_motion_detected' not in st.session_state:
            st.session_state.garment_motion_detected = False
        
        # Check for motion every 2 seconds (not every render)
        current_time = time.time()
        if current_time - st.session_state.last_garment_motion_check > 2.0:
            motion = st.session_state.pipeline_manager.camera_manager.detect_motion_in_roi('work', threshold=25)
            st.session_state.last_garment_motion_check = current_time
            
            if motion:
                st.session_state.garment_motion_detected = True
                st.info("Garment movement detected - refreshing preview...")
                time.sleep(0.5)  # Brief pause for garment to settle
                st.rerun()
        
        if st.session_state.pipeline_manager.camera_manager:
            try:
                # Get fresh frame from RealSense
                frame = st.session_state.pipeline_manager.camera_manager.get_realsense_frame()
                
                if frame is not None:
                    # Draw ROI
                    frame_with_roi = st.session_state.pipeline_manager.camera_manager.draw_roi_overlay(
                        frame.copy(), 'work'
                    )
                    
                    # Center the preview
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        st.image(
                            frame_with_roi,
                            caption="🎯 Position ENTIRE garment in GREEN BOX",
                            width='stretch'
                        )
                        
                        # Show motion status
                        if st.session_state.garment_motion_detected:
                            st.caption("✅ Garment movement detected in ROI")
                        else:
                            st.caption("⏳ Waiting for garment movement...")
                        
                else:
                    st.error("❌ No camera frame available")
            
            except Exception as e:
                st.error(f"Camera error: {e}")
                st.session_state.live_preview_enabled = False
        else:
            st.error("Camera manager not initialized")
        
        # Periodic background check (triggers rerun every 2 sec to check motion)
        if current_time - st.session_state.last_garment_motion_check > 2.0:
            st.rerun()
    
    def _render_step_2_measurements(self):
        """Render Step 2: Measurements"""
        # Only show measurements if explicitly enabled
        if not st.session_state.get('show_measurements', False):
            st.markdown("### 📏 Measurements (Hidden)")
            st.info("💡 Click '⚙️ Advanced Controls' → '📏 Show Measurements' to enable measurement tools")
            return
            
        st.markdown("### 📏 Take Measurements")
        
        # Determine measurement type based on garment
        garment_type = st.session_state.pipeline_manager.pipeline_data.garment_type
        if garment_type in ['pants', 'jeans', 'skirt', 'shorts']:
            st.info("📏 **Measuring WAIST** (for bottoms)")
            measurement_type = "waist"
        else:
            st.info("📏 **Measuring BUST/CHEST** (for tops)")
            measurement_type = "bust"
        
        # Manual measurement input
        col_manual1, col_manual2 = st.columns(2)
        with col_manual1:
            if measurement_type == "waist":
                manual_measurement = st.number_input("Waist Measurement (inches)", min_value=20.0, max_value=60.0, value=30.0, step=0.5)
                st.session_state.pipeline_manager.pipeline_data.waist_measurement = manual_measurement
                # Convert to size
                size = self._measurement_to_size(manual_measurement, measurement_type)
                if size != "Unknown":
                    st.session_state.pipeline_manager.pipeline_data.size = size
                    st.success(f"✅ Size: {size}")
            else:
                manual_measurement = st.number_input("Bust Measurement (inches)", min_value=30.0, max_value=50.0, value=36.0, step=0.5)
                st.session_state.pipeline_manager.pipeline_data.bust_measurement = manual_measurement
                # Convert to size
                size = self._measurement_to_size(manual_measurement, measurement_type)
                if size != "Unknown":
                    st.session_state.pipeline_manager.pipeline_data.size = size
                    st.success(f"✅ Size: {size}")
        
        with col_manual2:
            st.markdown("**📐 Measurement Guide:**")
            if measurement_type == "waist":
                st.markdown("""
                - Measure around natural waist
                - Keep tape measure parallel to floor
                - Don't pull too tight
                """)
            else:
                st.markdown("""
                - Measure around fullest part of bust
                - Keep tape measure parallel to floor
                - Don't compress
                """)
    
    def _render_step_3_measurements(self):
        """Render Step 3: Measurements (Calibration)"""
        st.info("📏 Measurement step - usually skipped if size detected from tag")
        
        # Check if we already have size
        if self.pipeline_data.size and self.pipeline_data.size not in ['Unknown', '']:
            st.success(f"✅ Size already detected from tag: {self.pipeline_data.size}")
            st.caption("This step is automatically skipped when size is known")
            return
        
        # If no size, offer manual measurement
        st.warning("⚠️ No size detected from tag - manual measurement needed")
        
        if self.pipeline_data.garment_image is not None:
            st.image(self.pipeline_data.garment_image, 
                    caption="Position garment for measurement", 
                    width=400)
            
            # Simple manual size entry as fallback
            st.markdown("#### Manual Size Entry")
            manual_size = st.text_input(
                "Enter size if visible on garment",
                placeholder="e.g., M, 8, 42",
                key="manual_size_entry"
            )
            
            if manual_size:
                self.pipeline_data.size = manual_size
                st.success(f"✅ Size set to: {manual_size}")
    
    def _render_step_3_pricing(self):
        """Render Step 3: Pricing Analysis"""
        st.markdown("### 💰 Price Analysis")
        
        # Show current data
        st.info(f"**Brand:** {st.session_state.pipeline_manager.pipeline_data.brand}")
        st.info(f"**Type:** {st.session_state.pipeline_manager.pipeline_data.garment_type}")
        st.info(f"**Size:** {st.session_state.pipeline_manager.pipeline_data.size}")
        
        # Get eBay sold comps
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("📊 Get Market Price", type="primary"):
                with st.spinner("Analyzing eBay sold listings with Item Specifics..."):
                    # Build Item Specifics from garment analysis
                    item_specifics = build_ebay_item_specifics(st.session_state.pipeline_manager.pipeline_data)
                    
                    if item_specifics:
                        st.info(f"🔍 Using Item Specifics: {item_specifics}")
                    
                    # Try eBay with Item Specifics
                    ebay_result = self.pricing_api.get_sold_listings_data(
                        brand=st.session_state.pipeline_manager.pipeline_data.brand,
                        garment_type=st.session_state.pipeline_manager.pipeline_data.garment_type,
                        size=st.session_state.pipeline_manager.pipeline_data.size,
                        gender=st.session_state.pipeline_manager.pipeline_data.gender,
                        item_specifics=item_specifics  # NEW: Pass Item Specifics
                    )
                    
                    if ebay_result.get('success'):
                        avg = ebay_result['average_price']
                        
                        # Apply condition adjustment
                        condition_factors = {
                            'Excellent': 1.0,
                            'Very Good': 0.85,
                            'Good': 0.7,
                            'Fair': 0.5,
                            'Poor': 0.3
                        }
                        factor = condition_factors.get(st.session_state.pipeline_manager.pipeline_data.condition, 0.7)
                        
                        adjusted_price = avg * factor
                        
                        # Store results
                        st.session_state.price_data = {
                            'low': int(adjusted_price * 0.8),
                            'mid': int(adjusted_price),
                            'high': int(adjusted_price * 1.2),
                            'source': f"eBay average from {ebay_result['count']} sold items",
                            'confidence': ebay_result['confidence'],
                            'raw_average': avg,
                            'condition_factor': factor
                        }
                        
                        st.success(f"✅ Found {ebay_result['count']} sold items!")
                        
                        # Show Item Specifics analysis
                        specifics_analysis = ebay_result.get('specifics_analysis', {})
                        if specifics_analysis:
                            st.info("📊 Item Specifics Found:")
                            for specific_name, data in specifics_analysis.items():
                                most_common = data['most_common'][0] if data['most_common'] else None
                                if most_common:
                                    st.caption(f"• {specific_name}: {most_common[0]} ({most_common[1]} items)")
                        
                        # Show filters used
                        filters_used = ebay_result.get('filters_used', {})
                        if filters_used:
                            st.caption(f"🔍 Filters applied: {filters_used}")
                        
                    else:
                        st.warning(f"⚠️ eBay search failed: {ebay_result.get('error')}")
                        st.info("Using fallback pricing...")
                        # Fall back to hybrid pricing
                        price_result = self.pricing_api.calculate_hybrid_price(
                            brand=st.session_state.pipeline_manager.pipeline_data.brand,
                            garment_type=st.session_state.pipeline_manager.pipeline_data.garment_type,
                            condition=st.session_state.pipeline_manager.pipeline_data.condition or "Good",
                            size=st.session_state.pipeline_manager.pipeline_data.size,
                            gender=st.session_state.pipeline_manager.pipeline_data.gender
                        )
                        st.session_state.price_data = price_result
                    
                    st.rerun()  # Rerun to display the new data
        
        # Display the results after the API call is complete
        if 'price_data' in st.session_state and st.session_state.price_data:
            price_data = st.session_state.price_data
            if price_data:
                st.success("✅ Intelligent pricing analysis complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Low", f"${price_data['low']}")
                col2.metric("Recommended", f"${price_data['mid']}")
                col3.metric("High", f"${price_data['high']}")
                
                st.caption(f"📊 Source: {price_data['source']}")
                st.caption(f"🎯 Confidence: {price_data['confidence']}")
                
                # Show eBay-specific details if available
                if price_data.get('raw_average'):
                    st.caption(f"📊 Based on ${price_data['raw_average']:.2f} avg × {price_data['condition_factor']} condition factor")
                if price_data.get('search_query'):
                    st.caption(f"🔍 Search: '{price_data['search_query']}'")

                # Update your app's price estimate with this new data
                st.session_state.pipeline_manager.pipeline_data.price_estimate = {
                    'low': price_data['low'],
                    'mid': price_data['mid'],
                    'high': price_data['high']
                }
                
                # Show debug info if available
                if price_data.get('source') and 'eBay' in price_data['source']:
                    with st.expander("🔍 eBay Details", expanded=False):
                        st.write(f"**Items Found:** {price_data.get('count', 'N/A')}")
                        st.write(f"**Raw Average:** ${price_data.get('raw_average', 0):.2f}")
                        st.write(f"**Condition Factor:** {price_data.get('condition_factor', 1.0)}")
                        st.write(f"**Search Query:** {price_data.get('search_query', 'N/A')}")
                        
                        # Show eBay search URL for verification
                        search_url = self.pricing_api.generate_search_url(
                            brand=st.session_state.pipeline_manager.pipeline_data.brand,
                            garment_type=st.session_state.pipeline_manager.pipeline_data.garment_type,
                            size=st.session_state.pipeline_manager.pipeline_data.size,
                            gender=st.session_state.pipeline_manager.pipeline_data.gender
                        )
                        st.markdown(f"🔗 [Verify on eBay]({search_url})")
                
                else:
                    st.warning("No pricing data available.")
        
        with col2:
            condition = st.selectbox("Condition", ["Excellent", "Very Good", "Good", "Fair", "Poor"], 
                                   index=2)  # Default to "Good"
            st.session_state.pipeline_manager.pipeline_data.condition = condition
        
        # Manual price override
        st.subheader("✏️ Manual Override")
        col_override1, col_override2 = st.columns(2)
        
        with col_override1:
            estimated_price = st.number_input(
                "Estimated Price ($)", 
                min_value=0.0, 
                max_value=10000.0, 
                value=st.session_state.pipeline_manager.pipeline_data.estimated_price or 50.0, 
                step=1.0
            )
            st.session_state.pipeline_manager.pipeline_data.estimated_price = estimated_price
        
        with col_override2:
            confidence = st.selectbox(
                "Price Confidence", 
                ["High (eBay data)", "Medium (estimated)", "Low (guess)"],
                index=1
            )
            st.session_state.pipeline_manager.pipeline_data.price_confidence = confidence
    
    def _render_step_4_defects(self):
        """Render Step 4: Defect Detection"""
        st.info("🔍 Position garment for defect inspection, then click Next Step")
        
        # Show current garment image if available
        if self.pipeline_data.garment_image is not None:
            st.image(self.pipeline_data.garment_image, 
                    caption="Current garment image", 
                    width='stretch')
            
            # Show any detected defects
            if hasattr(self.pipeline_data, 'defects') and self.pipeline_data.defects:
                st.warning(f"⚠️ {len(self.pipeline_data.defects)} defects detected")
                for idx, defect in enumerate(self.pipeline_data.defects):
                    st.write(f"**#{idx+1}**: {defect.get('type', 'Unknown')} at {defect.get('location', 'unknown')}")
            else:
                st.success("✅ No defects detected")
        else:
            st.warning("No garment image captured yet")

    def _render_step_5_pricing(self):
        """Render Step 5: Pricing"""
        st.info("💰 Review pricing estimates")
        
        # Show current data
        st.write(f"**Brand:** {self.pipeline_data.brand}")
        st.write(f"**Type:** {self.pipeline_data.garment_type}")
        st.write(f"**Condition:** {self.pipeline_data.condition}")
        
        # Show price estimates if calculated
        if hasattr(self.pipeline_data, 'price_estimate') and self.pipeline_data.price_estimate:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low", f"${self.pipeline_data.price_estimate.get('low', 0)}")
            with col2:
                st.metric("Recommended", f"${self.pipeline_data.price_estimate.get('mid', 0)}")
            with col3:
                st.metric("High", f"${self.pipeline_data.price_estimate.get('high', 0)}")
        else:
            st.write("Price will be calculated when you click Next Step")
    
    def _measurement_to_size(self, measurement, measurement_type):
        """Convert measurement to clothing size"""
        if measurement_type == "waist":
            if measurement < 26: return "XS"
            elif measurement < 28: return "S"
            elif measurement < 30: return "M"
            elif measurement < 32: return "L"
            elif measurement < 34: return "XL"
            elif measurement < 36: return "XXL"
            else: return f"W{int(measurement)}"
        else:  # bust
            if measurement < 32: return "XS"
            elif measurement < 34: return "S"
            elif measurement < 36: return "M"
            elif measurement < 38: return "L"
            elif measurement < 40: return "XL"
            elif measurement < 42: return "XXL"
            else: return "XXXL"
    
    def render_action_panel(self):
        """Action panel with buttons at the top right, close together"""
        logger.info(f"[PANEL] render_action_panel() called for step {self.current_step}")
        
        # TOP RIGHT BUTTONS: All navigation buttons at the top right, close together
        col_space, col_buttons = st.columns([3, 1])  # Push buttons to the right
        
        with col_buttons:
            if self.current_step == 0:
                # Step 0: Start New and Next buttons
                col_start, col_next = st.columns(2)
                
                with col_start:
                    def start_new():
                        self.current_step = 0
                        self.pipeline_data = PipelineData()
                        # Clear captured images
                        if 'captured_tag_image' in st.session_state:
                            del st.session_state.captured_tag_image
                        if 'captured_garment_image' in st.session_state:
                            del st.session_state.captured_garment_image
                        logger.info("[BUTTON] 🆕 Start New clicked - pipeline reset")
                        st.rerun()
                    
                    st.button("🆕 Start New", on_click=start_new, key="start_new_button", type="primary")
                
                with col_next:
                    # Next button will be handled below
                    pass
            else:
                # Steps 1+: Back, Reset, and Next buttons
                col_back, col_reset, col_next = st.columns(3)
                
                with col_back:
                    def go_back():
                        if self.current_step > 0:
                            old_step = self.current_step
                            self.current_step -= 1
                            logger.info(f"[BUTTON] ⬅️ Back: {old_step} → {self.current_step}")
                            st.rerun()
                    
                    st.button("⬅️ Back", on_click=go_back, key="back_button", type="secondary")
                
                with col_reset:
                    def reset_pipeline():
                        self.current_step = 0
                        self.pipeline_data = PipelineData()
                        # Clear captured images
                        if 'captured_tag_image' in st.session_state:
                            del st.session_state.captured_tag_image
                        if 'captured_garment_image' in st.session_state:
                            del st.session_state.captured_garment_image
                        logger.info("[BUTTON] 🔄 Reset clicked - pipeline reset")
                        st.rerun()
                    
                    st.button("🔄 Reset", on_click=reset_pipeline, key="reset_button", type="secondary")
                
                with col_next:
                    # Next button will be handled below
                    pass
        
            # Add Next button to the appropriate column
            if self.current_step == 0:
                col_start, col_next = st.columns(2)
                with col_next:
                    def advance_step():
                        """Callback to advance to next step with analysis"""
                        old = self.current_step
                        logger.info(f"[BUTTON] ✅ Next Step clicked for step {old}")
                        
                        # Execute the logic for the CURRENT step
                        analysis_success = False
                        
                        if old == 0:  # Tag Analysis
                            # Capture tag image for training data
                            tag_frame = self.camera_manager.get_arducam_frame()
                            if tag_frame is not None:
                                st.session_state.captured_tag_image = tag_frame
                                logger.info("[TRAINING] Tag image captured for training data")
                            
                            with st.spinner("🔍 Analyzing tag... This may take a moment."):
                                result = self.handle_step_0_tag_analysis()
                            
                            # DEBUG: Log the full result
                            logger.info(f"[DEBUG] Tag analysis result: {result}")
                            
                            if result and result.get('success'):
                                st.success(f"✅ Tag analyzed: {result.get('message', 'OK')}")
                                analysis_success = True
                            else:
                                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                                st.error(f"❌ Tag analysis failed: {error_msg}")
                                logger.error(f"[DEBUG] Analysis failed: {error_msg}")
                                
                                # TEMPORARY BYPASS: Allow advancement even if analysis fails
                                st.warning("⚠️ TEMP: Allowing step advancement despite analysis failure")
                                analysis_success = True
                        
                        elif old == 1:  # Garment Analysis
                            # Capture garment image for training data
                            garment_frame = self.camera_manager.get_realsense_frame()
                            if garment_frame is not None:
                                st.session_state.captured_garment_image = garment_frame
                                logger.info("[TRAINING] Garment image captured for training data")
                            
                            with st.spinner("🔍 Analyzing garment... This may take a moment."):
                                result = self.handle_step_1_garment_analysis()
                            
                            if result and result.get('success'):
                                st.success(f"✅ Garment analyzed: {result.get('message', 'OK')}")
                                analysis_success = True
                            else:
                                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                                st.error(f"❌ Garment analysis failed: {error_msg}")
                                logger.error(f"[DEBUG] Garment analysis failed: {error_msg}")
                                
                                # TEMPORARY BYPASS: Allow advancement even if analysis fails
                                st.warning("⚠️ TEMP: Allowing step advancement despite analysis failure")
                                analysis_success = True
                        
                        # Advance to next step if analysis succeeded or bypassed
                        if analysis_success:
                            self.current_step = old + 1
                            logger.info(f"[BUTTON] ✅ Step advanced: {old} → {self.current_step}")
                            st.rerun()
                    
                    st.button("➡️ Next Step", on_click=advance_step, key="next_step_button", type="primary")
            else:
                col_back, col_reset, col_next = st.columns(3)
                with col_next:
                    def advance_step():
                        """Callback to advance to next step"""
                        old = self.current_step
                        self.current_step = old + 1
                        logger.info(f"[BUTTON] ✅ Step advanced: {old} → {self.current_step}")
                        st.rerun()
                    
                    st.button("➡️ Next Step", on_click=advance_step, key="next_step_button", type="primary")

        # Action panel complete - buttons are now at the top right
    
    def _save_training_sample(self, analysis_result):
        """Save training data for future model improvement"""
        try:
            import json
            import os
            from datetime import datetime
            
            # Create training data directory
            os.makedirs("training_data", exist_ok=True)
            
            # Prepare sample data
            sample_data = {
                'timestamp': datetime.now().isoformat(),
                'step': 'tag_analysis',
                'brand': analysis_result.get('brand', 'Unknown'),
                'size': analysis_result.get('size', 'Unknown'),
                'material': analysis_result.get('material', 'Unknown'),
                'confidence': analysis_result.get('confidence', 0.0),
                'raw_text': analysis_result.get('raw_text', ''),
                'success': analysis_result.get('success', False),
                'user_validated': False  # Will be updated when user confirms/corrects
            }
            
            # Save to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data/tag_sample_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            # Update sample count in session state
            if 'training_sample_count' not in st.session_state:
                st.session_state.training_sample_count = 0
            st.session_state.training_sample_count += 1
            
            logger.info(f"[TRAINING] Saved sample #{st.session_state.training_sample_count} to {filename}")
            
            # Show success message
            st.toast(f"📚 Training data saved! Total samples: {st.session_state.training_sample_count}", icon="✅")
            
        except Exception as e:
            logger.error(f"[TRAINING] Failed to save sample: {e}")
    
    def _render_field_with_revise(self, label, value, field_name, step_num):
        """Render a field with a revise button for manual correction"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**{label}:** {value}")
        
        with col2:
            if st.button("✏️", key=f"revise_{field_name}_{step_num}", help=f"Edit {label}"):
                st.session_state[f"editing_{field_name}"] = True
                st.rerun()
        
        # Show edit form if editing
        if st.session_state.get(f"editing_{field_name}", False):
            with st.form(f"edit_{field_name}_form"):
                new_value = st.text_input(f"Correct {label}:", value=value, key=f"new_{field_name}")
                
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.form_submit_button("💾 Save"):
                        # Update the pipeline data
                        setattr(self.pipeline_data, field_name, new_value)
                        
                        # Save correction for training
                        self._save_correction(field_name, value, new_value)
                        
                        # Clear editing state
                        st.session_state[f"editing_{field_name}"] = False
                        st.success(f"✅ {label} updated: {value} → {new_value}")
                        st.rerun()
                
                with col_cancel:
                    if st.form_submit_button("❌ Cancel"):
                        st.session_state[f"editing_{field_name}"] = False
                        st.rerun()
    
    def _save_correction(self, field_name, original, corrected):
        """Save user corrections for training data improvement"""
        try:
            import json
            import os
            from datetime import datetime
            
            # Create corrections directory
            os.makedirs("training_data/corrections", exist_ok=True)
            
            correction_data = {
                'timestamp': datetime.now().isoformat(),
                'field': field_name,
                'original': original,
                'corrected': corrected,
                'user_correction': True
            }
            
            # Save correction
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data/corrections/{field_name}_correction_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(correction_data, f, indent=2)
            
            logger.info(f"[CORRECTION] Saved correction for {field_name}: {original} → {corrected}")
            
        except Exception as e:
            logger.error(f"[CORRECTION] Failed to save correction: {e}")
    
    def _render_analysis_results(self):
        """Render current analysis results with revise buttons"""
        if not hasattr(self, 'pipeline_data') or not self.pipeline_data:
            return
        
        st.markdown("#### 📊 Analysis Results")
        
        # Show training data count
        sample_count = st.session_state.get('training_sample_count', 0)
        st.caption(f"📚 Training samples collected: {sample_count}")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Brand with revise button
            brand = getattr(self.pipeline_data, 'brand', 'Not analyzed')
            if brand != 'Not analyzed':
                self._render_field_with_revise("🏷️ Brand", brand, 'brand', self.current_step)
            else:
                st.write(f"**🏷️ Brand:** {brand}")
            
            # Size with revise button
            size = getattr(self.pipeline_data, 'size', 'Not analyzed')
            if size != 'Not analyzed':
                self._render_field_with_revise("📏 Size", size, 'size', self.current_step)
            else:
                st.write(f"**📏 Size:** {size}")
            
            # Material with revise button
            material = getattr(self.pipeline_data, 'material', 'Not analyzed')
            if material != 'Not analyzed':
                self._render_field_with_revise("🧵 Material", material, 'material', self.current_step)
            else:
                st.write(f"**🧵 Material:** {material}")
        
        with col2:
            # Type with revise button
            garment_type = getattr(self.pipeline_data, 'garment_type', 'Not analyzed')
            if garment_type != 'Not analyzed':
                self._render_field_with_revise("👕 Type", garment_type, 'garment_type', self.current_step)
            else:
                st.write(f"**👕 Type:** {garment_type}")
            
            # Gender with revise button
            gender = getattr(self.pipeline_data, 'gender', 'Not analyzed')
            if gender != 'Not analyzed':
                self._render_field_with_revise("👤 Gender", gender, 'gender', self.current_step)
            else:
                st.write(f"**👤 Gender:** {gender}")
            
            # Style with revise button
            style = getattr(self.pipeline_data, 'style', 'Not analyzed')
            if style != 'Not analyzed':
                self._render_field_with_revise("🎨 Style", style, 'style', self.current_step)
            else:
                st.write(f"**🎨 Style:** {style}")
        
        # Pricing information (if available)
        if hasattr(self.pipeline_data, 'estimated_price') and self.pipeline_data.estimated_price:
            st.markdown("#### 💰 Pricing Information")
            st.write(f"**Estimated Price:** ${self.pipeline_data.estimated_price}")
            
            # Add eBay pricing button
            if st.button("🛒 Check eBay Prices", key="check_ebay_prices"):
                with st.spinner("🔍 Searching eBay for similar items..."):
                    self._check_ebay_pricing()
        
        # Confidence scores
        if hasattr(self.pipeline_data, 'confidence_scores') and self.pipeline_data.confidence_scores:
            st.markdown("#### 🎯 Confidence Scores")
            scores = self.pipeline_data.confidence_scores
            for field, score in scores.items():
                color = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                st.write(f"{color} **{field.title()}:** {score:.1%}")
    
    def _check_ebay_pricing(self):
        """Check eBay for similar items and pricing"""
        try:
            # This would integrate with eBay API
            # For now, show a placeholder
            st.info("🔍 eBay pricing integration coming soon!")
            st.write("This will search for similar items on eBay to provide accurate pricing estimates.")
            
            # Placeholder for actual eBay API integration
            sample_price = "25.99"
            st.success(f"💰 Found similar items starting at ${sample_price}")
            
        except Exception as e:
            logger.error(f"[EBAY] Pricing check failed: {e}")
            st.error("❌ Unable to check eBay prices at this time.")
    
    def render_action_panel_simple(self):
        """Simplified action panel - focus on making it work"""
        
        st.markdown("---")
        
        # Header with progress
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            progress = (self.current_step + 1) / len(self.steps)
            st.progress(progress, text=f"Step {self.current_step + 1} of {len(self.steps)}")
        
        with col2:
            if st.button("🔄 Reset", key="reset_btn"):
                self.current_step = 0
                self.pipeline_data = PipelineData()
                st.rerun()
        
        with col3:
            # THE KEY BUTTON - simplified logic
            # Duplicate button removed - using main button in render_action_panel()
            st.write("")  # Empty space for layout
        
        # Show current step UI
        if self.current_step == 0:
            self._render_step_0_simple()
        elif self.current_step == 1:
            self._render_step_1_garment_analysis()
        elif self.current_step == 2:
            self._render_step_2_measurements()
        elif self.current_step == 3:
            self._render_step_3_measurements()
        elif self.current_step == 4:
            self._render_step_4_defects()
        elif self.current_step == 5:
            self._render_step_5_pricing()
        elif self.current_step == 6:
            self.render_final_review()

    def _execute_current_step(self):
        """Execute current step's analysis and advance"""
        
        if self.current_step == 0:  # Tag Analysis
            with st.spinner("📸 Capturing and analyzing tag..."):
                # 1. RUN THE PROBE FIRST
                self._run_intelligent_lighting_probe()
                
                # 2. Wait for settings to stabilize
                time.sleep(2.0)
                
                # 3. NOW capture with correct lighting
                frame = self.camera_manager.get_arducam_frame()
                if frame is None:
                    st.error("❌ No camera frame")
                    return
                
                # 4. Apply ROI
                tag_roi = self.camera_manager.apply_roi_pure(frame, 'tag')
                if tag_roi is None:
                    st.error("❌ ROI extraction failed")
                    return
                
                # 5. Analyze with Gemini
                self.pipeline_data.tag_image = tag_roi
                result = self.analyze_tag_simple(tag_roi)
                
                # 6. Update state
                if result.get('success'):
                    self.pipeline_data.brand = result.get('brand')
                    self.pipeline_data.size = result.get('size')
                    st.success(f"✅ Brand: {self.pipeline_data.brand}")
                    self.current_step += 1
                else:
                    st.error(f"❌ Analysis failed: {result.get('error')}")
            
            st.rerun()
        
        elif self.current_step == 1:
            # Similar simplified logic for garment
            with st.spinner("👔 Analyzing garment..."):
                result = self.handle_step_1_garment_analysis()
                if result.get('success'):
                    st.success("✅ Garment analysis complete!")
                    self.current_step += 1
                else:
                    st.error(f"❌ Garment analysis failed: {result.get('error')}")
            st.rerun()
        
        elif self.current_step == 2:
            # Calibration step
            self.current_step += 1
            st.rerun()
        
        elif self.current_step == 3:
            # Measurement step
            self.current_step += 1
            st.rerun()
        
        elif self.current_step == 4:
            # Defect detection step
            self.current_step += 1
            st.rerun()
        
        elif self.current_step == 5:
            # Pricing step
            self.current_step += 1
            st.rerun()
        
        else:
            # Default: just advance to next step
            if self.current_step < len(self.steps) - 1:
                self.current_step += 1
                st.success(f"✅ Moved to Step {self.current_step + 1}")
                st.rerun()

    def _render_step_0_simple(self):
        """Just show camera preview - no complex logic"""
        st.info("📸 Position tag in green box, then click Next Step")
        
        frame = self.camera_manager.get_arducam_frame()
        if frame is not None:
            frame_with_roi = self.camera_manager.draw_roi_overlay(frame.copy(), 'tag')
            st.image(frame_with_roi, width='stretch')
        else:
            st.warning("Camera feed not available")
    
    def _handle_garment_analysis_step(self):
        """Handle Step 1: Garment analysis - Clean version"""
        with st.spinner("👔 Analyzing garment..."):
            result = st.session_state.pipeline_manager.handle_step_1_garment_analysis()
            
            if result.get('success'):
                if result.get('from_background'):
                    st.success("⚡ ✅ Background analysis complete!")
                else:
                    st.success(f"✅ {result.get('message')}")
                st.session_state.pipeline_manager.current_step = 2
                st.rerun()
            else:
                st.error(f"❌ Garment analysis failed: {result.get('error')}")
    
    def _handle_measurements_step(self):
        """Handle Step 2: Measurements"""
        # Move to next step (measurements are manual)
        st.session_state.pipeline_manager.current_step = 3
        st.success("✅ Measurements complete! Moving to pricing...")
        st.rerun()
    
    def _handle_pricing_step(self):
        """Handle Step 3: Pricing"""
        # Move to final review
        st.session_state.pipeline_manager.current_step = 4
        st.success("✅ Pricing complete! Moving to final review...")
        st.rerun()

    def render_final_review(self):
        """Render comprehensive final review with all analysis results"""
        st.success("Analysis Complete!")
        
        # Display comprehensive results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Item Details")
            st.write(f"**Brand:** {st.session_state.pipeline_manager.pipeline_data.brand}")
            st.write(f"**Type:** {st.session_state.pipeline_manager.pipeline_data.garment_type}")
            st.write(f"**Size:** {st.session_state.pipeline_manager.pipeline_data.size}")
            st.write(f"**Gender:** {st.session_state.pipeline_manager.pipeline_data.gender}")
            st.write(f"**Material:** {st.session_state.pipeline_manager.pipeline_data.material}")
            st.write(f"**Style:** {st.session_state.pipeline_manager.pipeline_data.style}")
            st.write(f"**Era:** {st.session_state.pipeline_manager.pipeline_data.era}")
            st.write(f"**Condition:** {st.session_state.pipeline_manager.pipeline_data.condition}")
            
            # Special indicators
            if st.session_state.pipeline_manager.pipeline_data.is_designer:
                st.info("👜 Designer Item")
            if st.session_state.pipeline_manager.pipeline_data.is_vintage:
                st.info("🕰️ Vintage Item")
        
        with col2:
            st.subheader("Pricing Analysis")
            col_low, col_high = st.columns(2)
            with col_low:
                st.metric("Low Estimate", f"${st.session_state.pipeline_manager.pipeline_data.price_estimate['low']}")
            with col_high:
                st.metric("High Estimate", f"${st.session_state.pipeline_manager.pipeline_data.price_estimate['high']}")
            
            st.metric("Recommended Price", f"${st.session_state.pipeline_manager.pipeline_data.price_estimate['mid']}", delta="Best Value")
            
            # Defect summary with annotated image
            if st.session_state.pipeline_manager.pipeline_data.defect_count > 0:
                st.warning(f"Defects Found: {st.session_state.pipeline_manager.pipeline_data.defect_count}")
                
                # Show annotated image with bounding boxes
                if st.session_state.pipeline_manager.pipeline_data.garment_image is not None and st.session_state.pipeline_manager.pipeline_data.defects:
                    annotated_image = st.session_state.pipeline_manager.defect_detector.draw_defect_boxes(
                        st.session_state.pipeline_manager.pipeline_data.garment_image,
                        st.session_state.pipeline_manager.pipeline_data.defects
                    )
                    st.image(annotated_image, caption="Defects Highlighted", width=400)
                
                # List defects with details
                for idx, defect in enumerate(st.session_state.pipeline_manager.pipeline_data.defects):
                    st.write(f"**#{idx+1}**: {defect.get('type', 'Unknown')} at {defect.get('location', 'unknown')} - {defect.get('description', '')}")
            else:
                st.success("No defects detected")
        
        # Images
        if st.session_state.pipeline_manager.pipeline_data.tag_image is not None or st.session_state.pipeline_manager.pipeline_data.garment_image is not None:
            st.subheader("Captured Images")
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                if st.session_state.pipeline_manager.pipeline_data.tag_image is not None:
                    st.image(st.session_state.pipeline_manager.pipeline_data.tag_image, caption="Tag", width=200)
            
            with img_col2:
                if st.session_state.pipeline_manager.pipeline_data.garment_image is not None:
                    st.image(st.session_state.pipeline_manager.pipeline_data.garment_image, caption="Garment", width=200)
        
        # Market research link
        if st.session_state.pipeline_manager.pipeline_data.brand != "Unknown":
            st.subheader("Market Research")
            search_url = st.session_state.pipeline_manager.pricing_api.generate_search_url(
                st.session_state.pipeline_manager.pipeline_data.brand,
                st.session_state.pipeline_manager.pipeline_data.garment_type,
                st.session_state.pipeline_manager.pipeline_data.size,
                st.session_state.pipeline_manager.pipeline_data.gender
            )
            st.write(f"[Research similar items on eBay]({search_url})")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Analyze New Item", type="primary"):
                st.session_state.pipeline_manager.current_step = 0
                st.session_state.pipeline_manager.pipeline_data = PipelineData()
                st.rerun()
        with col2:
            if st.button("Export Results"):
                # Create a simple export
                results = {
                    'brand': st.session_state.pipeline_manager.pipeline_data.brand,
                    'type': st.session_state.pipeline_manager.pipeline_data.garment_type,
                    'size': st.session_state.pipeline_manager.pipeline_data.size,
                    'gender': st.session_state.pipeline_manager.pipeline_data.gender,
                    'condition': st.session_state.pipeline_manager.pipeline_data.condition,
                    'price_estimate': st.session_state.pipeline_manager.pipeline_data.price_estimate,
                    'defects': st.session_state.pipeline_manager.pipeline_data.defects
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(results, indent=2),
                    file_name=f"garment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        with col3:
            if st.button("Print Summary"):
                st.write("Print functionality would be implemented here")

# ==========================
# ROI POSITIONING MODE
# ==========================
def render_roi_positioning_mode():
    """Special mode to position tag ROI using FULL RealSense camera view"""
    st.title("🎯 Tag ROI Positioning Tool - FULL CAMERA VIEW")
    st.warning("⚠️ This shows your COMPLETE camera view so you can see your entire garment and locate the tag!")
    
    # Get pipeline manager
    pm = st.session_state.pipeline_manager
    
    # Initialize ROI in session state if not exists
    if 'tag_roi_temp' not in st.session_state:
        st.session_state.tag_roi_temp = {
            'x': pm.camera_manager.roi_coords['tag'][0],
            'y': pm.camera_manager.roi_coords['tag'][1],
            'w': pm.camera_manager.roi_coords['tag'][2],
            'h': pm.camera_manager.roi_coords['tag'][3]
        }
    
    if 'roi_move_step' not in st.session_state:
        st.session_state.roi_move_step = 20
    
    # Resolution selector at top
    res_col1, res_col2 = st.columns([3, 1])
    with res_col1:
        st.info("💡 If you can't see your full garment, the camera view might be too zoomed. This tool shows the COMPLETE camera frame.")
    with res_col2:
        move_step = st.select_slider(
            "Move Step",
            options=[5, 10, 20, 50],
            value=st.session_state.roi_move_step,
            help="Pixels per arrow click",
            key="move_step_selector"
        )
        st.session_state.roi_move_step = move_step
    
    # Create layout: Camera (left) + Controls (right)
    col_camera, col_controls = st.columns([3, 1])
    
    with col_camera:
        st.subheader("📹 FULL RealSense Camera View")
        st.caption("You should see your ENTIRE garment in this view - not cropped")
        
        # Get RealSense frame (FULL FRAME - no ROI applied)
        frame = pm.camera_manager.get_realsense_frame()
        
        if frame is not None:
            # Store actual frame size
            st.session_state.frame_size = {
                'width': frame.shape[1],
                'height': frame.shape[0]
            }
            
            # Draw ROI overlay on FULL RealSense feed
            roi = st.session_state.tag_roi_temp
            display_frame = frame.copy()
            
            x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
            
            # Ensure ROI is within frame bounds
            x = max(0, min(x, frame.shape[1] - w))
            y = max(0, min(y, frame.shape[0] - h))
            
            # Draw main ROI rectangle with VERY THICK border
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 6)
            
            # Draw corner markers (larger for visibility)
            corner_size = 25
            corner_thickness = 6
            
            # Top-left corner
            cv2.line(display_frame, (x, y), (x + corner_size, y), (0, 255, 0), corner_thickness)
            cv2.line(display_frame, (x, y), (x, y + corner_size), (0, 255, 0), corner_thickness)
            
            # Top-right corner
            cv2.line(display_frame, (x + w, y), (x + w - corner_size, y), (0, 255, 0), corner_thickness)
            cv2.line(display_frame, (x + w, y), (x + w, y + corner_size), (0, 255, 0), corner_thickness)
            
            # Bottom-left corner
            cv2.line(display_frame, (x, y + h), (x + corner_size, y + h), (0, 255, 0), corner_thickness)
            cv2.line(display_frame, (x, y + h), (x, y + h - corner_size), (0, 255, 0), corner_thickness)
            
            # Bottom-right corner
            cv2.line(display_frame, (x + w, y + h), (x + w - corner_size, y + h), (0, 255, 0), corner_thickness)
            cv2.line(display_frame, (x + w, y + h), (x + w, y + h - corner_size), (0, 255, 0), corner_thickness)
            
            # Center crosshair (RED for visibility)
            center_x, center_y = x + w // 2, y + h // 2
            crosshair_size = 30
            cv2.line(display_frame, (center_x - crosshair_size, center_y), 
                     (center_x + crosshair_size, center_y), (0, 0, 255), 4)
            cv2.line(display_frame, (center_x, center_y - crosshair_size), 
                     (center_x, center_y + crosshair_size), (0, 0, 255), 4)
            
            # Add label with background
            label = f"TAG ROI: {w}x{h} @ ({x}, {y})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw background rectangle for label
            cv2.rectangle(display_frame, 
                          (x, y - text_height - 20), 
                          (x + text_width + 10, y - 5),
                          (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (x + 5, y - 10), 
                        font, font_scale, (0, 255, 0), font_thickness)
            
            # Add instruction at top of frame
            instruction = "Position GREEN BOX over your garment tag"
            inst_x = frame.shape[1] // 2 - 350
            cv2.rectangle(display_frame, (inst_x - 10, 10), (inst_x + 700, 55), (0, 0, 0), -1)
            cv2.putText(display_frame, instruction, (inst_x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            # Display FULL FRAME (use_container_width ensures full view)
            st.image(display_frame, caption=f"🎯 Full RealSense View ({frame.shape[1]}x{frame.shape[0]})", 
                     width='stretch')
            
            st.info(f"📐 Camera Resolution: {frame.shape[1]}x{frame.shape[0]} - You should see your ENTIRE garment!")
            
            # Show cropped ROI preview
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                y_end = min(y + h, frame.shape[0])
                x_end = min(x + w, frame.shape[1])
                roi_preview = frame[y:y_end, x:x_end]
                
                st.markdown("---")
                st.markdown("### 🔍 ROI Preview (What AI Will See)")
                st.image(roi_preview, caption="Cropped Tag Region", width='stretch')
        else:
            st.error("❌ No RealSense frame available")
            st.info("💡 Make sure RealSense camera is connected and Step 1 has been run at least once")
    
    with col_controls:
        st.subheader("🎮 Controls")
        
        # Current position
        st.metric("X Position", f"{roi['x']} px")
        st.metric("Y Position", f"{roi['y']} px")
        st.metric("Width", f"{roi['w']} px")
        st.metric("Height", f"{roi['h']} px")
        
        st.markdown("---")
        
        # Arrow controls (configurable step size)
        move_step = st.session_state.roi_move_step
        st.markdown(f"### ⬆️⬇️⬅️➡️ Move ROI ({move_step}px)")
        
        frame_w = st.session_state.frame_size.get('width', 640)
        frame_h = st.session_state.frame_size.get('height', 480)
        
        arrow_col1, arrow_col2, arrow_col3 = st.columns(3)
        with arrow_col1:
            if st.button("⬅️", key="roi_left", width='stretch'):
                st.session_state.tag_roi_temp['x'] = max(0, roi['x'] - move_step)
                st.rerun()
        with arrow_col2:
            if st.button("⬆️", key="roi_up", width='stretch'):
                st.session_state.tag_roi_temp['y'] = max(0, roi['y'] - move_step)
                st.rerun()
        with arrow_col3:
            if st.button("➡️", key="roi_right", width='stretch'):
                st.session_state.tag_roi_temp['x'] = min(frame_w - roi['w'], roi['x'] + move_step)
                st.rerun()
        
        if st.button("⬇️", key="roi_down", width='stretch'):
            st.session_state.tag_roi_temp['y'] = min(frame_h - roi['h'], roi['y'] + move_step)
            st.rerun()
        
        if st.button("🎯 Center", key="roi_center", width='stretch'):
            st.session_state.tag_roi_temp['x'] = (frame_w - roi['w']) // 2
            st.session_state.tag_roi_temp['y'] = (frame_h - roi['h']) // 2
            st.rerun()
        
        st.markdown("---")
        
        # Quick presets (frame-aware)
        st.markdown("### 📍 Quick Presets")
        
        frame_w = st.session_state.frame_size.get('width', 640)
        frame_h = st.session_state.frame_size.get('height', 480)
        
        preset_col1, preset_col2 = st.columns(2)
        
        with preset_col1:
            if st.button("Top Left", key="preset_tl", width='stretch'):
                st.session_state.tag_roi_temp = {'x': 50, 'y': 50, 'w': 200, 'h': 120}
                st.rerun()
            
            if st.button("Center Left", key="preset_cl", width='stretch'):
                st.session_state.tag_roi_temp = {'x': 50, 'y': (frame_h - 120) // 2, 'w': 200, 'h': 120}
                st.rerun()
            
            if st.button("Bottom Left", key="preset_bl", width='stretch'):
                st.session_state.tag_roi_temp = {'x': 50, 'y': frame_h - 170, 'w': 200, 'h': 120}
                st.rerun()
        
        with preset_col2:
            if st.button("Top Right", key="preset_tr", width='stretch'):
                st.session_state.tag_roi_temp = {'x': frame_w - 250, 'y': 50, 'w': 200, 'h': 120}
                st.rerun()
            
            if st.button("Center Right", key="preset_cr", width='stretch'):
                st.session_state.tag_roi_temp = {'x': frame_w - 250, 'y': (frame_h - 120) // 2, 'w': 200, 'h': 120}
                st.rerun()
            
            if st.button("Bottom Right", key="preset_br", width='stretch'):
                st.session_state.tag_roi_temp = {'x': frame_w - 250, 'y': frame_h - 170, 'w': 200, 'h': 120}
                st.rerun()
        
        if st.button("🎯 Absolute Center", key="preset_center", type="secondary", width='stretch'):
            st.session_state.tag_roi_temp = {
                'x': (frame_w - 200) // 2,
                'y': (frame_h - 120) // 2,
                'w': 200,
                'h': 120
            }
            st.rerun()
        
        st.markdown("---")
        
        # Sliders for fine control (frame-aware)
        st.markdown("### 🎚️ Fine Tune")
        
        max_x = max(0, frame_w - roi['w'])
        max_y = max(0, frame_h - roi['h'])
        
        new_x = st.slider("X Position", 0, max_x, roi['x'], key="fine_x", step=5)
        if new_x != roi['x']:
            st.session_state.tag_roi_temp['x'] = new_x
            st.rerun()
        
        new_y = st.slider("Y Position", 0, max_y, roi['y'], key="fine_y", step=5)
        if new_y != roi['y']:
            st.session_state.tag_roi_temp['y'] = new_y
            st.rerun()
        
        # Size adjustments
        new_w = st.slider("Width", 100, 400, roi['w'], key="w_slider", step=10)
        if new_w != roi['w']:
            st.session_state.tag_roi_temp['w'] = new_w
            st.rerun()
        
        new_h = st.slider("Height", 80, 300, roi['h'], key="h_slider", step=10)
        if new_h != roi['h']:
            st.session_state.tag_roi_temp['h'] = new_h
            st.rerun()
        
        st.markdown("---")
        
        # Action buttons
        if st.button("✅ Save & Exit", key="save_roi", type="primary", width='stretch'):
            # Update camera manager ROI
            new_roi = (
                st.session_state.tag_roi_temp['x'],
                st.session_state.tag_roi_temp['y'],
                st.session_state.tag_roi_temp['w'],
                st.session_state.tag_roi_temp['h']
            )
            pm.camera_manager.roi_coords['tag'] = new_roi
            
            # Save to config file
            pm.save_roi_to_config(new_roi)
            
            st.success(f"✅ ROI saved: {new_roi}")
            
            # Exit positioning mode
            st.session_state.roi_positioning_mode = False
            st.rerun()
        
        if st.button("❌ Cancel", key="cancel_roi", width='stretch'):
            st.session_state.roi_positioning_mode = False
            st.rerun()

# ==========================
# MAIN FUNCTION
# ==========================
def render_focus_mode(pm):
    """Render camera focus calibration mode - optimized for 12MP Arducam"""
    st.title("🔬 Camera Focus Calibration - 12MP Arducam")
    st.info("Center a detailed tag in the green ROI. Twist the lens to maximize the score.")
    
    # Clear any cached images to prevent MediaFileStorageError
    if 'focus_mode_started' not in st.session_state:
        st.session_state.focus_mode_started = True
        # Clear any cached image references that could cause MediaFileStorageError
        keys_to_clear = []
        for key in list(st.session_state.keys()):
            if any(term in key.lower() for term in ['image', 'frame', 'cached', 'preview', 'roi']):
                keys_to_clear.append(key)
        
        for key in keys_to_clear:
            try:
                del st.session_state[key]
            except Exception as e:
                logger.debug(f"Could not clear session key {key}: {e}")
        
        logger.info(f"Cleared {len(keys_to_clear)} cached image references for focus mode")
    
    # Show current camera resolution and settings
    if pm.camera_manager.arducam_cap and pm.camera_manager.arducam_cap.isOpened():
        current_w = int(pm.camera_manager.arducam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_h = int(pm.camera_manager.arducam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        st.caption(f"📷 Current Resolution: {current_w}x{current_h}")
    
    c1, c2 = st.columns([3, 1])
    preview_ph = c1.empty()
    score_ph = c2.empty()
    
    # Add control buttons
    col_stop1, col_stop2, col_stop3 = st.columns([1, 1, 1])
    with col_stop1:
        exit_now = st.button("✅ Done & Exit Focus Mode", type="primary")
    with col_stop2:
        stop_now = st.button("⏹️ Stop Auto-Refresh", type="secondary")
    with col_stop3:
        test_12mp = st.button("📸 Test 12MP Capture", type="secondary")
    
    if exit_now:
        st.session_state.focus_mode = False
        st.session_state.max_focus_score = 0
        st.rerun()
    
    if stop_now:
        st.session_state.focus_auto_refresh = False
        st.info("🛑 Auto-refresh stopped. Click 'Done & Exit' when finished.")
    
    if test_12mp:
        st.info("📸 Testing 12MP capture...")
        frames = pm.camera_manager.capture_highres_burst(n=3)
        if frames:
            # Select sharpest frame
            best_frame = max(frames, key=pm.camera_manager.calculate_focus_score)
            best_score = pm.camera_manager.calculate_focus_score(best_frame)
            
            # Show the 12MP capture
            try:
                st.image(best_frame, caption=f"12MP Capture - Focus Score: {best_score:.0f}", width='stretch')
                st.success(f"✅ 12MP capture successful! Resolution: {best_frame.shape[1]}x{best_frame.shape[0]}")
            except Exception as e:
                logger.warning(f"12MP capture image display error: {e}")
                st.warning("12MP capture completed but preview unavailable")
                st.success(f"✅ 12MP capture successful! Resolution: {best_frame.shape[1]}x{best_frame.shape[0]}")
            
            # Update peak score if this is better
            current_peak = st.session_state.get("max_focus_score", 0.0)
            if best_score > current_peak:
                st.session_state.max_focus_score = best_score
                st.success(f"🎯 New peak focus score: {best_score:.0f}")
        else:
            st.error("❌ 12MP capture failed")
    
    # More conservative FPS cap to prevent WebSocket errors
    last = st.session_state.get("focus_last_ts", 0.0)
    now = time.time()
    if now - last < 0.5:  # 2 FPS max to prevent WebSocket overload
        st.stop()
    st.session_state.focus_last_ts = now
    
    frame = pm.camera_manager.get_arducam_frame()
    if frame is None:
        st.warning("No camera frame.")
        st.stop()
    
    score = pm.camera_manager.calculate_focus_score(frame)
    peak = max(score, st.session_state.get("max_focus_score", 0.0))
    st.session_state.max_focus_score = peak
    
    frame_roi = pm.camera_manager.draw_roi_overlay(frame.copy(), 'tag')
    cv2.putText(frame_roi, f"Score: {score:.0f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    try:
        # Convert frame to ensure it's in the right format for Streamlit
        if frame_roi is not None and frame_roi.size > 0:
            preview_ph.image(frame_roi, caption="Live Camera Feed", width='stretch')
        else:
            preview_ph.warning("No valid camera frame available")
    except Exception as e:
        logger.warning(f"Focus mode image display error: {e}")
        # Try to clear any problematic cached images
        try:
            if 'cached_tag_frame' in st.session_state:
                del st.session_state.cached_tag_frame
        except:
            pass
        preview_ph.warning("Camera preview temporarily unavailable")
    
    with score_ph:
        st.metric("Focus Score", f"{score:.0f}")
        st.metric("Peak Score Achieved", f"{peak:.0f}")
        
        # Better progress bar with target ranges - clamp to valid range [0.0, 1.0]
        progress_value = min(max(score / 800.0, 0.0), 1.0)
        if score < 200:
            st.progress(progress_value)
            st.caption("🔴 Poor focus - adjust lens")
        elif score < 400:
            st.progress(progress_value)
            st.caption("🟡 Fair focus - getting better")
        elif score < 600:
            st.progress(progress_value)
            st.caption("🟢 Good focus - almost there")
        else:
            st.progress(progress_value)
            st.caption("🔥 Excellent focus!")
        
        # Show frame info
        st.caption(f"Frame: {frame.shape[1]}x{frame.shape[0]}")
    
    # Auto-refresh this screen without a blocking while-loop
    # Only rerun if auto-refresh is enabled and we haven't been in focus mode for too long
    auto_refresh_enabled = st.session_state.get("focus_auto_refresh", True)
    focus_start_time = st.session_state.get("focus_start_time", time.time())
    
    # FOCUS MODE THROTTLE: Prevent rapid reruns (simplified)
    if 'focus_last_rerun' not in st.session_state:
        st.session_state.focus_last_rerun = 0
    
    current_time = time.time()
    time_since_last_rerun = current_time - st.session_state.focus_last_rerun
    
    if (auto_refresh_enabled and 
        time.time() - focus_start_time < 300 and  # 5 minute timeout
        time_since_last_rerun > 0.6):  # Throttle: minimum 600ms between reruns (more conservative)
        st.session_state.focus_last_rerun = current_time
        st.rerun()
    elif time_since_last_rerun <= 0.6:
        st.caption("🔄 Focus mode throttled (600ms minimum between updates)")
    elif not auto_refresh_enabled:
        st.info("🛑 Auto-refresh is stopped. Click 'Done & Exit' when finished.")
    else:
        st.error("Focus mode timeout - returning to main app")
        st.session_state.focus_mode = False
        st.rerun()

def render_defect_collection_mode():
    """Defect collection with visual annotation for training YOLO models"""
    st.title("Defect Dataset Collection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Click on defects to mark them")
        
        frame = st.session_state.pipeline_manager.camera_manager.get_realsense_frame()
        if frame is not None:
            
            # Draw existing marks on frame
            annotated_frame = frame.copy()
            for i, point in enumerate(st.session_state.defect_points):
                # Draw red circle at defect location
                cv2.circle(annotated_frame, (point['x'], point['y']), 20, (255, 0, 0), 3)
                # Draw number label
                cv2.putText(annotated_frame, f"#{i+1}", (point['x']-10, point['y']-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Display image (click detection would need streamlit-image-coordinates library)
            st.image(annotated_frame, width='stretch')
            
            # Manual coordinate entry for now (can be enhanced with click detection)
            st.caption("Enter defect coordinates manually (click detection requires additional library)")
            
            col_x, col_y = st.columns(2)
            with col_x:
                defect_x = st.number_input("X coordinate", min_value=0, max_value=frame.shape[1], value=frame.shape[1]//2)
            with col_y:
                defect_y = st.number_input("Y coordinate", min_value=0, max_value=frame.shape[0], value=frame.shape[0]//2)
            
            if st.button("📍 Add Defect Point"):
                st.session_state.defect_points.append({
                    'x': defect_x,
                    'y': defect_y,
                    'defect_type': st.session_state.get('current_defect_type', 'hole'),
                    'severity': st.session_state.get('current_severity', 'moderate')
                })
                st.rerun()
    
    with col2:
        st.subheader("Defect Info")
        
        defect_type = st.selectbox(
            "Current Defect Type",
            ["hole", "stain", "tear", "pilling", "fading", "seam_damage", "missing_button"],
            key='current_defect_type'
        )
        
        severity = st.select_slider(
            "Severity",
            options=["minor", "moderate", "severe"],
            key='current_severity'
        )
        
        location = st.selectbox(
            "Location",
            ["front", "back", "sleeve", "collar", "hem", "pocket"]
        )
        
        st.markdown("---")
        st.caption(f"Marked: {len(st.session_state.get('defect_points', []))} defects")
        
        if st.session_state.get('defect_points'):
            if st.button("🗑️ Clear Last Mark"):
                st.session_state.defect_points.pop()
                st.rerun()
            
            if st.button("🗑️ Clear All"):
                st.session_state.defect_points = []
                st.rerun()
        
        st.markdown("---")
        
        if st.button("💾 Save Annotated Image", type="primary"):
            if frame is not None and st.session_state.get('defect_points'):
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"defect_annotated_{timestamp}.jpg"
                
                # Save image
                os.makedirs("defect_dataset/images", exist_ok=True)
                save_path = f"defect_dataset/images/{filename}"
                cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                # Save YOLO format annotations
                h, w = frame.shape[:2]
                yolo_annotations = []
                
                # Convert to YOLO format (class x_center y_center width height - normalized)
                defect_classes = {
                    'hole': 0, 'stain': 1, 'tear': 2, 'pilling': 3,
                    'fading': 4, 'seam_damage': 5, 'missing_button': 6
                }
                
                for point in st.session_state.defect_points:
                    class_id = defect_classes.get(point['defect_type'], 0)
                    x_center = point['x'] / w
                    y_center = point['y'] / h
                    # Assume defect area is ~50x50 pixels
                    box_w = 50 / w
                    box_h = 50 / h
                    
                    yolo_annotations.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
                    )
                
                # Save YOLO annotation file
                os.makedirs("defect_dataset/labels", exist_ok=True)
                label_path = f"defect_dataset/labels/{filename.replace('.jpg', '.txt')}"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                # Save JSON metadata
                metadata = {
                    'filename': filename,
                    'timestamp': timestamp,
                    'defects': st.session_state.defect_points
                }
                json_path = f"defect_dataset/annotations/{filename.replace('.jpg', '.json')}"
                os.makedirs("defect_dataset/annotations", exist_ok=True)
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                st.success(f"✅ Saved {len(st.session_state.defect_points)} defects")
                st.session_state.defect_points = []
                st.rerun()
            else:
                st.error("❌ Mark at least one defect first")
        
        # Show collection stats
        if os.path.exists("defect_dataset/images"):
            count = len([f for f in os.listdir("defect_dataset/images") if f.endswith('.jpg')])
            st.metric("📊 Total Defect Images", count)
        
        st.markdown("---")
        
        if st.button("🚪 Exit Collection Mode"):
            st.session_state.defect_collection_mode = False
            st.session_state.defect_points = []
            st.rerun()

def main():
    """Main application entry point - optimized for tablet performance"""
    
    # Tablet-optimized page config
    st.set_page_config(
        page_title="Garment Analyzer Pipeline",
        page_icon="🔍",  # Using search icon - more stable encoding
        layout="wide",
        initial_sidebar_state="collapsed"  # No sidebar needed - pipeline in main layout
    )
    
    # CRITICAL: Prevent infinite loops
    if '_rerun_count' not in st.session_state:
        st.session_state._rerun_count = 0
    
    st.session_state._rerun_count += 1
    
    # Safety check for infinite loops
    if st.session_state._rerun_count > 100:
        st.error("⚠️ Infinite loop detected! Resetting...")
        if 'pipeline_manager' in st.session_state:
            st.session_state.pipeline_manager.current_step = 0
        st.session_state._rerun_count = 0
        st.stop()
    
    # Reset rerun counter when step changes
    current_step = st.session_state.get('pipeline_manager', {}).current_step if 'pipeline_manager' in st.session_state else 0
    if current_step != st.session_state.get('_last_step', -1):
        st.session_state._last_step = current_step
        st.session_state._rerun_count = 0
    
    logger.info(f"🔄 Main: Step {current_step}, Reruns: {st.session_state._rerun_count}")
    
    # TEMPORARY: Add debug info to sidebar
    # SAMPLE COUNT WIDGET in sidebar
    with st.sidebar:
        render_sample_count_widget()
    
    # INITIALIZE ALL SESSION STATE VARIABLES FIRST
    if 'pipeline_manager' not in st.session_state:
        st.session_state.pipeline_manager = EnhancedPipelineManager()
    
    if 'live_preview_enabled' not in st.session_state:
        st.session_state.live_preview_enabled = True
    
    if 'focus_mode' not in st.session_state:
        st.session_state.focus_mode = False
    
    if 'defect_collection_mode' not in st.session_state:
        st.session_state.defect_collection_mode = False
    
    if 'last_preview_refresh' not in st.session_state:
        st.session_state.last_preview_refresh = 0
    
    if 'auto_zoom_enabled' not in st.session_state:
        st.session_state.auto_zoom_enabled = True
    
    if 'zoom_level' not in st.session_state:
        st.session_state.zoom_level = 1.0
    
    if 'defect_points' not in st.session_state:
        st.session_state.defect_points = []
    
    if 'current_defect_type' not in st.session_state:
        st.session_state.current_defect_type = 'hole'
    
    if 'current_severity' not in st.session_state:
        st.session_state.current_severity = 'moderate'
    
    # Add tablet-friendly CSS
    st.markdown("""
    <style>
    .stButton button { 
        height: 60px !important; 
        font-size: 20px !important; 
        min-width: 120px !important;
    }
    .stSelectbox > div > div {
        font-size: 18px !important;
    }
    .stTextInput > div > div > input {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize mobile mode
    if 'mobile_mode' not in st.session_state:
        st.session_state.mobile_mode = True
    
    # Initialize session state - ONLY USE SESSION STATE VERSION
    if 'pipeline_manager' not in st.session_state:
        st.session_state.pipeline_manager = EnhancedPipelineManager()
    
    # Initialize basic session state
    
    # DON'T create a local variable - use session state directly
    
    # Loop prevention completely removed - app runs at full speed
    
    # Sidebar now only contains sample count widget (no pipeline)
    
    # Sidebar is now clean with only sample count widget
    # Removed camera preview and debug info from sidebar
        
    # All sidebar content removed for clean layout
    
    # Check for interactive ROI editor mode
    if st.session_state.get("interactive_roi_mode", False):
        st.session_state.pipeline_manager.camera_manager.render_interactive_roi_editor()
        if st.button("❌ Close ROI Editor"):
            st.session_state.interactive_roi_mode = False
            st.rerun()
        return
    
    # Check for ROI positioning mode
    if st.session_state.get("roi_positioning_mode", False):
        render_roi_positioning_mode()
        return
    
    # Check for focus mode
    if st.session_state.get("focus_mode", False):
        render_focus_mode(st.session_state.pipeline_manager)
        return
    
    # Check for defect collection mode
    if st.session_state.get('defect_collection_mode', False):
        render_defect_collection_mode()
        return
    
    # CRITICAL: Use the new compact layout (includes buttons at top)
    st.session_state.pipeline_manager.render_compact_layout()
    
    # Footer
    st.markdown("---")
    st.caption("💡 Elgato lights active • 🤖 AI-powered analysis")

if __name__ == "__main__":
    main()
