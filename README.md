# 🌾 AI Crop Recommendation System

An **AI-powered Crop Recommendation System** built using Machine Learning and Flask that helps farmers and agricultural planners choose the most suitable crops based on **soil nutrients, climate conditions, agro-climatic zones, and historical district performance data**.

The system provides **interactive map-based crop recommendations** along with **detailed farming guidance**.

---

## 🚀 Features

- 🗺️ Interactive map to select farm location
- 🤖 Machine Learning–based crop prediction
- 🌡️ Environmental analysis (N, P, K, temperature, humidity, pH, rainfall)
- 🏞️ Automatic agro-climatic zone detection
- 🏛️ Integration of district-level crop performance data
- 📊 Confidence score and suitability ranking
- 🌱 Detailed crop recommendations:
  - Soil preparation
  - Fertilizer usage
  - Irrigation guidance
  - Best crop varieties
  - Growing season (Kharif / Rabi)
- ⚡ Optimized model loading using caching

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Backend:** Flask  
- **Machine Learning:** scikit-learn, NumPy  
- **Frontend:** HTML, CSS, JavaScript  
- **Maps:** Leaflet.js (OpenStreetMap)  

---

## 📁 Project Structure

├── complete_crop_predictor_fixed.py
├── crop_cache/
│ └── models_cache.pkl.gz
└── README.md
---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install flask numpy pandas scikit-learn geopy
python complete_crop_predictor_fixed.py
http://127.0.0.1:5000/