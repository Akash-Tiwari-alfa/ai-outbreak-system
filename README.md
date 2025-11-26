# 🚑 **AI-Driven Public Health Outbreak Prediction System**

*A Machine Learning + Flask web platform for predicting outbreak risk, providing AI-generated recommendations, air quality impact, and region-wise health analytics.*

---

## 📌 **Overview**

The **AI-Driven Public Health Outbreak Prediction System** is a full-stack analytic platform that predicts disease outbreak risk using real-time health indicators such as:

* Fever cases
* Cough cases
* Diarrhea cases
* Region population

It enhances the prediction by integrating:

* 🌫️ **Air Quality Index (AQI)** for the region
* 🏥 **Nearby hospital availability**
* 🧠 **AI-generated actionable recommendations**
* 📊 **User-specific dashboards**
* 🔐 **User authentication (Login/Register)**

This system is designed for health departments, hospitals, and research teams who need a **smart, AI-powered outbreak early warning system**.

---

## 🎯 **Key Features**

### 🧠 **Machine Learning Model**

* RandomForestClassifier trained on health datasets
* Predicts outbreak probability (%)
* Categorizes risk into **Low / Medium / High**

---

### 🔐 **User Authentication**

* Register / Login / Logout system using Flask-Login
* Passwords hashed using `werkzeug.security`
* Every user gets their **own dashboard** and **private prediction history**

---

### 📈 **Dynamic User Dashboard**

* Region-wise stacked bar chart (powered by Chart.js)
* Priority regions with high outbreak probability
* Recent predictions table
* Data filtered **user-wise**, not global

---

### 🧪 **AI-Driven Recommendations**

Smart suggestions based on:

* Outbreak risk level
* Dominant symptoms (fever/cough/diarrhea)
* AQI category
* Local hospital availability
* Population density

Example suggestions:

* Increase testing capacity
* Strengthen respiratory hygiene campaigns
* Investigate water safety
* Issue public health advisories

---

### 🌫️ **Air Quality Index (AQI) Integration**

Each prediction fetches simulated (or API-ready) AQI:

* Good
* Satisfactory
* Moderate
* Poor
* Very Poor
* Severe

AQI influences outbreak recommendations automatically.

---

### 🏥 **Nearby Hospitals Mapping**

Based on region, users get:

* Hospital name
* Address
* Contact
* Region-wise availability

---

### 🗃️ **SQLite Database (SQLAlchemy ORM)**

Stores:

* Users
* Predictions
* AQI data
* Risk levels
* Region details

Keeps user dashboards consistent across sessions.

---

### 🌐 **REST API Endpoint**

Predict via API (for mobile apps or dashboards):

```
POST /api/predict
```

Body:

```json
{
  "region": "Surat",
  "fever_cases": 50,
  "cough_cases": 70,
  "diarrhea_cases": 20,
  "region_population": 15000
}
```

Returns:

```json
{
  "prediction": 1,
  "probability": 82.35,
  "risk_level": "High",
  "aqi": {"value": 130, "category": "Moderate"},
  "hospitals": [...],
  "suggestions": [...]
}
```

---

## 🛠️ **Tech Stack**

| Layer         | Technology                                 |
| ------------- | ------------------------------------------ |
| Frontend      | HTML5, CSS3, Bootstrap 5, Chart.js         |
| Backend       | Flask (Python)                             |
| ML Model      | Scikit-Learn (RandomForestClassifier)      |
| Database      | SQLite + SQLAlchemy                        |
| Auth          | Flask-Login                                |
| Packaging     | Joblib                                     |
| Hosting Ready | Render / PythonAnywhere / AWS EC2 / Docker |

---

## 📂 **Folder Structure**

```
ai_outbreak_system/
│
├── app.py                     # Main Flask application
├── train_model.py             # ML model trainer
├── requirements.txt           # Python dependencies
│
├── models/
│   └── outbreak_model.pkl     # Trained ML model
│
├── templates/                 # Frontend HTML templates
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── login.html
│   └── register.html
│
├── static/
│   └── style.css              # Custom CSS
│
├── data/
│   └── sample_health_data.csv # Training dataset
│
└── instance/
    └── outbreak.db            # SQLite database
```

---

## 🚀 **How to Run Locally (Ubuntu/Linux)**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/USERNAME/ai-outbreak-system.git
cd ai-outbreak-system
```

---

### 2️⃣ Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Train the ML model

```bash
python train_model.py
```

---

### 5️⃣ Run Flask server

```bash
python app.py
```

App will run at:

👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📸 **Screenshots (Add your own)**

> Replace placeholders with actual screenshots.

| Feature            | Screenshot                              |
| ------------------ | --------------------------------------- |
| Login Page         | (https://github.com/Akash-Tiwari-alfa/ai-outbreak-system/blob/main/Screenshot%20From%202025-11-26%2010-36-00.png)|
| Dashboard          | (https://github.com/Akash-Tiwari-alfa/ai-outbreak-system/blob/main/Screenshot%20From%202025-11-26%2010-36-42.png)|
| Prediction Result  | (https://github.com/Akash-Tiwari-alfa/ai-outbreak-system/blob/main/Screenshot%20From%202025-11-26%2010-36-55.png)|
| Overview           | (https://github.com/Akash-Tiwari-alfa/ai-outbreak-system/blob/main/Screenshot%20From%202025-11-26%2010-36-24.png)|
Create folder:

```
mkdir screenshots
```

---

## 🧠 **How the ML Model Works**

* Input features:

  * Fever cases
  * Cough cases
  * Diarrhea cases
  * Population
* Output:

  * Outbreak probability (0–1)
  * Risk Level:

    * < 0.3 → Low
    * 0.3–0.7 → Medium
    * > 0.7 → High

Model stored using:

```python
joblib.dump(model, 'models/outbreak_model.pkl')
```

---

## 🔮 **Future Enhancements**

* Live AQI integration (OpenWeather, IQAir API)
* Real-time hospital API (Google Places / Government data)
* SMS/Email early warning alerts
* Heatmap visualization per region
* Docker containerization
* Multi-admin dashboard
* RBAC (Role-Based Access Control)

---

## 🧑‍💻 **Author**

**Akash Tiwari**
GitHub: [Akash-Tiwari-alfa](https://github.com/Akash-Tiwari-alfa)

---
