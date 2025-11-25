from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
import joblib
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# ---- BASIC CONFIG ----
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///outbreak.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "change-this-to-a-strong-secret"  # for sessions & login

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"  # redirect here if not logged in

MODEL_PATH = "models/outbreak_model.pkl"


# ---- DB MODELS ----
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    predictions = db.relationship("Prediction", backref="user", lazy=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    region = db.Column(db.String(100))
    fever_cases = db.Column(db.Integer)
    cough_cases = db.Column(db.Integer)
    diarrhea_cases = db.Column(db.Integer)
    region_population = db.Column(db.Integer)
    prediction = db.Column(db.Integer)  # 0/1
    probability = db.Column(db.Float)   # percent 0–100
    risk_level = db.Column(db.String(20))
    aqi_value = db.Column(db.Integer, nullable=True)
    aqi_category = db.Column(db.String(50), nullable=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


# ---- LOGIN MANAGER ----
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---- UTILS ----
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def compute_risk_level(probability: float) -> str:
    """Convert probability (0–1) to human-friendly risk level."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def get_region_aqi(region: str):
    """
    Dummy AQI provider.
    You can later replace this with a real API call (OpenWeather, IQAir, etc.)
    For now, we map some regions, else give a moderate default.
    """
    region_key = (region or "unknown").strip().lower()

    sample_data = {
        "surat": 130,
        "ahmedabad": 160,
        "mumbai": 140,
        "delhi": 220,
        "pune": 110,
    }

    aqi = sample_data.get(region_key, 90)  # default moderate
    # Basic categorization (Indian AQI-ish)
    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Satisfactory"
    elif aqi <= 200:
        category = "Moderate"
    elif aqi <= 300:
        category = "Poor"
    elif aqi <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    return aqi, category


def get_hospitals_for_region(region: str):
    """
    Static demo list of hospitals.
    You can move this to DB or external API later.
    """
    region_key = (region or "unknown").strip().lower()

    hospitals_map = {
        "surat": [
            {
                "name": "New Civil Hospital, Surat",
                "phone": "+91-261-XYZ-0001",
                "address": "Ring Road, Surat",
            },
            {
                "name": "SMIMER Hospital",
                "phone": "+91-261-XYZ-0002",
                "address": "Dumas Road, Surat",
            },
        ],
        "ahmedabad": [
            {
                "name": "Civil Hospital, Ahmedabad",
                "phone": "+91-79-XYZ-0001",
                "address": "Asarwa, Ahmedabad",
            },
            {
                "name": "Sardar Vallabhbhai Patel Hospital",
                "phone": "+91-79-XYZ-0002",
                "address": "Ellisbridge, Ahmedabad",
            },
        ],
    }

    # default: empty list
    return hospitals_map.get(region_key, [])


def generate_recommendations(risk_level: str, region: str, features: dict,
                             aqi_value: int, aqi_category: str, hospital_count: int) -> list:
    """
    AI-style suggestions based on:
    - outbreak risk,
    - dominant symptoms,
    - air quality,
    - hospital availability.
    """
    fever = features["fever_cases"]
    cough = features["cough_cases"]
    diarrhea = features["diarrhea_cases"]
    pop = max(features["region_population"], 1)

    # Cases per 10,000
    fever_rate = (fever / pop) * 10000
    cough_rate = (cough / pop) * 10000
    diarrhea_rate = (diarrhea / pop) * 10000

    rates = {
        "Fever": fever_rate,
        "Cough": cough_rate,
        "Diarrhea": diarrhea_rate,
    }
    top_symptom = max(rates, key=rates.get)

    suggestions = []

    # ---- outbreak risk based ----
    if risk_level == "High":
        suggestions.append(
            f"Immediately alert public health authorities for {region} and activate outbreak response protocols."
        )
        suggestions.append(
            "Escalate testing capacity and prepare isolation facilities for high-risk patients."
        )
    elif risk_level == "Medium":
        suggestions.append(
            f"Enable enhanced surveillance in {region} with daily case tracking at ward/zone level."
        )
        suggestions.append(
            "Review hospital readiness (beds, oxygen, ICU) in anticipation of a possible surge."
        )
    else:
        suggestions.append(
            f"Maintain routine surveillance in {region} while focusing on early case reporting."
        )

    # ---- symptom-based ----
    if top_symptom == "Fever":
        suggestions.append(
            "Investigate clusters of fever in schools, hostels and workplaces; run fever screening camps."
        )
    elif top_symptom == "Cough":
        suggestions.append(
            "Strengthen respiratory hygiene measures: masks in crowded places, cough etiquette, ventilation checks."
        )
    else:  # Diarrhea
        suggestions.append(
            "Investigate water and food safety, test local water sources, and promote use of safe drinking water."
        )

    # ---- AQI-based ----
    if aqi_value is not None:
        if aqi_value > 200:
            suggestions.append(
                f"Air quality in {region} is {aqi_category} (AQI {aqi_value}). Advise masks outdoors and limit exposure for vulnerable groups."
            )
        elif aqi_value > 100:
            suggestions.append(
                f"Air quality in {region} is {aqi_category} (AQI {aqi_value}). Consider public advisories for people with asthma and heart disease."
            )
        else:
            suggestions.append(
                f"Air quality in {region} is {aqi_category} (AQI {aqi_value}). Maintain current environmental controls."
            )

    # ---- hospital-based ----
    if hospital_count == 0:
        suggestions.append(
            "No mapped hospitals found for this region. Maintain an updated health facility directory to improve response time."
        )
    elif hospital_count <= 2 and risk_level in ["Medium", "High"]:
        suggestions.append(
            f"Limited hospital coverage ({hospital_count} known facilities). Prepare referral pathways to nearby districts."
        )
    else:
        suggestions.append(
            "Coordinate with listed hospitals to ensure triage, isolation, and referral protocols are clearly defined."
        )

    # ---- data quality ----
    suggestions.append(
        "Standardize daily data reporting (same time, same fields) to improve model reliability and early warning accuracy."
    )

    return suggestions


# ---- ROUTES: AUTH ----
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "danger")
            return redirect(url_for("register"))

        existing = User.query.filter_by(email=email).first()
        if existing:
            flash("Email is already registered. Please login.", "warning")
            return redirect(url_for("login"))

        user = User(name=name or email, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ---- ROUTES: MAIN ----
@app.route("/")
def index():
    """Overview page. If logged in: show user-specific stats."""
    if not current_user.is_authenticated:
        return render_template("landing.html")

    base_query = Prediction.query.filter_by(user_id=current_user.id)

    total_predictions = base_query.count()
    high_risk_count = base_query.filter_by(risk_level="High").count()
    region_count = (
        db.session.query(Prediction.region)
        .filter_by(user_id=current_user.id)
        .distinct()
        .count()
    )

    recent_predictions = (
        base_query.order_by(Prediction.id.desc()).limit(5).all()
    )

    return render_template(
        "index.html",
        total_predictions=total_predictions,
        high_risk_count=high_risk_count,
        region_count=region_count,
        recent_predictions=recent_predictions,
    )


@app.route("/dashboard")
@login_required
def dashboard():
    """Detailed dashboard: region-wise risk stats for current user only."""
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()

    region_stats = defaultdict(lambda: {"Low": 0, "Medium": 0, "High": 0})
    for p in predictions:
        region = p.region or "Unknown"
        region_stats[region][p.risk_level] += 1

    labels = list(region_stats.keys())
    low_counts = [region_stats[r]["Low"] for r in labels]
    medium_counts = [region_stats[r]["Medium"] for r in labels]
    high_counts = [region_stats[r]["High"] for r in labels]

    region_summary = []
    for r in labels:
        total = (
            region_stats[r]["Low"]
            + region_stats[r]["Medium"]
            + region_stats[r]["High"]
        )
        high = region_stats[r]["High"]
        high_percent = (high / total) * 100 if total > 0 else 0.0
        region_summary.append(
            {
                "region": r,
                "total_predictions": total,
                "high_count": high,
                "high_percent": round(high_percent, 1),
            }
        )

    region_summary.sort(key=lambda x: (-x["high_percent"], -x["total_predictions"]))

    recent_predictions = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.id.desc())
        .limit(20)
        .all()
    )

    return render_template(
        "dashboard.html",
        labels=labels,
        low_counts=low_counts,
        medium_counts=medium_counts,
        high_counts=high_counts,
        recent_predictions=recent_predictions,
        region_summary=region_summary,
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """Web form to predict outbreak risk."""
    model = load_model()
    if model is None:
        return "Model not trained yet. Please train model first.", 500

    prediction_val = None
    probability = None
    risk_level = None
    suggestions = []
    aqi_value = None
    aqi_category = None
    hospitals = []
    region_for_display = None

    if request.method == "POST":
        try:
            region = request.form.get("region", "").strip() or "Unknown"
            region_for_display = region

            fever_cases = float(request.form["fever_cases"])
            cough_cases = float(request.form["cough_cases"])
            diarrhea_cases = float(request.form["diarrhea_cases"])
            region_population = float(request.form["region_population"])

            features_array = np.array(
                [[fever_cases, cough_cases, diarrhea_cases, region_population]]
            )
            proba = model.predict_proba(features_array)[0][1]
            prediction_val = int(proba > 0.5)
            probability = round(float(proba) * 100, 2)
            risk_level = compute_risk_level(proba)

            # AQI + hospitals
            aqi_value, aqi_category = get_region_aqi(region)
            hospitals = get_hospitals_for_region(region)

            suggestions = generate_recommendations(
                risk_level,
                region,
                {
                    "fever_cases": fever_cases,
                    "cough_cases": cough_cases,
                    "diarrhea_cases": diarrhea_cases,
                    "region_population": region_population,
                },
                aqi_value,
                aqi_category,
                len(hospitals),
            )

            # Save to DB
            pred_row = Prediction(
                user_id=current_user.id,
                region=region,
                fever_cases=int(fever_cases),
                cough_cases=int(cough_cases),
                diarrhea_cases=int(diarrhea_cases),
                region_population=int(region_population),
                prediction=prediction_val,
                probability=probability,
                risk_level=risk_level,
                aqi_value=aqi_value,
                aqi_category=aqi_category,
            )
            db.session.add(pred_row)
            db.session.commit()

        except Exception as e:
            return f"Error in prediction: {e}", 400

    return render_template(
        "predict.html",
        prediction=prediction_val,
        probability=probability,
        risk_level=risk_level,
        suggestions=suggestions,
        aqi_value=aqi_value,
        aqi_category=aqi_category,
        hospitals=hospitals,
        region=region_for_display,
    )


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """
    JSON API for prediction for logged-in user.
    """
    model = load_model()
    if model is None:
        return jsonify({"error": "Model not trained"}), 500

    data = request.get_json() or {}
    try:
        region = data.get("region", "Unknown")
        fever_cases = float(data["fever_cases"])
        cough_cases = float(data["cough_cases"])
        diarrhea_cases = float(data["diarrhea_cases"])
        region_population = float(data["region_population"])
    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except ValueError:
        return jsonify({"error": "Invalid numeric values"}), 400

    features_array = np.array(
        [[fever_cases, cough_cases, diarrhea_cases, region_population]]
    )
    proba = model.predict_proba(features_array)[0][1]
    prediction_val = int(proba > 0.5)
    risk_level = compute_risk_level(proba)
    probability_percent = round(float(proba) * 100, 2)

    aqi_value, aqi_category = get_region_aqi(region)
    hospitals = get_hospitals_for_region(region)

    suggestions = generate_recommendations(
        risk_level,
        region,
        {
            "fever_cases": fever_cases,
            "cough_cases": cough_cases,
            "diarrhea_cases": diarrhea_cases,
            "region_population": region_population,
        },
        aqi_value,
        aqi_category,
        len(hospitals),
    )

    pred_row = Prediction(
        user_id=current_user.id,
        region=region,
        fever_cases=int(fever_cases),
        cough_cases=int(cough_cases),
        diarrhea_cases=int(diarrhea_cases),
        region_population=int(region_population),
        prediction=prediction_val,
        probability=probability_percent,
        risk_level=risk_level,
        aqi_value=aqi_value,
        aqi_category=aqi_category,
    )
    db.session.add(pred_row)
    db.session.commit()

    return jsonify(
        {
            "prediction": prediction_val,
            "probability": probability_percent,
            "risk_level": risk_level,
            "aqi": {
                "value": aqi_value,
                "category": aqi_category,
            },
            "hospitals": hospitals,
            "suggestions": suggestions,
        }
    )


# ---- MAIN ----
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
