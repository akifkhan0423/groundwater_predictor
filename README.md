# 💧 Groundwater Depletion Risk Predictor
**AIML Mini Project | PES University | Akifkhan Badagi | PES1PG25CA017**

> Predicts groundwater depletion risk levels (Low / Medium / High) for agricultural zones using supervised ML models.

---

## 📁 Project Structure

```
groundwater_predictor/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🚀 Step-by-Step Setup in VS Code

### Step 1 — Install Python
1. Go to https://www.python.org/downloads/
2. Download **Python 3.10 or 3.11**
3. During install, ✅ check **"Add Python to PATH"**
4. Verify: open a terminal and run `python --version`

---

### Step 2 — Open Project in VS Code
1. Open **VS Code**
2. Go to **File → Open Folder**
3. Select the `groundwater_predictor` folder
4. VS Code will open with all project files visible in the Explorer panel

---

### Step 3 — Open the Integrated Terminal
- Press **Ctrl + `** (backtick) to open the VS Code terminal
- Or go to **Terminal → New Terminal**

---

### Step 4 — Create a Virtual Environment
Run these commands in the terminal:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

### Step 5 — Install Dependencies
```bash
pip install -r requirements.txt
```
This installs: streamlit, pandas, numpy, plotly, scikit-learn, joblib, xgboost

> ⏳ This may take 2–3 minutes depending on your internet speed.

---

### Step 6 — Run the App
```bash
streamlit run app.py
```

The app will automatically open in your browser at:
👉 **http://localhost:8501**

---

## 🎮 How to Use the App

| Tab | What it does |
|-----|-------------|
| 🎯 Prediction | Enter parameters → get risk level + probabilities |
| 📊 Model Performance | Compare accuracy of all 5 ML models |
| 🗂️ Dataset | Browse and filter the generated dataset |
| 📈 EDA | Visual exploration of patterns and correlations |

### Sidebar Controls:
- **Select Model** — choose from Random Forest, Gradient Boosting, Decision Tree, Logistic Regression, KNN
- **Rainfall (mm)** — annual rainfall in the district
- **Groundwater Depth (m)** — depth of water table below ground
- **Temperature (°C)** — average temperature
- **Soil Moisture** — fraction (0.0–1.0)
- **Irrigation Type** — Borewell / Canal / Drip / Sprinkler / Tube well
- **Crop Type** — Rice / Wheat / Sugarcane / Cotton / Bajra / Maize / Soybean

Click **⚡ Predict Risk** to get results.

---

## 🤖 ML Models Used

| Model | Type |
|-------|------|
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |
| Decision Tree | Tree-based |
| Logistic Regression | Linear |
| K-Nearest Neighbors | Instance-based |

**Target Classes:** Low / Medium / High (derived from groundwater depth)

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit: command not found` | Make sure venv is activated |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Port already in use | Run `streamlit run app.py --server.port 8502` |
| Browser doesn't open | Manually go to http://localhost:8501 |

---

## 🌍 SDG Alignment
- **SDG 2 – Zero Hunger**: Sustainable water use → food security
- **SDG 15 – Life on Land**: Protecting aquifers → ecosystem health

---

*Built with Streamlit + scikit-learn + Plotly*
