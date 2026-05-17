# ✈️ Flight Delay Prediction System

A full-stack machine learning application that predicts real-time flight delays using an XGBoost classification model and live API data. 

## 🚀 Features
- **Live Flight Tracking:** Fetches real-time flight data globally using the OpenSky Network (Free/Real-time) and AviationStack APIs.
- **Machine Learning Engine:** Powered by a pre-trained XGBoost Classifier trained on over 300,000 flight records.
- **Risk Assessment:** Classifies live flights into `🟢 Low`, `🟡 Medium`, and `🔴 High` delay risk categories.
- **Premium UI/UX:** Built with Streamlit but heavily customized with modern CSS, dark-theme gradients, metric cards, and responsive Plotly visualizations.

## 🧠 Model Architecture
The core model is an **XGBoost Classifier**.
- **Accuracy:** ~79.62%
- **ROC-AUC:** 0.85+
- **Features Used:** Airline, Origin Airport, Month, Day of Week, Departure Hour, IsWeekend, and Flight Distance.
- Categorical features are encoded using `LabelEncoder`.

## 🛠️ Tech Stack
- **Backend:** Python, Pandas, Scikit-Learn, XGBoost
- **Frontend:** Streamlit, Custom CSS/HTML
- **Visualizations:** Plotly Express / Plotly Graph Objects
- **APIs:** AviationStack, OpenSky Network

## 💻 How to Run Locally

1. **Activate the Virtual Environment:**
   ```bash
   venv\Scripts\activate
   ```

2. **Install Dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application:**
   *Note: Do NOT run this using `python app_Enhanced.py`.*
   ```bash
   streamlit run app_Enhanced.py
   ```

4. **View the Dashboard:**
   Open your browser and navigate to `http://localhost:8501`.

## 📂 Project Structure
- `app_Enhanced.py` - The main Streamlit web application.
- `implementation.ipynb` - Jupyter notebook containing data exploration, preprocessing, model training, and evaluation.
- `utils.py` - Helper functions for loading the model and encoding features safely.
- `style.css` & `.streamlit/config.toml` - Custom styling and theming for the premium UI.
- `models/` - Pickled machine learning models (`xgboost_model.pkl`, encoders).
- `data/` - Contains the preprocessed dataset used for training.