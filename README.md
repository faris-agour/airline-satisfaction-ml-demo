# Airline Passenger Satisfaction - ML Project

This project predicts passenger satisfaction (`satisfied` vs `neutral or dissatisfied`) using classical ML models.

## Project Files
- `DT_TASK.ipynb`: presentation-ready notebook (EDA, preprocessing, outlier handling, modeling, tuning, evaluation)
- `app.py`: Streamlit deployment app
- `rebuild_artifact.py`: rebuilds the final model artifact from data
- `best_model.joblib`: saved model + preprocessing metadata
- `requirements.txt`: Python dependencies

## Features Used in App
- Gender
- Customer Type
- Age
- Type of Travel
- Class
- Flight Distance
- Inflight wifi service
- Departure/Arrival time convenient
- Ease of Online booking
- Gate location
- Food and drink
- Online boarding
- Seat comfort
- Inflight entertainment
- On-board service
- Leg room service
- Baggage handling
- Checkin service
- Inflight service
- Cleanliness

## Outlier Strategy
Outliers were identified mainly in distance and delay fields and handled using **IQR capping** to reduce extreme-value impact (especially for SVM) while keeping rows.

## Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) rebuild artifact:
   ```bash
   python rebuild_artifact.py
   ```
3. Run Streamlit app:
   ```bash
   streamlit run app.py
   ```
