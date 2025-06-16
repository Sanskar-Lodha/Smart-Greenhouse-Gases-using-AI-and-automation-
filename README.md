# 🌿 Smart Greenhouse Gases Using AI and Automation

This project integrates AI-based weather forecasting with intelligent crop recommendation to optimize greenhouse efficiency and support sustainable agriculture. By predicting key weather parameters and recommending suitable crops, this system can help automate decisions inside a smart greenhouse.

> 📄 The accompanying **research paper** and **datasets** are also included in this repository for reference and reproducibility.

---

## 🚀 Features

- 🔮 **Next-Day Weather Forecasting**
  - Predicts minimum/maximum temperature, rainfall, and snow using Ridge Regression.

- 🧠 **Feature Engineering**
  - Rolling means, percentage changes, and seasonal trends (monthly/daily averages) enhance model accuracy.

- 🌾 **Crop Recommendation System**
  - Suggests ideal crop based on forecasted weather using a Random Forest classifier trained on agricultural data.

- 📉 **Performance Evaluation**
  - Includes robust backtesting with Mean Absolute Error (MAE) metrics.

---

## 🧠 Technologies Used

- Python 3
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook (optional for exploration)

---

## 📊 Sample Output
🌾 Forecast for tomorrow:
- Temperature (min): 40.45 °F
- Temperature (max): 56.95 °F
- Rainfall: 0.03 mm
- Humidity (est.): 48.70 %
- Recommended Crop to Grow: orange


  
  ---
  
## 📌 Requirements

Install dependencies using pip:

```bash
pip install pandas scikit-learn matplotlib
```
---
## 📬 Future Enhancements
- Incorporate greenhouse sensor feedback (CO₂, humidity, soil moisture)

- Deploy the model via a Flask or FastAPI backend

- Add a dashboard interface using Streamlit or Dash

- Integrate with IoT systems for real-time control

  



