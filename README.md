# Predictive Maintenance Simulator  
A complete end-to-end project for generating IoT sensor data, training fault-detection models, and visualizing results through an interactive web dashboard.

---

## 🚀 Overview

This project simulates real-time machine sensor data and applies machine-learning techniques to predict potential equipment failures.  
It covers:

- Data simulation for multiple IoT devices  
- Feature engineering using time-window aggregation  
- Classification model for fault detection  
- Interactive dashboard for monitoring device health  
- Exportable predictions for analysis  

The project is structured and designed to reflect an industry-grade workflow.

---

## 🛠 Features  

### ✔ Synthetic IoT Data Generation  
- Vibration (x, y, z)  
- Temperature  
- Pressure  
- RPM  
- Current  
- Automatic injection of failure states (leak, electrical fault)  

### ✔ ML Pipeline  
- Feature extraction using rolling windows  
- RandomForest-based fault classifier  
- Model evaluation (accuracy, confusion matrix, classification report)  
- Saved trained model (`baseline.pkl`)  

### ✔ Streamlit Dashboard  
- Device-wise sensor visualization  
- Risk scoring for each time window  
- Highest-risk time window highlight  
- Predictions table with CSV download option  
- Clean white-blue theme with neat Plotly charts  

---

## 📂 Project Structure

```
predictive-maintenance-simulator/
│
├── data/
│   ├── dataset_forced_failures.csv
│   ├── sample_dataset.csv
│   ├── sample_dataset_generated.csv
│   └── ...
│
├── models/
│   └── baseline.pkl
│
├── src/
│   ├── simulate.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── dashboard.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📘 How It Works

### **1️⃣ Data Simulation**
Synthetic sensor readings are generated using controlled randomness and injected equipment failures.

Script:
```bash
python -m src.simulate
```

Outputs:
- `dataset_forced_failures.csv`  
- Contains 3 devices, ~21k rows, multiple failure events  

---

### **2️⃣ Feature Engineering**
Sensor data is aggregated into sliding windows (mean, std, max, min, slope).

Function:  
```python
make_window_features(df, window_sec=60, step_sec=30)
```

---

### **3️⃣ Model Training**
The classifier learns to identify:

- **0** = Normal  
- **1** = Electrical fault  
- **2** = Leak  

Train using:
```bash
python -m src.train
```

Model saved to:
```
models/baseline.pkl
```

---

### **4️⃣ Prediction**
Runs inference on windowed features.

```bash
python -m src.predict
```

Outputs CSV:
```
models/predictions_sample.csv
```

---

### **5️⃣ Dashboard**
Start the dashboard:

```bash
streamlit run src/dashboard.py
```

Features inside dashboard:

- Device selection  
- Sensor streams (Plotly interactive charts)  
- Highest-risk time windows  
- Downloadable predictions  
- Clean design for presentations & demos  

---

## 📈 Model Performance

Example results from training:

| Metric | Score |
|-------|--------|
| Accuracy | 0.97 |
| Precision (faults) | 0.93+ |
| Recall (faults) | 0.90+ |
| Supports 3 fault categories | ✔ |

Confusion Matrix, Classification Report, and Feature Importances are printed during training.

---

## 🧰 Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **XGBoost (optional)**
- **TensorFlow (optional)**
- **Matplotlib**, **Plotly**
- **Streamlit**
- **FastAPI (future expansion)**

---

## 🚀 Future Enhancements

- Real-time MQTT data ingestion  
- Multiclass regression for Time-To-Failure  
- LSTM or Transformer-based sequence models  
- Cloud deployment (AWS EC2 / Railway / Render)  
- API endpoint for live monitoring  

---

## 📄 License
MIT License  
Free to use for learning, research, or portfolio projects.

---

## 🙌 Acknowledgements
Built with industry-style practices: modular code, reproducible pipelines, and production-ready dashboards.
