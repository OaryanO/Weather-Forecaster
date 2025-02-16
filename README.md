# 🌦️ Weather Forecaster - LSTM Based Prediction

This is a **weather forecasting application** built using **FastAPI** and an **LSTM model**. It predicts whether it will rain based on user-input weather conditions.

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Weather-Forecaster.git
cd Weather-Forecaster
```

### 2️⃣ Create a Virtual Environment (Optional)
```bash
python -m venv venv
```
#### **On Windows, activate with:**
```bash
venv\Scripts\activate
```
#### **On macOS/Linux, activate with:**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App Locally
```bash
uvicorn main:app --reload
```
🚀 Now, open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 🌐 API Endpoints

### 1️⃣ **Web Interface**
- Open: **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**
- Enter weather conditions → **Get Prediction**

### 2️⃣ **API Endpoint (POST Request)**
- **URL:** `/predict/`
- **Request Format (JSON):**
```json
{
  "temperature": 25,
  "humidity": 60,
  "wind_speed": 10,
  "pressure": 1013,
  "cloud_cover": 50
}
```
- **Response Format:**
```json
{
  "prediction": "☀️ No Rain Expected",
  "probability": "30.25%"
}
```

---

## 🛠️ Technologies Used
- **FastAPI** (Backend)
- **TensorFlow/Keras** (LSTM Model)
- **Scikit-Learn** (Data Scaling)
- **Jinja2** (HTML Templates)
- **Render** (Deployment)

---

## 📌 To-Do
- [ ] Add a front-end UI for better user experience
- [ ] Improve model accuracy with additional features
- [ ] Optimize API response times

---

## 📜 License
This project is built by **Aryan Singh**. Feel free to use and improve it! 🚀

---

💡 *Contributions & Feedback are Welcome!* ✨
