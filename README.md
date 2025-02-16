# 🌦️ FastAPI Weather Forecaster

A **Deep Learning-based Weather Forecaster** using **LSTM**, built with **FastAPI** and deployed on **Render**. 🚀

---

## 📌 Features
✅ Predicts rainfall probability based on weather conditions  
✅ Uses **LSTM (Long Short-Term Memory) Neural Network**  
✅ **FastAPI** backend with Jinja2 templating  
✅ Scalable **deployment on Render**  
✅ Accepts both **web form input** & **API requests**  

---

## 🛠️ Tech Stack
- **Backend:** FastAPI, Jinja2
- **Machine Learning:** TensorFlow (LSTM Model)
- **Data Processing:** NumPy, joblib
- **Deployment:** Render
- **Frontend:** HTML, CSS (Optional)

---

## 🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/Weather-Forecaster.git
cd Weather-Forecaster

2️⃣ Create a Virtual Environment (Optional)
python -m venv venv
On Windows to activate:
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the App Locally
uvicorn main:app --reload
🚀 Now, open http://127.0.0.1:8000 in your browser.

🌐 API Endpoints
1️⃣ Web Interface
Open: http://127.0.0.1:8000/
Enter weather conditions → Get prediction
2️⃣ API Endpoint (POST Request)
URL: /predict/



