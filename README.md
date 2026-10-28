# 🦋 The Butterfly Effect — Retail Sales Forecasting

**Predict retail sales using Machine Learning and LightGBM.**

🚀 **Live Demo:** [Try The Butterfly Effect](https://the-butterfly-effect-ngfrju5rjaizzniva9vze8.streamlit.app/)

📂 **GitHub Repository:** [The Butterfly Effect](https://github.com/manmeetmanchanda10016/The-Butterfly-Effect)

---

## 📌 Project Overview

The Butterfly Effect is an interactive retail sales forecasting application built using **Python, LightGBM, and Streamlit**. It predicts retail sales based on factors such as store number, product family, promotions, date information, and historical sales.

The application provides an intuitive dashboard where users can adjust input parameters, generate sales predictions, and explore model performance.

## ✨ Key Features

- 📈 **Sales Forecasting:** Predict retail sales using a trained LightGBM model.
- 🎛️ **Interactive Dashboard:** Customize store, product, promotion, date, and historical sales inputs.
- 📊 **Model Evaluation:** View MAE, RMSE, and R² metrics.
- 📉 **Sales Visualization:** Explore actual versus predicted sales when evaluation data is available.
- 🔍 **Feature Importance:** Understand which input features contribute to model predictions.
- 🎨 **User-Friendly Interface:** Clean, interactive dashboard built with Streamlit.

## 🤖 Machine Learning Model

The project uses **LightGBM**, a gradient boosting framework designed for efficient and accurate predictions.

### Input Features

- `store_nbr` — Store identifier
- `family` — Product family
- `onpromotion` — Number of items on promotion
- `day_of_week` — Day of the week
- `month` — Month of the year
- `year` — Year
- `day_of_month` — Day of the month
- `lag_1` — Previous day's sales
- `lag_7` — Sales from seven days earlier

## 📊 Model Performance

| Metric | Score |
|---|---:|
| MAE | 82.2774 |
| RMSE | 374.2801 |
| R² Score | 0.9245 |

These are the previously recorded evaluation results for the trained model. Actual performance may vary depending on the evaluation dataset.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning:** LightGBM, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Dashboard:** Streamlit
- **Visualization:** Plotly
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
The-Butterfly-Effect/
│
├── app.py
├── butterfly_effect_model.pkl
├── requirements.txt
├── README.md
└── evaluation_results.csv   # Optional
```

The optional `evaluation_results.csv` file can be used to visualize actual versus predicted sales when it contains the required columns: `Actual Sales` and `Predicted Sales`.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/manmeetmanchanda10016/The-Butterfly-Effect.git
```

### 2. Navigate to the project folder

```bash
cd The-Butterfly-Effect
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🎯 Project Objective

The goal of this project is to demonstrate how machine learning can be used for retail sales forecasting through an interactive dashboard. It combines model development, data processing, performance evaluation, and deployment into one application.

## 👨‍💻 Author

**Manmeet Singh Manchanda**

Computer Science and Engineering Student

- GitHub: [@manmeetmanchanda10016](https://github.com/manmeetmanchanda10016)
- Project Repository: [The Butterfly Effect](https://github.com/manmeetmanchanda10016/The-Butterfly-Effect)
- Live Application: [Launch Dashboard](https://the-butterfly-effect-ngfrju5rjaizzniva9vze8.streamlit.app/)

---

⭐ If you find this project interesting, feel free to explore the repository and try the live demo!

**Thank you for visiting The Butterfly Effect! 🦋**
