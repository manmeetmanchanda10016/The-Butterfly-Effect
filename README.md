# 🦋 The Butterfly Effect
### Sales Forecasting Using Machine Learning

The Butterfly Effect is a machine learning project that predicts retail sales using historical sales patterns, store information, product categories, promotional activities, and time-based features.

Built using **LightGBM and Streamlit**, this project provides an interactive dashboard for sales prediction and model performance analysis.

## ✨ Features

- 📈 Retail sales prediction using LightGBM
- 🏪 Store and product family selection
- 🛍️ Promotional activity analysis
- 📅 Date-based feature selection
- 🔄 Historical sales features using lag values
- 📊 Actual vs. predicted sales visualization
- 🎯 Feature importance visualization
- 📉 Model evaluation metrics
- 🎨 Interactive dashboard with an Ice Blue and White theme

## 🧠 Machine Learning Model

**Algorithm:** LightGBM Regressor

**Input Features:**
- `store_nbr`
- `family`
- `onpromotion`
- `day_of_week`
- `month`
- `year`
- `day_of_month`
- `lag_1`
- `lag_7`

The trained model is saved as a serialized pipeline and used by the Streamlit application to generate predictions.

## 📊 Model Performance

| Metric | Score |
|---|---:|
| MAE | 82.28 |
| RMSE | 374.28 |
| R² Score | 0.9245 |

*Note: These are the recorded evaluation results. Actual performance depends on the dataset and evaluation methodology.*

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Plotly
- Streamlit
- Joblib

## 📁 Project Structure

```text
The-Butterfly-Effect/
├── app.py
├── requirements.txt
├── butterfly_effect_model.pkl
├── evaluation_results.csv
└── README.md
```

`evaluation_results.csv` is optional and can be used to display actual-versus-predicted sales when it contains the required columns.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/manmeetmanchanda10016/The-Butterfly-Effect.git
cd The-Butterfly-Effect
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## 📈 How It Works

1. Select a store and product family.
2. Enter promotional and date-related information.
3. Provide historical sales values for the previous day and seven days earlier.
4. Click **Predict Sales**.
5. Explore the predicted sales, evaluation metrics, and feature importance.

## 🎯 Project Objective

The objective is to explore how machine learning can support retail demand forecasting, sales analysis, and informed inventory planning.

## 👨‍💻 Author

**Manmeet Singh Manchanda**

- GitHub: [@manmeetmanchanda10016](https://github.com/manmeetmanchanda10016)
- Repository: [The Butterfly Effect](https://github.com/manmeetmanchanda10016/The-Butterfly-Effect)

---

*Built with Python, Machine Learning, and Streamlit. 🦋*
