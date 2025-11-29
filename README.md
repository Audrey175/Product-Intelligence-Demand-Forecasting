# Product-Intelligence-Demand-Forecasting
App link: 
## **Project Overview**

This project is a **mini dashboard for demand forecasting** that helps visualize historical sales data, identify seasonal patterns, detect anomalies, and predict future demand for products at different stores.

It demonstrates a **full data science workflow** including:

* Exploratory Data Analysis (EDA)
* Feature engineering with lag features and rolling averages
* Time series analysis and seasonality visualization
* Machine learning modeling (RandomForest)
* Forecasting using **Prophet**
* Interactive dashboard with **Streamlit**

This project aligns with data science tasks such as those described for **Data Scientist Intern roles**, showcasing skills in Python, machine learning, data visualization, and reporting.

---

## **Features**

1. **Interactive Store & Product Selection**

   * Users can select a store and item to view sales trends.

2. **Sales Time Series Visualization**

   * Plot of actual historical sales data.

3. **Seasonality Analysis**

   * Monthly and weekly sales trends.
   * Heatmaps and bar plots highlight seasonal patterns.

4. **RandomForest Prediction**

   * Forecast short-term sales using lag and rolling features.
   * Visual comparison of actual vs predicted values.

5. **Prophet Forecast**

   * 30-day forecast using Prophet for time series prediction.
   * Shows trend, weekly and yearly seasonality.

6. **Anomaly Detection**

   * Identify outlier sales points using IQR method.
   * Highlights unusual sales spikes or drops.

---

## **Dataset**

* `train.csv` – historical sales data with columns:

  * `date` – date of sales
  * `store` – store ID
  * `item` – product ID
  * `sales` – quantity sold

> **Note:** `test.csv` is used for future predictions and does not include `sales`.

**Source:** [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/store-item-demand-forecasting)

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/Product-Intelligence-Demand-Forecasting.git
cd Product-Intelligence-Demand-Forecasting/app
```

2. Install required packages:

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn prophet
```

---

## **Run the Dashboard**

```bash
streamlit run app.py
```

* Open the URL shown in the terminal (`http://localhost:8501`) to access the interactive dashboard.
* Select **store** and **item** to view historical sales, forecasts, seasonality, and anomalies.

---

## **Project Structure**

```
project-product-intel/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
│  ├─ 01_data_overview.ipynb
│  └─ 02_modeling_and_evaluation.ipynb
├─ models/
│  └─ rf_model.joblib
├─ app/
│  └─ app.py
├─ reports/
│  └─ anomalies.csv
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## **Technologies & Libraries**

* **Python 3.12**
* **Pandas, NumPy** – data manipulation
* **Matplotlib, Seaborn** – visualization
* **Scikit-learn** – machine learning (RandomForest)
* **Prophet** – time series forecasting
* **Streamlit** – interactive dashboard

---

## **Key Skills Demonstrated**

* Data cleaning & EDA
* Feature engineering (lag features, rolling averages)
* Time series analysis & forecasting
* Machine learning modeling & evaluation
* Data visualization & storytelling
* Interactive dashboard development

---

## **Next Steps / Optional Enhancements**

* Add **forecast horizon selection** (7, 30, 90 days)
* Aggregate forecasts for all stores & items
* Deploy the dashboard online (Streamlit Cloud or Heroku)
* Add more advanced anomaly detection (e.g., Z-score, Isolation Forest)