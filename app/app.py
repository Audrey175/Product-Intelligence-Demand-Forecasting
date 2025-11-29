
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

st.title("Product Intelligence - Demand Forecasting Dashboard")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("../data/raw/train.csv", parse_dates=['date'])
    return df

df = load_data()

# ---------- Select Store & Item ----------
store_list = df['store'].unique()
store_id = st.selectbox("Select Store", store_list)

item_list = df[df['store']==store_id]['item'].unique()
item_id = st.selectbox("Select Item", item_list)

data = df[(df['store']==store_id) & (df['item']==item_id)].sort_values('date').copy()

# ---------- Create Features ----------
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['weekday'] = data['date'].dt.weekday

# Lag features
data['lag_1'] = data['sales'].shift(1)
data['lag_7'] = data['sales'].shift(7)
data['rolling_7'] = data['sales'].shift(1).rolling(7).mean()

data = data.dropna()

# ---------- Train RandomForest ----------
X = data[['lag_1','lag_7','rolling_7','day','month','weekday']]
y = data['sales']

# Train-test split (last 20% for validation)
cutoff = int(len(data)*0.8)
X_train = X.iloc[:cutoff]
y_train = y.iloc[:cutoff]
X_val = X.iloc[cutoff:]
y_val = y.iloc[cutoff:]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

st.subheader("RandomForest Prediction - Actual vs Predicted")
fig1, ax1 = plt.subplots(figsize=(12,4))
ax1.plot(data['date'].iloc[cutoff:], y_val, label='Actual')
ax1.plot(data['date'].iloc[cutoff:], y_pred, label='Predicted')
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")
ax1.legend()
st.pyplot(fig1)

# ---------- Seasonality Plots ----------
st.subheader("Monthly Seasonality")
monthly = data.groupby('month')['sales'].mean().reset_index()
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.barplot(x='month', y='sales', data=monthly, palette='Blues_d', ax=ax2)
ax2.set_xlabel("Month")
ax2.set_ylabel("Average Sales")
st.pyplot(fig2)

st.subheader("Weekly Seasonality")
weekly = data.groupby('weekday')['sales'].mean().reset_index()
fig3, ax3 = plt.subplots(figsize=(8,4))
sns.barplot(x='weekday', y='sales', data=weekly, palette='Oranges_d', ax=ax3)
ax3.set_xlabel("Weekday (0=Monday)")
ax3.set_ylabel("Average Sales")
st.pyplot(fig3)

# ---------- Prophet Forecast ----------
st.subheader("Prophet Forecast Next 30 Days")
prophet_df = data[['date','sales']].rename(columns={'date':'ds','sales':'y'})
m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.fit(prophet_df)

future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

fig4 = m.plot(forecast)
st.pyplot(fig4)

# ---------- Anomaly Detection ----------
st.subheader("Anomaly Detection (IQR Method)")
Q1 = data['sales'].quantile(0.25)
Q3 = data['sales'].quantile(0.75)
IQR = Q3 - Q1
data['anomaly'] = ((data['sales'] < (Q1 - 1.5*IQR)) | (data['sales'] > (Q3 + 1.5*IQR)))

fig5, ax5 = plt.subplots(figsize=(12,4))
ax5.plot(data['date'], data['sales'], label='Sales')
ax5.scatter(data['date'][data['anomaly']], data['sales'][data['anomaly']], color='red', label='Anomaly')
ax5.set_xlabel("Date")
ax5.set_ylabel("Sales")
ax5.legend()
st.pyplot(fig5)
