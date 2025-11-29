import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

st.title("Mini Dashboard - Sales Forecast")

# Load train data
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", parse_dates=['date'])
    return df

df = load_data()

# Select store & item
store_list = df['store'].unique()
store_id = st.selectbox("Select Store", store_list)

item_list = df[df['store']==store_id]['item'].unique()
item_id = st.selectbox("Select Item", item_list)

data = df[(df['store']==store_id) & (df['item']==item_id)].sort_values('date')

st.subheader("Sales Time Series")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(data['date'], data['sales'], label='Actual Sales')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)

# Seasonality plots
st.subheader("Monthly Seasonality")
data['month'] = data['date'].dt.month
monthly = data.groupby('month')['sales'].mean().reset_index()
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.barplot(x='month', y='sales', data=monthly, palette='Blues_d', ax=ax2)
ax2.set_xlabel("Month")
ax2.set_ylabel("Average Sales")
st.pyplot(fig2)

st.subheader("Weekly Seasonality")
data['weekday'] = data['date'].dt.weekday
weekly = data.groupby('weekday')['sales'].mean().reset_index()
fig3, ax3 = plt.subplots(figsize=(8,4))
sns.barplot(x='weekday', y='sales', data=weekly, palette='Oranges_d', ax=ax3)
ax3.set_xlabel("Weekday (0=Monday)")
ax3.set_ylabel("Average Sales")
st.pyplot(fig3)

# Forecast with Prophet
st.subheader("Forecast Next 30 Days")
prophet_df = data[['date','sales']].rename(columns={'date':'ds','sales':'y'})
m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.fit(prophet_df)

future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

fig4 = m.plot(forecast)
st.pyplot(fig4)

# Anomaly detection (simple IQR)
st.subheader("Anomaly Detection")
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
