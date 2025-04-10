import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#Loading Data
transactions = pd.read_excel(r"C:\Users\anuja\OneDrive\Desktop\NexGen\DetergentProject\TransactionClean.xlsx")
products = pd.read_excel(r"C:\Users\anuja\OneDrive\Desktop\NexGen\DetergentProject\ProductClean.xlsx")
stores = pd.read_excel(r"C:\Users\anuja\OneDrive\Desktop\NexGen\DetergentProject\Store.xlsx")

st.title("DETERGENT SALES ANALYSIS")

st.subheader("Report and Insights for Declining Sales (11% decrease from 2018 to 2019)")
st.write("This report analyzes the reasons behind the decline in sales and provides insights into inventory management and optimization.")

#Loading Images
image = Image.open("profitloss.jpeg")
st.image(image, caption="Power BI Dashboard", use_container_width=True)

image1 = Image.open("insights.jpeg")
st.image(image1, caption="Power BI Insights", use_container_width=True)

st.subheader("SOLUTIONS")
st.write("1. Increase inventory levels for high-selling products while reducing stock for low-performing ones.")
st.write("2. Offer discounts on slow-moving items to boost revenue.")
st.write("3. Implement targeted loyalty programs to retain Elite and Good customers.")
st.write("4. Enhance data tracking to analyze the Unknown Customer segment for better sales attribution.")
st.write("5. Adjust inventory to align with the increasing demand for budget-friendly products.")

st.markdown("---")

st.subheader("Data Preview")
st.write("### Transactions Data", transactions.head())
st.write("### Products Data", products.head())
st.write("### Stores Data", stores.head())

# Data Cleaning and Merging
transactions.dropna(inplace=True)
products.dropna(inplace=True)
stores.dropna(inplace=True)

transactions['TXN_DT'] = pd.to_datetime(transactions['TXN_DT'])
transactions['day_of_week'] = transactions['TXN_DT'].dt.dayofweek  # Monday=0, Sunday=6

# Compute average sales per day
avg_sales_by_day = transactions.groupby("day_of_week")["ITEM_QTY"].mean()

data = transactions.merge(products, on='UPC_ID', how='left').merge(stores, on='STORE_ID', how='left')

st.subheader("Train Machine Learning Model to Predict High Sales Days")

X = data[['day_of_week', 'ITEM_QTY', 'FACTS_2019', 'FACTS_2018', 'TRUPRICE_2019', 'TRUPRICE_2018']]
y = data['ITEM_QTY']  

X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
score = model.score(X_test, y_test)

st.write(f"Model Accuracy: {score * 100:.2f}%")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Predict which day has higher sales
st.subheader("Predict High Sales Day")
day_of_week = st.selectbox("Select a Day (0=Monday, 6=Sunday)", [0, 1, 2, 3, 4, 5, 6])

if st.button("Predict Sales for Selected Day"):
    input_data = np.array([[day_of_week, 50, 10, 15, 5, 8]])  # Sample input for prediction
    prediction = model.predict(input_data)
    st.write(f"Predicted Sales for Day {day_of_week}: {prediction[0]:.2f}")
    
    # Compare prediction with actual average sales for that day
    if prediction[0] > avg_sales_by_day[day_of_week]:  
        st.success(f"Sales on this day ({day_of_week}) are higher than usual!")
    else:
        st.warning(f"Sales on this day ({day_of_week}) are lower than usual.")