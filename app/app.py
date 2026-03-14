import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Ecommerce Delivery Analytics",
    layout="wide"
)

st.title("Ecommerce Delivery Analytics Dashboard")

# -------------------------
# Load Data
# -------------------------

df = pd.read_csv("data/processed/cleaned_data.csv")

# Convert target variable
df["Reached.on.Time_Y.N"] = df["Reached.on.Time_Y.N"].map({1:"Delayed",0:"On Time"})

# -------------------------
# Sidebar Filters
# -------------------------

st.sidebar.header("Filters")

shipment_filter = st.sidebar.multiselect(
    "Shipment Mode",
    df["Mode_of_Shipment"].unique(),
    default=df["Mode_of_Shipment"].unique()
)

warehouse_filter = st.sidebar.multiselect(
    "Warehouse Block",
    df["Warehouse_block"].unique(),
    default=df["Warehouse_block"].unique()
)

filtered = df[
    (df["Mode_of_Shipment"].isin(shipment_filter)) &
    (df["Warehouse_block"].isin(warehouse_filter))
]

# -------------------------
# KPI Section
# -------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Orders", len(filtered))
col2.metric("Average Discount", round(filtered["Discount_offered"].mean(),2))
col3.metric("Average Weight (g)", round(filtered["Weight_in_gms"].mean(),2))

# -------------------------
# Charts Section
# -------------------------

col1, col2 = st.columns(2)

with col1:
    fig1 = px.pie(
        filtered,
        names="Reached.on.Time_Y.N",
        title="Delivery Status Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(
        filtered,
        x="Mode_of_Shipment",
        color="Reached.on.Time_Y.N",
        barmode="group",
        title="Shipment Mode vs Delivery Status"
    )
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(
    filtered,
    x="Discount_offered",
    y="Weight_in_gms",
    color="Reached.on.Time_Y.N",
    title="Discount vs Package Weight"
)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Insights Section
# -------------------------

st.header("Key Insights")

st.write("""
• A noticeable portion of deliveries are delayed.

• Shipment mode influences delivery performance.

• Certain warehouse blocks show higher delay counts.

• Higher discounts may correlate with delayed shipments during promotions.
""")

# -------------------------
# Machine Learning Model
# -------------------------

st.header("Delivery Delay Prediction")

model_df = pd.read_csv("data/processed/cleaned_data.csv")

features = model_df[[
    "Customer_care_calls",
    "Discount_offered",
    "Weight_in_gms",
    "Cost_of_the_Product",
    "Prior_purchases"
]]

target = model_df["Reached.on.Time_Y.N"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

# -------------------------
# User Input
# -------------------------

st.subheader("Predict Delivery Delay")

calls = st.number_input("Customer Care Calls",1,10,3)
discount = st.number_input("Discount Offered",0,100,10)
weight = st.number_input("Package Weight (g)",100,5000,1000)
cost = st.number_input("Product Cost",50,500,200)
prior = st.number_input("Prior Purchases",0,20,5)

input_data = np.array([[calls,discount,weight,cost,prior]])

prediction = model.predict(input_data)

if st.button("Predict Delivery Status"):

    if prediction[0] == 1:
        st.error(" High chance of delivery delay")
    else:
        st.success("Delivery likely on time")

# -------------------------
# Model Accuracy
# -------------------------

accuracy = model.score(X_test,y_test)

st.write("Model Accuracy:", round(accuracy*100,2), "%")

# -------------------------
# Feature Importance
# -------------------------

st.header("Factors Influencing Delivery Delay")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance for Delivery Delay Prediction"
)

st.plotly_chart(fig_importance, use_container_width=True)