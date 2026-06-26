import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Senior Citizen Dashboard",
    page_icon="👴",
    layout="wide"
)

st.title("👴 Senior Citizen Identification Dashboard")
st.markdown("---")
CSV_FILE = "visits.csv"

if not os.path.exists(CSV_FILE):
    st.error("No detection data found.")
    st.stop()

df = pd.read_csv(CSV_FILE)
total = len(df)

senior = len(df[df["Status"] == "Senior Citizen"])

male = len(df[df["Gender"] == "Male"])

female = len(df[df["Gender"] == "Female"])
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Visitors", total)

c2.metric("Senior Citizens", senior)

c3.metric("Male", male)

c4.metric("Female", female)
st.markdown("---")

st.subheader("Detection History")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)
st.markdown("---")

st.subheader("Age Distribution")

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(df["Age"], bins=10)

ax.set_xlabel("Age")

ax.set_ylabel("Count")

st.pyplot(fig)
st.markdown("---")

st.subheader("Gender Distribution")

gender_count = df["Gender"].value_counts()

fig2, ax2 = plt.subplots(figsize=(5,5))

ax2.pie(
    gender_count,
    labels=gender_count.index,
    autopct="%1.1f%%"
)

st.pyplot(fig2)
st.markdown("---")

with open(CSV_FILE, "rb") as file:

    st.download_button(
        "⬇ Download CSV",
        file,
        file_name="visits.csv",
        mime="text/csv"
    )
st.markdown("---")

st.success("Dashboard Loaded Successfully")
    