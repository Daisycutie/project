import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def page_page2():
    st.title("**Neural Network**")
    st.write("ยินดีต้อนรับสู่หน้า Neural Network")
    st.subheader("กราฟตัวอย่าง")
