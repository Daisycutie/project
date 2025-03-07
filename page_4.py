import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def page_page4():
    st.title("**Demo Machine Learning**")

    # โหลดข้อมูล
    df = pd.read_csv("https://raw.githubusercontent.com/Daisycutie/project/refs/heads/main/projrct-intel/train%20(1).csv")

    # จัดการค่า Missing Values
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Features และ Target Variable
    X = df.drop(['Item_Outlet_Sales'], axis=1)
    y = df['Item_Outlet_Sales']

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # โหลดโมเดล SVR และ Random Forest
    svr_model = SVR(kernel='rbf')
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # เทรนโมเดล
    svr_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train, y_train)

    # 📌 **ให้ผู้ใช้เลือกโมเดลที่ต้องการใช้**
    model_choice = st.radio("📊 เลือกโมเดลที่ต้องการใช้ทำนาย", ("SVR", "Random Forest"))

    # 📌 **แสดงฟอร์มให้ผู้ใช้กรอกข้อมูลเอง**
    st.subheader("🔍 กรอกค่าคุณสมบัติของสินค้าเพื่อลองทำนายยอดขาย")

    item_weight = st.number_input("Item Weight (น้ำหนักสินค้า)", min_value=0.0, value=5.0)
    item_mrp = st.number_input("Item MRP (ราคาขายปลีกสูงสุด)", min_value=0.0, value=200.0)
    item_visibility = st.number_input("Item Visibility (%)", min_value=0.0, max_value=1.0, value=0.02)
    outlet_year = st.number_input("Outlet Establishment Year", min_value=1980, max_value=2025, value=2005)

    # เมื่อกดปุ่ม "ทำนาย"
    if st.button("🔮 ทำนายยอดขาย"):
        # สร้าง DataFrame จากค่าที่ผู้ใช้กรอก
        input_data = pd.DataFrame([[item_weight, item_mrp, item_visibility, outlet_year]],
                                  columns=['Item_Weight', 'Item_MRP', 'Item_Visibility', 'Outlet_Establishment_Year'])

        # ปรับข้อมูลให้เหมือนกับข้อมูลฝึก
        input_data = pd.get_dummies(input_data)  # ทำ one-hot encoding เช่นเดียวกับข้อมูลฝึก

        # ตรวจสอบให้แน่ใจว่า input_data มีคอลัมน์ที่เหมือนกับข้อมูลที่ฝึก
        input_data = input_data.reindex(columns=X.columns, fill_value=0)  # เติมค่าศูนย์ในคอลัมน์ที่ขาด

        # Standardize ข้อมูลอินพุต สำหรับ SVR เท่านั้น
        if model_choice == "SVR":
            input_data_scaled = scaler.transform(input_data)

        # ใช้โมเดลที่เลือกทำนาย
        if model_choice == "SVR":
            prediction = svr_model.predict(input_data_scaled)[0]
            st.success(f"📈 **SVR ทำนายยอดขาย:** {prediction:.2f}")
        else:
            prediction = rf_model.predict(input_data)[0]
            st.success(f"🌳 **Random Forest ทำนายยอดขาย:** {prediction:.2f}")

