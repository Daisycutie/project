import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


st.sidebar.title("เลือกหน้า")
page = st.sidebar.radio("Select page", ("Machine Learning", "Neural Network", 
                                        "Demo Neural Network", "Demo Machine Learning"))

def page_page1():
    st.title("**Machine Learning**")
    st.markdown('หาข้อมูลจากเว็บ Kaggle ➡ [**Big Mart Sales Dataset**](https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets/data)', unsafe_allow_html=True)
    st.subheader("**เนื้อหาเกี่ยวกับ**")
    st.write("ยอดขายของ BigMart สำหรับผลิตภัณฑ์ 1,559 รายการจากร้านค้า 10 แห่ง")
    
    st.subheader("**Features**")
    st.write("""
    - **Item_Identifier** - รหัสผลิตภัณฑ์เฉพาะ  
    - **Item_Weight** - น้ำหนักของผลิตภัณฑ์  
    - **Item_Fat_Content** - ผลิตภัณฑ์มีไขมันต่ำหรือไม่  
    - **Item_Visibility** - % ของพื้นที่แสดงผลรวมของผลิตภัณฑ์ทั้งหมดในร้านค้า  
    - **Item_Type** - หมวดหมู่ของผลิตภัณฑ์  
    - **Item_MRP** - ราคาขายปลีกสูงสุด  
    - **Outlet_Identifier** - รหัสร้านค้าเฉพาะ  
    - **Outlet_Establishment_Year** - ปีที่ก่อตั้งร้านค้า  
    - **Outlet_Size** - ขนาดของร้านค้า  
    - **Outlet_Location_Type** - ประเภทของเมืองที่ร้านตั้งอยู่  
    - **Outlet_Type** - ประเภทของร้านค้า (ร้านขายของชำหรือซูเปอร์มาร์เก็ต)  
    """)
    

    
    st.subheader("**ข้อมูลในไฟล์ csv**")
    df = pd.read_csv("https://raw.githubusercontent.com/Daisycutie/project/refs/heads/main/projrct-intel/train%20(1).csv")
 
    st.dataframe(df)

    st.header("การเตรียมข้อมูลและการพัฒนา")
    
    #หาnull
    code = """ 
    df.isnull().sum()
    """
    st.code(code, language="python")
    
    st.write("""
        ดูว่าแต่ละ column มีค่าเป็นnullไหมแล้วถ้ามีให้แสดงว่ามีกี่อัน
    """)
    #เติมค่าที่หายไป
    code = """
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
    """
    st.code(code, language="python")
    st.write("""
        จัดการกับค่าที่หายไปในคอลัมน์ Item_Weight เติมค่าที่หายไปด้วยค่า Mean \n
        และ Outlet_Size จะแทนที่ค่าที่หายไปในคอลัมน์ด้วย Mode
    """)
    #แบ่งtrain
    code = """
    X = df.drop(['Item_Outlet_Sales'], axis=1)  # Features
    y = df['Item_Outlet_Sales'] 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 
    """
    st.code(code, language="python")
    st.write("""
        เราเอาข้อมูลทั้งหมดมายกเว้นคอลัมน์ Item_Outlet_Salesกำหนดให้เป็นX และ เรากำหนดให้ y เป็นค่าที่เราต้องการพยากรณ์ เรากำหนดให้ y เป็นค่าที่เราต้องการพยากรณ์ คือ Item_Outlet_Sales 
        จากนั้นก็ทำการ split ข้อมูลแบ่งเป็น train 80% กับ test 20%  โดยให้ random state = 42
    """)
    
    #standartscaler
    code = """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    """
    st.code(code, language="python")
    st.write("""
        ปรับค่าของแต่ละฟีเจอร์ให้อยู่ในช่วงที่เหมาะสม โดยใช้ค่าเฉลี่ย (mean) และส่วนเบี่ยงเบนมาตรฐาน (standard deviation)
    """)
    
    #train svr
    code = """
    # โมเดล SVR (Support Vector Regression)
    svr_model = SVR(kernel='rbf')  # ใช้ kernel แบบ Radial Basis Function
    svr_model.fit(X_train_scaled, y_train)
    svr_val_pred = svr_model.predict(X_val_scaled)
    svr_mae = mean_absolute_error(y_val, svr_val_pred)
    print(f"SVR MAE: {svr_mae:.2f}")
    """
    st.code(code, language="python")
    st.write("""
        สร้างและฝึกโมเดลใช้ Support Vector Regression โดยใช้แนวคิดของ SVM ใช้ RBF Kernel เพื่อช่วยให้โมเดลจับความสัมพันธ์ที่ซับซ้อนได้ดีขึ้น
        หลังจากฝึกโมเดลเรียบร้อยแล้ว นำโมเดลที่ได้มาทดสอบจากนั้นทำการวัดประสิทธิภาพของโมเดลโดยใช้ Mean Absolute Error (MAE) ซึ่งเป็นค่าที่บอกว่าโดยเฉลี่ยแล้วค่าที่โมเดลทำนายแตกต่างจากค่าจริงมากน้อยแค่ไหน
        ยิ่งค่านี้ต่ำ หมายความว่าโมเดลสามารถทำนายได้แม่นยำมากขึ้น
    """)
    
    #train random forest
    code = """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train) 
    rf_val_pred = rf_model.predict(X_val)
    rf_mae = mean_absolute_error(y_val, rf_val_pred)
    print(f"Random Forest MAE: {rf_mae:.2f}")
    """
    st.code(code, language="python")
    st.write("""
       ร้างและฝึกโมเดล Random Forest Regressor ซึ่งเป็นโมเดลที่ใช้ หลายๆ Decision Tree มาทำนายค่า โดยใช้เทคนิคการสุ่มข้อมูล
       เพื่อให้แต่ละต้นไม้เรียนรู้จากข้อมูลที่แตกต่างกัน ช่วยเพิ่มความแม่นยำ หลังจากนั้นทำการประเมินโมเดลด้วย Mean Absolute Error (MAE) 
    """)
    
    
    

   

def page_page2():
    st.title("**Neural Network**")
    st.write("ยินดีต้อนรับสู่หน้า Neural Network")
    st.subheader("กราฟตัวอย่าง")

def page_page3():
    st.title("**Demo Neural Network**")
   


def page_page4():
    st.title("**Demo Machine Learning**")
    # โหลดข้อมูล
    df = pd.read_csv("https://raw.githubusercontent.com/Daisycutie/project/refs/heads/main/projrct-intel/train%20(1).csv")

    # ตรวจสอบว่าโหลดข้อมูลได้หรือไม่
    if df.empty:
        st.error("❌ ไม่สามารถโหลดข้อมูลได้")
        return
    
    # จัดการค่า Missing Values
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
    df = pd.get_dummies(df, drop_first=True)

    # Features และ Target Variable
    if 'Item_Outlet_Sales' not in df.columns:
        st.error("❌ คอลัมน์ 'Item_Outlet_Sales' ไม่พบในข้อมูล!")
        return

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

    # ทำนายค่า
    svr_pred = svr_model.predict(X_val_scaled)
    rf_pred = rf_model.predict(X_val)

    # คำนวณค่า MAE
    svr_mae = mean_absolute_error(y_val, svr_pred)
    rf_mae = mean_absolute_error(y_val, rf_pred)

    # แสดงผล
    st.write(f"📊 **SVR MAE:** {svr_mae:.2f}")
    st.write(f"🌳 **Random Forest MAE:** {rf_mae:.2f}")

# แสดงเนื้อหาของหน้าที่ผู้ใช้เลือก
if page == "Machine Learning":
    page_page1()
elif page == "Neural Network":
    page_page2()
elif page == "Demo Neural Network":
    page_page3()
elif page == "Demo Machine Learning":
    page_page4()
