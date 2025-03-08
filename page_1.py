import streamlit as st
import pandas as pd

def page_page1():
    st.title("**Machine Learning**")
    st.markdown('หาข้อมูลจากเว็บ Kaggle ➡️ [**Big Mart Sales Dataset**](https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets/data)', unsafe_allow_html=True)
    st.subheader("**เนื้อหาเกี่ยวกับ**")
    st.write("การพยากรณ์ยอดขายของร้านค้า โดยใช้ข้อมูลสินค้าและสาขาของ Big Mart 🍽️")
    
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
        เรามีข้อมูลบางส่วนที่หายไป เช่นในคอลัมน์ 'Item_Weight' และ 'Outlet_Size' เพื่อไม่ให้ข้อมูลขาดหายไป ทำการเติมค่าให้กับคอลัมน์เหล่านี้:
        คอลัมน์ 'Item_Weight' จะเติมค่าด้วยค่าเฉลี่ยของน้ำหนักสินค้าทั้งหมด
        คอลัมน์ 'Outlet_Size' จะเติมค่าด้วยค่าที่เกิดบ่อยที่สุดในคอลัมน์นั้น (ค่าโหมด)
    """)
    #
    code = """ 
    df = pd.get_dummies(df, drop_first=True)
    """
    st.code(code, language="python")
    st.write("""
       get_dummies() ใช้แปลงข้อมูลที่เป็นหมวดหมู่ (เช่น คอลัมน์ที่เป็นข้อความหรือประเภทต่าง ๆ) ให้กลายเป็นตัวเลข (One-Hot Encoding)
       drop_first=True ลดจำนวนคอลัมน์ลง โดยการลบคอลัมน์แรกในแต่ละหมวดหมู่ที่มีการแปลงเป็นตัวเลข (เพื่อหลีกเลี่ยงปัญหาการ multicollinearity)

    """)
    
    #แบ่ง
    code = """
    X = df.drop(['Item_Outlet_Sales'], axis=1)  # Features
    y = df['Item_Outlet_Sales'] 
    """
    st.code(code, language="python")
    st.write("""
        ขั้นตอนนี้คือการแยกข้อมูลเป็น Features (X) และ Target (y):
        X คือข้อมูลคุณสมบัติของสินค้า (ทุกคอลัมน์ยกเว้น 'Item_Outlet_Sales')
        y คือผลลัพธ์ที่เราต้องการทำนาย ซึ่งก็คือยอดขายสินค้าในแต่ละร้าน ('Item_Outlet_Sales')
    """)
    #train
    code = """
     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    """
    st.code(code, language="python")
    st.write("""
        เราจะนำข้อมูล Features และ Target ที่มีทั้งหมดมาแบ่งออกเป็น 2 ส่วน\n
        - 80% ของข้อมูลจะใช้ในการฝึกโมเดล (Training Set)\n
        - 20% ของข้อมูลจะใช้ในการทดสอบโมเดล (Test Set)\n
    """)
    
    #standartscaler
    code = """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    """
    st.code(code, language="python")
    st.write("""
        ในขั้นตอนนี้เราใช้ StandardScaler เพื่อทำการปรับขนาดข้อมูลทั้งหมดให้มีค่าเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1
        การทำ standardization นี้สำคัญเมื่อใช้โมเดลที่ขึ้นอยู่กับระยะห่างของข้อมูล (เช่น SVM หรือ KNN) เนื่องจากมันช่วยให้โมเดลสามารถเรียนรู้ได้ดีขึ้น
    """)
    
    #train svr
    code = """
    # โมเดล SVR (Support Vector Regression)
    svr_model = SVR(kernel='rbf')  # ใช้ kernel แบบ Radial Basis Function
    svr_model.fit(X_train_scaled, y_train)
    """
    st.code(code, language="python")
    st.write("""
       เราสร้างโมเดล SVR (Support Vector Regression) โดยใช้ Radial Basis Function (RBF) เป็น kernel ซึ่งเหมาะกับการทำนายข้อมูลที่ไม่เป็นเชิงเส้น
       จากนั้นเราฝึกโมเดลโดยใช้ข้อมูลใน X_train_scaled และ y_train
    """)
    
    #train random forest
    code = """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train) 
    """
    st.code(code, language="python")
    st.write("""
      โมเดล Random Forest Regressor จะใช้การสร้างป่าไม้ตัดสิน (Decision Trees) หลายๆ ต้นมาใช้ร่วมกันเพื่อทำการทำนาย โดยจะใช้ 100 ต้น (n_estimators=100) และฝึกด้วยข้อมูล X_train และ y_train
    """)
    
    #predict
    code = """
    svr_val_pred = svr_model.predict(X_val_scaled)
    rf_val_pred = rf_model.predict(X_val)

    """
    st.code(code, language="python")
    st.write("""
      เราทำนายค่าผลลัพธ์จากชุดทดสอบ (X_val) สำหรับทั้งสองโมเดล:
        - svr_val_pred คือการทำนายจากโมเดล SVR
        - rf_val_pred คือการทำนายจากโมเดล Random Forest
      
    """)
    
    #คำนวณค่าความผิดพลาด (MAE)
    code = """
    svr_mae = mean_absolute_error(y_val, svr_val_pred)
    rf_mae = mean_absolute_error(y_val, rf_val_pred)
    """
    st.code(code, language="python")
    st.write("""
      เราคำนวณค่า MAE (Mean Absolute Error) ซึ่งคือค่าความผิดพลาดเฉลี่ยของการทำนายจากทั้งสองโมเดล:
        - svr_mae คือความผิดพลาดจากโมเดล SVR
        - rf_mae คือความผิดพลาดจากโมเดล Random Forest
      
    """)
    
    #แสดงผลลัพธ์
    code = """
    print(f"SVR MAE: {svr_mae:.2f}")
    print(f"Random Forest MAE: {rf_mae:.2f}")

    """
    st.code(code, language="python")
    st.write("""
         สุดท้ายเราจะแสดงผลค่าความผิดพลาด (MAE) ของทั้งสองโมเดล เพื่อดูว่าโมเดลไหนทำงานได้ดีกว่าในการทำนาย
     """)
