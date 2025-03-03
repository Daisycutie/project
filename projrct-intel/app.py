import streamlit as st
import pandas as pd

df = pd.read_csv("train(1).csv")
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

    st.write("ที่นี่คุณสามารถใส่รายละเอียดเกี่ยวกับแอป, วิธีการใช้งาน หรือข้อมูลที่เกี่ยวข้องกับแอป.")
    st.dataframe(df)
def page_page2():
    st.title("**Neural Network**")
    st.write("ยินดีต้อนรับสู่หน้า 1")
    st.write("นี่คือหน้าที่ 1 ของแอป.")
    st.subheader("กราฟตัวอย่าง")

def page_page3():
    st.title("**Demo Neural Network**")
    st.write("ยินดีต้อนรับสู่หน้า Demo Neural Network")
    st.subheader("กราฟตัวอย่าง")

def page_page4():
    st.title("**Demo Machine Learning**")
    st.write("ยินดีต้อนรับสู่หน้า Demo Machine Learning")
    st.subheader("ภาพตัวอย่าง")
    st.image("https://via.placeholder.com/150", caption="ตัวอย่างภาพ")

# สร้างแถบปุ่มใน Sidebar
st.sidebar.title("เลือกหน้า")
page = st.sidebar.radio("Select page", ("Machine Learning", "Neural Network", 
                                      "Demo Neural Network", "Demo Machine Learning"))

# แสดงเนื้อหาของหน้าที่ผู้ใช้เลือก
if page == "Machine Learning":
    page_page1()
elif page == "Neural Network":
    page_page2()
elif page == "Demo Neural Network":
    page_page3()
elif page == "Demo Machine Learning":
    page_page4()
