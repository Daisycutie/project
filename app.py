import streamlit as st
import pandas as pd
from page_1 import page_page1 
from page_2 import page_page2 
from page_3 import page_page3
from page_4 import page_page4 
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
