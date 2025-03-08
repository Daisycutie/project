import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ฟังก์ชัน page_page3 ที่จะถูกเรียกเมื่อแสดงหน้า demo
def page_page3():
    st.title("**Demo Neural Network**")
    
    # โหลดโมเดลที่บันทึกไว้
    model = tf.keras.models.load_model('your_model.h5')

    # กำหนดชื่ออารมณ์ที่คาดการณ์
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # ฟังก์ชันในการประมวลผลและทำนาย
    def predict_emotion(img):
        # แปลงภาพเป็น array และทำการ preprocess
        img = img.convert('L')  # แปลงภาพเป็น grayscale
        img = img.resize((48, 48))  # ปรับขนาดให้เหมาะสม
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize ค่า pixel
        img_array = img_array.reshape(-1, 48, 48, 1)  # Reshape ให้เป็น 4D array
        
        # ทำนายอารมณ์
        pred = model.predict(img_array)
        predicted_emotion = np.argmax(pred)
        
        return predicted_emotion, pred

    
    st.write("Upload an image to predict the emotion")

    # อัพโหลดภาพ
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # เปิดภาพที่อัพโหลด
        img = Image.open(uploaded_file)
        
        # แสดงภาพที่อัพโหลด
        st.image(img, caption='Uploaded Image', use_container_width=True)
        
        # ทำนายอารมณ์
        predicted_emotion, prediction_probs = predict_emotion(img)
        
        # แสดงผลการทำนาย
        st.subheader(f"Predicted Emotion: {emotions[predicted_emotion]}")
