import streamlit as st


def page_page2():
    st.title("**Neural Network**")
    st.markdown("""
    หาข้อมูลจากเว็บ Kaggle ➡️ 
    [**Fer2013 Facial Expression Recognition Dataset**](https://www.kaggle.com/datasets/xavier00/fer2013-facial-expression-recognition-dataset)
    """, unsafe_allow_html=True)

    st.subheader("เนื้อหาเกี่ยวกับ")
    st.markdown("""
        การจดจำอารมณ์จากใบหน้ามนุษย์  โดยใช้ภาพใบหน้าขาวดำขนาด 48x48 พิกเซล ซึ่งถูกจัดให้อยู่ใน 7 ประเภทของอารมณ์ ได้แก่
        - **Angry** (โกรธ)  😡
        - **Disgust** (รังเกียจ)  🤢
        - **Fear** (กลัว)  😨
        - **Happy** (มีความสุข)  😄
        - **Neutral** (เป็นกลาง)  😐
        - **Sad** (เศร้า)  😭
        - **Surprise** (ประหลาดใจ) 😮
    """)
    
    st.subheader("Feature")
    st.write("""
    มี 2 Feature หลักอยู่ในรูปแบบ CSV File ได้แก่

    1. **emotion** → ค่าตัวเลขที่แสดงอารมณ์ของภาพ
        - เป็น ค่าป้ายกำกับ (Label) ที่บ่งบอกว่าใบหน้าในภาพมีอารมณ์อะไร
        - มีทั้งหมด 7 ค่า แทนแต่ละอารมณ์ ดังนี้:
        - 0 → Angry (โกรธ)
        - 1 → Disgust (รังเกียจ)
        - 2 → Fear (กลัว)
        - 3 → Happy (มีความสุข)
        - 4 → Neutral (เป็นกลาง)
        - 5 → Sad (เศร้า)
        - 6 → Surprise (ประหลาดใจ)

    2. **pixels** → ค่าพิกเซลของภาพใบหน้า
        - เป็น สตริงของตัวเลขพิกเซล ที่มีค่า 0 - 255 (Grayscale)
        - แต่ละภาพมีขนาด 48 × 48  พิกเซล
    """)

    st.subheader("การเตรียมข้อมูลและการพัฒนา")
    
     #data preprocessing
    code = """
        data.head()
    """
    st.code(code, language="python")
    st.write("""
    ดูข้อมูล
    """)
    
    
    #data preprocessing
    code = """
        X = np.array([np.fromstring(img, dtype=int, sep=' ').reshape(48, 48) for img in data['pixels']])
        y = data['emotion'].values
    """
    st.code(code, language="python")
    st.write("""
    X คือข้อมูลภาพที่แปลงจากค่าสตริงของพิกเซลในแต่ละภาพ (ที่อยู่ในฟีลด์ pixels) มาเป็นอาร์เรย์ของตัวเลข โดยแต่ละภาพมีขนาด 48x48 พิกเซล.
    y คือข้อมูลป้ายกำกับ (label) ที่แสดงถึงอารมณ์ของแต่ละบุคคลที่มีอยู่ในแต่ละภาพ
    """)
    
    #์Normalization
    code = """
        X = X / 255.0

    """
    st.code(code, language="python")
    st.write("""
        ค่าพิกเซลในภาพเริ่มต้นจะอยู่ระหว่าง 0-255. เราทำการ Normalize (ปรับขนาด) ค่าพิกเซลให้เป็นช่วง 0-1 โดยการหารค่าพิกเซลด้วย 255
    """)
    
    #CNN
    code = """
    X = X.reshape(-1, 48, 48, 1)

    """
    st.code(code, language="python")
    st.write("""
    CNN (Convolutional Neural Networks) ต้องการข้อมูลที่มีลักษณะเป็น 4D Array (ตัวอย่าง, ความสูง, ความกว้าง, ช่องสี).
    ในที่นี้ ภาพมีขนาด 48x48 พิกเซล และเป็นภาพ Grayscale (ขาวดำ) ดังนั้น ช่องสีมีค่าเป็น 1
    """)
    
    #data preprocessing
    code = """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """
    st.code(code, language="python")
    st.write("""
        เราทำการแบ่งข้อมูลออกเป็นสองส่วน:
        - Training Data (80% ของข้อมูล)
        - Testing Data (20% ของข้อมูล)
        ใช้ train_test_split จากไลบรารี scikit-learn เพื่อให้การแบ่งข้อมูลเป็นไปอย่างสุ่ม (random)
    """)
    
    # การแปลง Labels 
    code = """
        y_train = to_categorical(y_train, num_classes=7)
        y_test = to_categorical(y_test, num_classes=7)

    """
    st.code(code, language="python")
    st.write("""
        โมเดลจะต้องการ labels ที่เป็น One-hot encoding (เช่น สำหรับ 7 อารมณ์ เช่น Happy, Sad, Fear, ฯลฯ).
        ทำการแปลงป้ายกำกับของทั้งชุดฝึกและชุดทดสอบให้อยู่ในรูปแบบ One-hot (เช่น แทน Happy ด้วย [0, 0, 0, 1, 0, 0, 0])
    """)
    #สร้างmodel
    code = """
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])

    """
    st.code(code, language="python")
    st.write("""
    โมเดลที่เราสร้างคือ Convolutional Neural Network (CNN) ซึ่งใช้เลเยอร์ Conv2D (การคำนวณภาพ) และ MaxPooling2D (การย่อลดขนาดภาพ)
    Conv2D ใช้สำหรับการดึงคุณลักษณะจากภาพ เช่น ขอบ หรือ รูปทรง
    MaxPooling2D ใช้ในการย่อลงขนาดข้อมูล (ลดความซับซ้อน)
    โมเดลประกอบด้วย
    3 เลเยอร์ Convolutional และ MaxPooling.
    - Flatten เพื่อนำข้อมูลที่ย่อขนาดแล้วมาทำการประมวลผล.
    - Dense เพื่อเชื่อมโยงข้อมูลเข้าสู่ชั้นที่มีหน่วยประมวลผล.
    - Dropout สำหรับการป้องกัน Overfitting.
    - Softmax ใช้ในการทำนายค่าอารมณ์ (7 อารมณ์)
    """)
    
    #Compile Model
    code = """
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    """
    st.code(code, language="python")
    st.write("""
    เราใช้ Adam Optimizer สำหรับการปรับค่าพารามิเตอร์ในระหว่างการฝึก.
    ใช้ categorical_crossentropy เป็นฟังก์ชันค่าเสียหาย (Loss function) เนื่องจากเป็นปัญหาหลายคลาส
    การคำนวณ accuracy ใช้ในการประเมินผล.
    """)
    
    #Model Training
    code = """
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=64)

    """
    st.code(code, language="python")
    st.write("""
        ทำการฝึกโมเดลโดยใช้ข้อมูลฝึก X_train และ y_train.
        เราทดสอบโมเดลกับชุดข้อมูล X_test และ y_test ในทุกๆ epoch เพื่อดูว่าประสิทธิภาพของโมเดลดีขึ้นหรือไม่
        Epochs คือจำนวนรอบการฝึก และ Batch size คือขนาดของชุดข้อมูลที่นำมาอัปเดตค่าพารามิเตอร์ในแต่ละรอบ
    """)
    
    #ประเมินผลโมเดล
    code = """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    """
    st.code(code, language="python")
    st.write("""
        ทำการประเมินผลโมเดลโดยใช้ชุดข้อมูลทดสอบ X_test และ y_test.
        เราจะได้ test_acc ซึ่งแสดงถึงความแม่นยำในการทำนายของโมเดล
    """)
    
    #ทำนายผล
    code = """
    pred = model.predict(sample_image)
    predicted_emotion = np.argmax(pred)

    """
    st.code(code, language="python")
    st.write("""
       เราทำการทำนายอารมณ์จากภาพตัวอย่างที่เลือกมา.
       ใช้ np.argmax(pred) เพื่อเลือกคลาสที่มีคะแนนสูงสุด (อารมณ์ที่ทำนายได้)
    """)
    
     #แสดงผล
    code = """
    plt.imshow(X_test[index].reshape(48, 48), cmap="gray")
    plt.title(f"Predicted Emotion: {emotions[predicted_emotion]}")
    plt.show()


    """
    st.code(code, language="python")
    st.write("""
       แสดงภาพที่ทำนายอารมณ์ ภาพนี้จะถูกแสดงพร้อมกับชื่ออารมณ์ที่โมเดลทำนาย
    """)
