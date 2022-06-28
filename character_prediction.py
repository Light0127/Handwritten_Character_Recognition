import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import os
import tensorflow as tf
from keras.models import load_model

model = load_model('model_hand.h5')

def show_character():
      
    st.title("""Handwritten Character Recognition""")
    st.write("""### Choose an image which has a Handwritten Character""")
    
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg','jpeg'])
    #if st.button("Yes I'm ready to rumble"):
    pred = st.button("Predict")

    word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

    

    if pred:
        try:
            st.image(uploaded_file, caption="Uploaded Image", width=120)
            img = Image.open(uploaded_file)
          
            img_copy = img.copy()
          
            img = img.save("ref_1.jpg")
          
            img_copy=img_copy.save("ref_2.jpg")
          
            img1=cv2.imread(r"C:\Users\bhuva\OneDrive\Desktop\Handwritten_character_recognition\ref_1.jpg")
          
            img2=cv2.imread(r"C:\Users\bhuva\OneDrive\Desktop\Handwritten_character_recognition\ref_2.jpg")
          
            img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
          
            img = cv2.resize(img1, (400,440))
          
            img_copy = cv2.GaussianBlur(img2, (7,7), 0)
          
            img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
          
            _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
          
            img_final = cv2.resize(img_thresh, (28,28))
          
            img_final =np.reshape(img_final, (1,28,28,1))
          
            img_pred = word_dict[np.argmax(model.predict(img_final))]
          
            st.subheader(f"The predicted character is \"{img_pred}\" ")
          
        except:
            st.error("Please select a \'.jpg\' or \'.jpeg\' or \'.png\' file")


