
import numpy as np
import pickle
import pandas as pd
import streamlit as st 
# import joblib
from tensorflow.keras.applications import  VGG19
import cv2
from keras.applications.vgg19 import preprocess_input

#app=Flask(__name__)
#Swagger(app)

model = open("skin_model.pkl","rb")
classifier=pickle.load(model)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_disease(image):

    d = {1:'Psoriasis pictures Lichen Planus and related diseases', 2:'Seborrheic Keratoses and other Benign Tumors', 
         3:'Tinea Ringworm Candidiasis and other Fungal Infections'}
   
    prediction=np.argmax(classifier.predict(image))
    
    return d[prediction]

def load_img(img_file):
    images=[]
    img=cv2.imread(img_file)
    img=cv2.resize(img,(240,240))
    images.append(img)
    x_test=np.asarray(images)
    test_img=preprocess_input(x_test)
    vgg19 = VGG19(include_top=False,weights='imagenet')
    features_test=vgg19.predict(test_img)
    num_test=x_test.shape[0]
    f_img=features_test.reshape(num_test,25088)
    
    return f_img

#<h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>

def main():
    st.title("Skin Disease Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # variance = st.text_input("Variance","Type Here")
    # skewness = st.text_input("skewness","Type Here")
    # curtosis = st.text_input("curtosis","Type Here")
    # entropy = st.text_input("entropy","Type Here")

    img_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    image = load_img(img_file)
    result=""

    if st.button("Predict"):
        result=predict_disease(image)
    st.success('Predicted disease is: {}'.format(result))
    # if st.button("About"):
    #     st.text("Lets LEarn")
    #     st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    