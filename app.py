import pickle
import streamlit as st
import keras
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.web.cli as stcli
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


def return_prediction(model, scaler, sample_json):
    x1 = sample_json[' Age (yrs)']
    x2 = sample_json['Avg. F size (L) (mm)']
    x3 = sample_json['Avg. F size (R) (mm)']
    x4 = sample_json['BMI']
    x5 = sample_json['BP _Diastolic (mmHg)']
    x6 = sample_json['BP _Systolic (mmHg)']
    x7 = sample_json['Cycle(R/I)']
    x8 = sample_json['Endometrium (mm)']
    x9 = sample_json['FSH(mIU/mL)']
    x10 = sample_json['Fast food (Y/N)']
    x11 = sample_json['Follicle No. (L)']
    x12 = sample_json['Follicle No. (R)']
    x13 = sample_json['Hair loss(Y/N)']
    x14 = sample_json['Hb(g/dl)']
    x15 = sample_json['Hip(inch)']
    x16 = sample_json['I   beta-HCG(mIU/mL)']
    x17 = sample_json['II    beta-HCG(mIU/mL)']
    x18 = sample_json['LH(mIU/mL)']
    x19 = sample_json['PRL(ng/mL)']
    x20 = sample_json['Pimples(Y/N)']
    x21 = sample_json['RR (breaths/min)']
    x22 = sample_json['Reg.Exercise(Y/N)']
    x23 = sample_json['Skin darkening (Y/N)']
    x24 = sample_json['TSH (mIU/L)']
    x25 = sample_json['Waist(inch)']
    x26 = sample_json['Waist:Hip Ratio']
    x27 = sample_json['Weight (Kg)']
    x28 = sample_json['Weight gain(Y/N)']
    x29 = sample_json['hair growth(Y/N)']
    data = [
        [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
         x25, x26, x27, x28, x29]]
    classes = np.array([1, 0])
    data = scaler.fit_transform(data)
    class_ind = np.argmax(model.predict(data)[0])
    print("Indicator-> 1:has Pcos, 0: no Pcos")
    print("Results as per the values entered in report: ", classes[class_ind])


Scaler = pickle.load(open("C:\\Users\\Suyash Pandey\\PycharmProjects\\PCOS_PREDICTOR\\pcos_scaler.pkl", "rb"))

model = load_model("Final_Pcos_model.h5")

st.title("PCOS_Predictor")

x1 = st.number_input('Age (yrs)')
x2 = st.number_input('Avg. F size (L) (mm)')
x3 = st.number_input('Avg. F size (R) (mm)')
x4 = st.number_input('BMI')
x5 = st.number_input('BP Diastolic (mmHg)')
x6 = st.number_input('BP Systolic (mmHg)')
x7 = st.number_input('Cycle(R/I)')
x8 = st.number_input('Endometrium (mm)')
x9 = st.number_input('FSH (mIU/mL)')
x10 = st.number_input('Fast food (Y/N)')
x11 = st.number_input('Follicle No. (L)')
x12 = st.number_input('Follicle No. (R)')
x13 = st.number_input('Hair loss (Y/N)')
x14 = st.number_input('Hb (g/dl)')
x15 = st.number_input('Hip (inch)')
x16 = st.number_input('I beta-HCG (mIU/mL)')
x17 = st.number_input('II beta-HCG (mIU/mL)')
x18 = st.number_input('LH (mIU/mL)')
x19 = st.number_input('PRL (ng/mL)')
x20 = st.number_input('Pimples (Y/N)')
x21 = st.number_input('RR (breaths/min)')
x22 = st.number_input('Reg. Exercise (Y/N)')
x23 = st.number_input('Skin darkening (Y/N)')
x24 = st.number_input('TSH (mIU/L)')
x25 = st.number_input('Waist (inch)')
x26 = st.number_input('Waist:Hip Ratio')
x27 = st.number_input('Weight (Kg)')
x28 = st.number_input('Weight gain (Y/N)')
x29 = st.number_input('Hair growth (Y/N)')
if st.button('Predict'):
    dc = [[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10. x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29]]
    inp = Scaler.fit_transform(dc)
    res = model.predict(inp)
    class_x = np.argmax(res, axis=1)
    if class_x == [[1]]:
        st.header("Patient is suffering from PCOS")
    else:
        st.header("No PCOS")
