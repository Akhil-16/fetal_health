import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler


scaler = pickle.load(open('scaling.pkl', "rb"))
model = pickle.load(open('fetal.pkl', 'rb'))


def predict(float_features):
    y = scaler.transform(float_features)
    predicted_value = model.predict(y)
    if predicted_value[0] == 1:
        return "Fetal health is Normal"
    elif predicted_value[0] == 2:
        return "Fetal health is  Suspect"
    else:
        return "Fetal health is Pathological"


def main():
    st.set_page_config(page_title='Fetal Health Classification')
    st.title('Fetal Health Classification')

    st.write("Please enter the required input parameters to predict the fetal health classification:")

    features = ['baseline value','accelerations','uterine_contractions',
       'prolongued_decelerations', 'abnormal_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'histogram_mode', 'histogram_mean', 'histogram_median',
       'histogram_variance']

    float_features = []
    for feature in features:
        value = st.number_input(feature, min_value=0.0)
        float_features.append(value)

    if st.button('Predict'):
        result = predict([float_features])
        st.write('The fetal health classification is:', result)


if __name__ == "__main__":
    main()
