import tensorflow as tf
import pandas as pd
import pickle
import streamlit as st


#* load the trained model

model=tf.keras.models.load_model('model.h5')


#* load the encoders and scaler

with open('label_encode_geography.pkl','rb')as f:
    onehot_encode_geo=pickle.load(f)


with open('label_encode_gender.pkl','rb')as f:
    label_encode_gender=pickle.load(f)


with open('scaler.pkl','rb')as f:
    scaler=pickle.load(f)


#* streamlit app

st.title('Customer Churn Prediction')

#*user input
geography=st.selectbox('Geography',onehot_encode_geo.categories_[0])
gender=st.selectbox('Gender',label_encode_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_number=st.selectbox('Is Active Member',[0,1])


#*input data dataframe
input_df=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encode_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_number],
    'EstimatedSalary':[estimated_salary]
})

#*one hot encode 'Geography'
geo_encoded=onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encode_geo.get_feature_names_out(['Geography']))



#* combine one hot encoded columns with input data
#!input_data = pd.concat([input_df.reset_index(drop=True),geo_encoded_df],axis=1) (safer approach when rows are deleted or changed)
input_df = pd.concat([input_df,geo_encoded_df],axis=1)

#* scale the input data
input_df_scaled=scaler.transform(input_df)

#*prediction and writing in streamilit app
prediction=model.predict(input_df_scaled)
prediction_prob=prediction[0][0]

st.write(f"Churn prediction probability: {prediction_prob:.2f}")

if prediction_prob >0.5:
    st.write('The customer likely to leave (churn --> Yes)')
else:
    st.write('The customer is not likely to leave (churn --> No)')
