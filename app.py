import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import check
from PIL import Image
# Set page title
st.set_page_config(page_title="My Streamlit App")
img = Image.open('ani.jpg')
# Add a header
st.header("Welcome to my Streamlit App")
# Add an image
st.image(img,
         caption="Show your favourite anime",
         use_column_width=True)

file = open("uid.pkl",'rb')
user_id = pickle.load(file)
file.close()
model = keras.models.load_model('model1')
st.title('Anime Recommendation system')

usr_id = st.selectbox(
    'Please enter your UserID from drop down menu',
    user_id)

st.write('You selected:', usr_id)
st.write('How many anime you want to recommend')
number = st.number_input('Enter a number:',min_value=1)
st.write('The number entered by u is: ', number)
if st.button('Recommend'):
    anime_name = check.pred_top_n_anime(model, usr_id, number)
    st.write(anime_name)
