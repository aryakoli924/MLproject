import streamlit as st

# Title
st.title("Simple Machine Learning App")

# Text input
user_text = st.text_input("Enter some text:")

# Display entered text
if user_text:
    st.write("You entered:", user_text)
