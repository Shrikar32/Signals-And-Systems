import streamlit as st

st.title("Simple Streamlit app")
st.header("Welcome")
st.write("This is a basic streamlit app")

name = st.text_input("Enter your name")
if name:
    st.success(f"Hello, {name}! \nHow are you?")

number = st.slider("Pick your age", 0, 100)
st.write(f"Your age is {number}")    