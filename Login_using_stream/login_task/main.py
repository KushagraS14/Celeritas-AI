import streamlit as st

# --- Custom Background CSS ---
st.markdown(
    """
    <style>
    /* Page background */
    body {
        background-image: url("https://cdn.pixabay.com/photo/2023/06/12/00/11/smartphone-8057248_1280.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Make Streamlit’s containers transparent */
    [data-testid="stAppViewContainer"],
    .stApp,
    .block-container {
        background: transparent !important;
    }
 """,
    unsafe_allow_html=True
)

# --- Page Title ---
st.markdown("<h1 style='color: white; text-align: center;'>Login Page</h1>", unsafe_allow_html=True)


# --- Centered Login Form ---
st.markdown('<div class="login-container"><div class="login-box">', unsafe_allow_html=True)

username_db = "admin"
password_db = "1234"

username = st.text_input("", placeholder="Enter your username")
password = st.text_input("", placeholder="Enter password", type="password")
login = st.button("Login")

if login:
    if username == username_db and password == password_db:
        st.success("Login Successful")
    else:
        st.error("Invalid credentials")

st.markdown('</div></div>', unsafe_allow_html=True)
