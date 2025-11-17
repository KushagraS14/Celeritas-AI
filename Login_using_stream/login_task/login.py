import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")




# --- CREDENTIALS ---
username_db = "admin"
password_db = "1234"

# --- LOGIN LOGIC ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h1 style='color: black; text-align: center;'>Admin Page</h1>", unsafe_allow_html=True)

    st.markdown('<div class="login-container"><div class="login-box">', unsafe_allow_html=True)
    username = st.text_input(" " ,placeholder="Enter your Username")
    password = st.text_input("", placeholder="Enter password", type="password")
    login = st.button("Login", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if login:
        if username == username_db and password == password_db:
            st.session_state.authenticated = True
            st.success("✅ Login Successful! Redirecting...")
            st.switch_page("pages/Chatbot4.py")  # Redirect to the main app
        else:
            st.error("❌ Invalid credentials")

