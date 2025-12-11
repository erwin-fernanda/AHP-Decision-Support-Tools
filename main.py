import numpy as np
import pandas as pd
import streamlit as st
from function import AHP
from pathlib import Path
from function import AHP


def metric_box(label, value, delta=None, color="#f8f9fa"):
    delta_html = ""
    if delta is not None:
        delta_html = f"""
            <div style='color: {"green" if float(delta)>=0 else "red"}; 
                        font-weight:600; font-size: 14px;'>
                {delta}
            </div>
        """

    st.markdown(f"""
    <div style="
        background-color:{color};
        padding:20px;
        border-radius:12px;
        box-shadow:0 2px 6px rgba(0,0,0,0.1);
        width:230px;
        text-align:center;
        border:1px solid #eee;
        margin-bottom:15px;
    ">
        <div style="font-size:15px; color:#555; font-weight:600;">{label}</div>
        <div style="font-size:32px; color:#555; font-weight:700; margin-top:5px;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
    
ROOT_DIR = Path().resolve()

### Login for Application ###
# ---- LOAD USER DATA ----
@st.cache_data
def load_users():
    return pd.read_csv(f"{ROOT_DIR}/user_account/list_user_account.csv", sep=",")  # must be in your project folder

# ---- LOGIN FUNCTIONS ----
def login():
    st.session_state["logged_in"] = True
    st.session_state["username"] = st.session_state["temp_user"]

def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

users_df = load_users()

# ---- LOGIN SCREEN ----
if not st.session_state["logged_in"]:

    st.subheader("🔐 Login to AHP Decision Support Tool")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        login_btn = st.form_submit_button("Login")

        if login_btn:
            if username in users_df["username"].values:
                stored_password = users_df.loc[users_df["username"] == username, "password"].values[0]
                
                if password == stored_password:
                    st.session_state["temp_user"] = username
                    login()
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
            else:
                st.error("❌ Username not found")

    st.stop()  # Stop entire app until logged in


# --- Create a modal dialog ---
@st.dialog("AHP Reference Guide")
def ahp_reference_popup():
    st.markdown("""
    ### 🔢 Weighted Scores (Saaty Scale, 1980)
    - 1 → Equally important
    - 2 → Equally to moderately important
    - 3 → Moderately important
    - 4 → Moderately to strongly important
    - 5 → Strongly important
    - 6 → Strongly to very strongly important
    - 7 → Very strongly important
    - 8 → Very strongly to extremely important
    - 9 → Extremely important

    ### 🧮 Consistency Ratio (CR)
    - CR < **10 %** → ✅ **Acceptable**
    - CR >= **10 %** → ⚠️ **Not consistent**  
    """)
    
    st.subheader("📊 Random Index (RI) Table")

    ri_data = {
        "Matrix Size (n)": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "RI Value":        [0.00, 0.00, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59]
    }

    st.table(pd.DataFrame(ri_data))
    
    st.markdown("""
    ### 📚 References
    - Saaty, T. L. (1980). *The Analytic Hierarchy Process*.  
    - Saaty, T. L. (2008). "Decision making with AHP." *Int. Journal of Services Sciences*.

    """)
    
    
### Frontend for SideBar ###

with st.sidebar:
    st.write(f"Welcome, **{st.session_state['username']}**!")
    st.button("Logout", on_click=logout)
    
    st.markdown("---")

    st.subheader("📘 Description")
    st.markdown("""
    This tool helps users apply the **Analytic Hierarchy Process (AHP)**  
    to compare criteria using **pairwise comparisons** and generate:
    - Weighted priority scores  
    - Pairwise comparison matrix  
    - Consistency Ratio (CR)  
    - Eigenvector-based AHP weights  
    
    Built for ease of use with dynamic sliders and automated calculations.
    """)
    
    # --- Button to open popup ---
    if st.button("AHP Reference Guide"):
        ahp_reference_popup()

    st.markdown("---")

    st.subheader("👨‍💻 Creator")
    st.markdown("""
    **Erwin Fernanda**  
    Data and AI/ML Engineer  
    Medco E&P Indonesia  

    📧 *erwinfernanda.official@gmail.com*  
    """)

    st.markdown("---")

    st.subheader("📄 Project Info")
    st.markdown("""
    **Version:** 1.0.0  
    **Last Update:** Dec 2025  

    This app is developed using  
    **Python + Streamlit + NumPy + Pandas + Scikit Learn**.
    """)

    st.markdown("---")


### Frontend for AHP Tools ### 
st.title("📘 AHP Decision Support Tool")
    
# --- Step 1: enter variables ---
st.header("1. Define Variables")
variable_input = st.text_input("Enter variable name")

if "variables" not in st.session_state:
    st.session_state.variables = []

if st.button("Add variable"):
    if variable_input and variable_input not in st.session_state.variables:
        st.session_state.variables.append(variable_input)

st.subheader("Current Variables")

# Create 2-column grid
cols = st.columns(2)

for i, v in enumerate(st.session_state.variables):
    with cols[i % 2]:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])  # left: label, right: delete button
            c1.write(f"**{v}**")
            if c2.button("❌", key=f"del_{v}"):
                st.session_state.variables.remove(v)
                st.rerun()

st.markdown("---")

# st.write("Current variables:", st.session_state.variables)
# st.markdown("---")

# --- Step 2: Pairwise comparison ---
if len(st.session_state.variables) >= 2:
    st.header("2. Pairwise Comparison")
    variables = st.session_state.variables
    n = len(variables)

    # Always rebuild matrix if shape mismatches
    if "matrix" not in st.session_state or st.session_state.matrix.shape != (n, n):
        st.session_state.matrix = np.ones((n, n))
    
    if "init_score" not in st.session_state or st.session_state.matrix.shape != (n, n):
        st.session_state.init_score = {}

    matrix = st.session_state.matrix
    init_score = st.session_state.init_score
    
    for i in range(n):
        for j in range(i+1, n):
            key = f"{i}-{j}"
            weight = st.slider(
                f"How important is **{st.session_state.variables[i]}** compared to **{st.session_state.variables[j]}**?",
                1, 9, 1, key=key
            )
            
            if i < j:
                st.session_state.init_score[(st.session_state.variables[i], st.session_state.variables[j])] = weight

            st.session_state.matrix[i, j] = weight
            st.session_state.matrix[j, i] = 1 / weight

    st.subheader("Pairwise Comparison Matrix")
    st.dataframe(pd.DataFrame(st.session_state.matrix, 
                              index=st.session_state.variables, 
                              columns=st.session_state.variables))

    st.markdown("---")
    
    # --- Step 3: AHP Weights ---
    st.header("3. AHP Results")
    
    # Eigenvector method
    # eigvals, eigvecs = np.linalg.eig(st.session_state.matrix)
    # max_eig_index = np.argmax(eigvals)
    # weights = np.real(eigvecs[:, max_eig_index])
    # weights = weights / weights.sum()

    ### Method 1 - Approach with Eigen Calculation from Linear Regression
    # λ_max = np.real(eigvals[max_eig_index])
    # RI_dict = AHP.GetEigenValues(st.session_state.variables, st.session_state.init_score).load_RI()
    # RI = RI_dict[n]
    # CI = (λ_max - n) / (n - 1)
    # CR = CI / RI if RI != 0 else 0
    
    ### Method 2 - Aprroach from Manual Calculation 
    Eigen_Final = AHP.GetEigenValues(st.session_state.variables, st.session_state.init_score).run_calculation()
    weights = []
    
    for val in st.session_state.variables:
        weights.append(Eigen_Final[val])
    
    λ_max = Eigen_Final['Eigenvalue Maximum']
    CI = Eigen_Final['CI (Consistency Index)']
    CR = np.abs(Eigen_Final['CR (Consistency Ratio)']) * 100
    
    df_w = pd.DataFrame({"Variable": st.session_state.variables, "Weight": weights})
    st.write(df_w)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric_box("Eigen Values", f"{λ_max:.4f}")
        
    with col2:
        metric_box("CI (Consistency Index)", f"{CI:.4f}")

    with col3:
        metric_box("CR (Consistency Ratio)", f"{CR:.2f} %")
    
    if CR >= 10:
        st.warning(f"⚠️ Consistency Ratio is not consistent (CR = {CR:.2f} % >= 10 %). Please revise your pairwise comparisons.")
    else:
        st.success(f"✅ Consistency Ratio is acceptable (CR = {CR:.2f} % < 10 %).")
    
    st.markdown("---")
    
else:
    st.warning("Add at least 2 variables to continue.")
    