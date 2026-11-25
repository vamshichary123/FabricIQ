
import streamlit as st
import pandas as pd
import os
import hashlib
from datetime import datetime
from PIL import Image

from config import (
    MODEL_PATH,
    MODEL_NAME,
    MODEL_ACCURACY,
    CLASS_NAMES,
    HISTORY_PATH
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FabricIQ | Intelligent Fabric Inspection",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM DARK THEME
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

[data-testid="stHeader"] {
    background-color: #0B1120;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

h1, h2, h3, h4, p, label, span {
    color: #F1F5F9;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #60A5FA;
}

.subtitle {
    color: #94A3B8;
    font-size: 16px;
}

.metric-card {
    background: #172338;
    border: 1px solid #2B3B55;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    min-height: 125px;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #60A5FA;
}

.metric-label {
    font-size: 14px;
    color: #CBD5E1;
}

.section-card {
    background: #172338;
    border: 1px solid #2B3B55;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
    border: 1px solid #60A5FA;
}

[data-testid="stFileUploader"] {
    background-color: #172338;
    border: 1px dashed #3B82F6;
    border-radius: 10px;
    padding: 12px;
}

[data-testid="stMetric"] {
    background-color: #172338;
    border: 1px solid #2B3B55;
    padding: 15px;
    border-radius: 10px;
}

[data-testid="stMetricValue"] {
    color: #60A5FA;
}

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            return pd.read_csv(HISTORY_PATH)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return pd.DataFrame()


def save_history(defect, confidence, filename):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    record = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image Name": filename,
        "Predicted Class": defect,
        "Confidence": round(confidence, 2),
        "Decision": "Accepted" if defect == "defect free" else "Defect Detected"
    }])

    if os.path.exists(HISTORY_PATH):
        try:
            old_history = pd.read_csv(HISTORY_PATH)
            record = pd.concat(
                [old_history, record],
                ignore_index=True
            )
        except pd.errors.EmptyDataError:
            pass

    record.to_csv(HISTORY_PATH, index=False)


@st.cache_resource
def get_model():
    # Import TensorFlow only when prediction is requested.
    from prediction import load_model
    return load_model()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "inspection_result" not in st.session_state:
    st.session_state.inspection_result = None

if "image_hash" not in st.session_state:
    st.session_state.image_hash = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        "<h1 style='color:#60A5FA;'>🧵 FabricIQ</h1>",
        unsafe_allow_html=True
    )

    st.caption("Intelligent Fabric Inspection")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Defect Inspection",
            "Inspection History",
            "Model Information"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### System Status")
    st.caption("VGG16 Transfer Learning")
    st.caption("9 Fabric Categories")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        "<div class='main-title'>FabricIQ Dashboard</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Intelligent fabric defect detection and quality inspection system.</div>",
        unsafe_allow_html=True
    )

    st.write("")

    history = load_history()
    total_inspections = len(history)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{MODEL_ACCURACY}%</div>
            <div class="metric-label">Model Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(CLASS_NAMES)}</div>
            <div class="metric-label">Defect Categories</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_inspections}</div>
            <div class="metric-label">Total Inspections</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    st.subheader("Welcome to FabricIQ")

    st.markdown("""
    FabricIQ is an intelligent fabric inspection system that uses
    deep learning and VGG16 transfer learning to identify fabric
    defects from uploaded images.

    **Key capabilities:**

    - Automated fabric defect classification
    - Identification of nine fabric categories
    - Prediction confidence analysis
    - Inspection history tracking
    - Model performance information
    """)

    st.info("Navigate to Defect Inspection to analyze a fabric image.")


# --------------------------------------------------
# DEFECT INSPECTION
# --------------------------------------------------

elif page == "Defect Inspection":

    st.markdown(
        "<div class='main-title'>Fabric Defect Inspection</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Upload a fabric image to identify possible defects.</div>",
        unsafe_allow_html=True
    )

    st.write("")

    uploaded_file = st.file_uploader(
        "Upload Fabric Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        current_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        if current_hash != st.session_state.image_hash:
            st.session_state.image_hash = current_hash
            st.session_state.inspection_result = None

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

        with col2:
            st.subheader("Inspection")

            st.markdown("""
            <div class="section-card">
                <p>Image uploaded successfully.</p>
                <p>Click the button below to start defect classification.</p>
            </div>
            """, unsafe_allow_html=True)

            analyze = st.button(
                "🔍 Analyze Fabric",
                use_container_width=True
            )

            if analyze:

                try:
                    with st.spinner("Analyzing fabric image..."):

                        model = get_model()

                        from prediction import predict_fabric_defect

                        defect, confidence, probabilities = (
                            predict_fabric_defect(image, model)
                        )

                        st.session_state.inspection_result = {
                            "defect": defect,
                            "confidence": confidence,
                            "probabilities": probabilities
                        }

                        save_history(
                            defect,
                            confidence,
                            uploaded_file.name
                        )

                except Exception as e:
                    st.error("Unable to perform prediction.")
                    st.exception(e)

        result = st.session_state.inspection_result

        if result is not None:

            st.divider()
            st.subheader("Inspection Result")

            defect = result["defect"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Predicted Category**")

                if defect.lower() == "defect free":
                    st.success(f"✅ {defect}")
                else:
                    st.error(f"⚠️ {defect}")

            with col2:
                st.markdown("**Prediction Confidence**")
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            st.progress(
                max(0.0, min(1.0, confidence / 100))
            )

            st.divider()

            st.subheader("Class Probability Analysis")

            probability_df = pd.DataFrame({
                "Fabric Category": CLASS_NAMES,
                "Probability (%)": [
                    round(float(value) * 100, 2)
                    for value in probabilities
                ]
            }).sort_values(
                "Probability (%)",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Fabric Category")
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

    else:
        st.info("Please upload a fabric image to begin inspection.")


# --------------------------------------------------
# INSPECTION HISTORY
# --------------------------------------------------

elif page == "Inspection History":

    st.markdown(
        "<div class='main-title'>Inspection History</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Review previously analyzed fabric images.</div>",
        unsafe_allow_html=True
    )

    st.write("")

    history = load_history()

    if not history.empty:

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Inspections", len(history))

        with col2:
            if "Predicted Class" in history.columns:
                defect_count = (
                    history["Predicted Class"].str.lower()
                    != "defect free"
                ).sum()
                st.metric("Defects Detected", int(defect_count))

        st.divider()

        st.dataframe(
            history.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Inspection History",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="fabric_inspection_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Inspection History"):
            os.remove(HISTORY_PATH)
            st.success("Inspection history cleared.")
            st.rerun()

    else:
        st.info("No inspection records available yet.")


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.markdown(
        "<div class='main-title'>Model Information</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Technical details of the fabric classification model.</div>",
        unsafe_allow_html=True
    )

    st.write("")

    st.subheader("Model Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3 style="color:#60A5FA;">Architecture</h3>
            <p>VGG16 Transfer Learning</p>
            <h3 style="color:#60A5FA;">Input Image Size</h3>
            <p>224 × 224 RGB</p>
            <h3 style="color:#60A5FA;">Framework</h3>
            <p>TensorFlow / Keras</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="section-card">
            <h3 style="color:#60A5FA;">Test Accuracy</h3>
            <p>{MODEL_ACCURACY}%</p>
            <h3 style="color:#60A5FA;">Model Type</h3>
            <p>Image Classification</p>
            <h3 style="color:#60A5FA;">Model Name</h3>
            <p>{MODEL_NAME}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Supported Fabric Categories")

    for i, category in enumerate(CLASS_NAMES, start=1):
        st.write(f"{i}. {category}")

    st.divider()

    if os.path.exists(MODEL_PATH):
        st.success("Trained model file found.")
    else:
        st.error("Trained model file not found. Check the models folder.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94A3B8; padding:10px;">
        FabricIQ | Intelligent Fabric Inspection
        <br>
        Powered by VGG16 Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)
