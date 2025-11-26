import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
import base64

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(
    page_title="PhysiSense VLR – Advanced Physics Lab Detector",
    layout="wide",
    page_icon="🧪"
)

# -------------------------------------
# Custom Global Styling (Premium UI)
# -------------------------------------
st.markdown("""
<style>
/* Global Styling */
body {
    background: radial-gradient(circle at top, #141414, #0a0a0a 60%);
    color: #eaeaea;
    font-family: 'Inter', sans-serif;
}

/* Premium Title */
.big-title {
    font-size: 3rem;
    text-align: center;
    font-weight: 800;
    background: linear-gradient(90deg, #4caf50, #b4ffcb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 15px;
}

/* Top Navbar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 25px;
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}

.nav-item {
    font-size: 18px;
    font-weight: 600;
    color: #80e27e;
    transition: 0.3s;
}
.nav-item:hover {
    color: white;
}

/* Container Card */
.container-box {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    box-shadow: 0 4px 40px rgba(0,0,0,0.45);
}

/* Buttons */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #4caf50, #6fff9c);
    color: black;
    font-weight: 700;
    border-radius: 12px;
    height: 3rem;
    font-size: 17px;
}

/* Detected Items */
.result-item {
    background: rgba(0,255,127,0.15);
    padding: 12px;
    margin: 6px 0;
    border-left: 6px solid #00e676;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 500;
}

/* Reason Box */
.reason-box {
    background: rgba(255,255,0,0.12);
    padding: 18px;
    margin-top: 10px;
    border-left: 5px solid #ffeb3b;
    border-radius: 12px;
}

/* Accordion Enhancement */
details {
    background: rgba(255,255,255,0.04);
    padding: 10px 14px;
    border-radius: 10px;
    margin-top: 8px;
    border: 1px solid rgba(255,255,255,0.1);
}
summary {
    cursor: pointer;
    font-size: 18px;
    font-weight: 600;
    color: #baffc9;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------
# Load YOLO Model
# -------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------------
# Class Labels
# -------------------------------------
class_names = [
    'AC-Ammeter', 'Brass-Scale-Weights', 'Burette-Stand', 'DC-Ammeter', 'DC-Power-Supply',
    'Deflection-Magnetometer', 'Deflection-Magnetometer-Power-supply', 'Helical-Extension-Spring',
    'Lens', 'Meldes-Apparatus', 'Meldes-Apparatus-weight', 'Micrometer-Screw-Gauge', 'Multimeter',
    'Pendulum-Clamp', 'Retort-Stand', 'Rubber-Mallet-Hammer', 'Spherometer', 'Stopwatch',
    'Vernier-Caliper', 'Weight-carrier'
]

# -------------------------------------
# Gemini Reasoning API
# -------------------------------------
def gemini_reason(instrument):
    API_KEY = "AIzaSyCofn-wO6jqJT40l-SqKpL4fXfEj3Y7TN0"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Explain what '{instrument}' is, how it works, and its purpose in physics laboratories. Write a clean, concise 3-line paragraph."
                    }
                ]
            }
        ]
    }

    resp = requests.post(url, json=body)
    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "⚠ Gemini reasoning unavailable."

# -------------------------------------
# Top Navigation Bar
# -------------------------------------
st.markdown("""
<div class='navbar'>
    <div class='nav-item'>🧪 PhysiSense VLR</div>
    <div class='nav-item'>Physics Lab Detector</div>
    <div class='nav-item'>AI Powered</div>
</div>
""", unsafe_allow_html=True)

# Main Header
st.markdown("<div class='big-title'>Advanced Physics Experiment Instrument Detector</div>", unsafe_allow_html=True)

# -------------------------------------
# Upload Section
# -------------------------------------
st.markdown("### 📤 Upload a Physics Instrument Image")
uploaded = st.file_uploader("Upload here", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    colA, colB = st.columns([1.3, 1])

    # LEFT: Uploaded Image
    with colA:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("### 📸 Uploaded Image")
        resized = img.resize((800,600))  # width, height
        st.image(resized)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT: Detection Panel
    with colB:
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        st.markdown("### 🧪 Instrument Detection Panel")

        detect = st.button("🔍 Start Detection", use_container_width=True)

        if detect:
            results = model(img)
            boxes = results[0].boxes

            if not boxes:
                st.error("❌ No instruments detected!")
            else:
                st.success("✔ Instrument(s) Detected")

                detected = []
                for box in boxes:
                    cid = int(box.cls[0])
                    detected.append(class_names[cid])

                detected = list(set(detected))

                # Display detected
                st.markdown("### 🧾 Detected Instruments:")
                for item in detected:
                    st.markdown(f"<div class='result-item'>🔹 {item}</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 🤖 AI Reasoning")

                for item in detected:
                    exp = gemini_reason(item)

                    st.markdown(f"""
                    <details>
                        <summary>🟢 {item}</summary>
                        <div class='reason-box'>{exp}</div>
                    </details>
                    """, unsafe_allow_html=True)

        # Clear button
        if st.button("🗑 Delete Image", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

#streamlit run main.py
