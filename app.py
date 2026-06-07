# app.py - Streamlit Vehicle Detection System (2 Tabs - Simple & Clean)
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CYBER GRADIENT CSS
# =====================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 50%, #0d0d0d 100%);
    }
    
    .main > div {
        padding: 0.2rem !important;
    }
    
    h1 {
        font-size: 1.2rem !important;
        margin: 0 !important;
        text-align: center;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #00ff41, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cyber cards */
    .cyber-card {
        background: rgba(10, 10, 10, 0.8);
        border: 1px solid #00ff41;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        box-shadow: 0 0 10px rgba(0,255,65,0.2);
    }
    
    .card-title {
        color: #00ff41;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        text-align: center;
    }
    
    /* Small frames */
    .small-frame {
        position: relative;
        width: 100%;
        max-width: 280px;
        margin: 0 auto;
        padding-bottom: 75%;
        background: #0a0a0a;
        border: 2px solid #00ff41;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .small-frame img {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    
    .frame-label {
        text-align: center;
        color: #ff00ff;
        font-size: 0.65rem;
        margin-top: 0.2rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff00ff, #00ff41);
        color: #0a0a0a;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.3rem 1rem;
        border: none;
        border-radius: 6px;
        width: 100%;
    }
    
    /* Metrics */
    .metric-box {
        background: rgba(0,0,0,0.7);
        border: 1px solid #ff00ff;
        border-radius: 6px;
        padding: 0.3rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #00ff41;
    }
    .metric-label {
        font-size: 0.6rem;
        color: #ff00ff;
    }
    
    /* Table */
    .dataframe {
        background: #0a0a0a !important;
        border: 1px solid #00ff41 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(0,0,0,0.6);
        border-radius: 8px;
        padding: 0.3rem;
        border: 1px solid #ff00ff;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.75rem;
        padding: 0.2rem 1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff00ff, #00ff41);
        color: #0a0a0a;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 1px solid #00ff41;
        border-radius: 6px;
        background: rgba(0,0,0,0.6);
        margin-bottom: 0.5rem;
    }
    
    /* Slider */
    .stSlider label {
        color: #00ff41 !important;
        font-size: 0.7rem !important;
    }
    
    .status-text {
        font-size: 0.6rem;
        color: #00ff41;
        text-align: center;
        margin: 0.3rem 0;
    }
    
    hr {
        border-color: #ff00ff;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    try:
        possible_paths = [
            Path("train/weights/best.pt"),
            Path("best.pt"),
            Path(r"C:\Users\abc\OneDrive\Desktop\Vehicle Detection System\train\weights\best.pt")
        ]
        model_path = next((p for p in possible_paths if p.exists()), None)
        
        if model_path:
            model = YOLO(str(model_path))
            model.fuse()
            return model
        return None
    except:
        return None

model = load_model()

# =====================================================
# DETECTION FUNCTIONS
# =====================================================
def detect_image(image, conf_thresh):
    if image is None or model is None:
        return None, {}, 0, 0
    
    try:
        start = time.time()
        results = model(image, conf=conf_thresh, verbose=False)
        result = results[0]
        
        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        counts = {}
        if result.boxes:
            for box in result.boxes:
                cls = int(box.cls[0])
                name = result.names[cls]
                counts[name] = counts.get(name, 0) + 1
        
        fps = round(1 / (time.time() - start), 1)
        return annotated, counts, fps, sum(counts.values())
    except:
        return None, {}, 0, 0

def detect_video(video_path, conf_thresh, progress_bar):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = 320
    height = 240
    
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        small_frame = cv2.resize(frame, (320, 240))
        results = model(small_frame, conf=conf_thresh, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    return output_path, frame_count

# =====================================================
# MAIN UI
# =====================================================

# Header
st.markdown('<h1 class="gradient-text">⚡ VEHICLE DETECTION SYSTEM ⚡</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#00ff41; font-size:0.7rem;">YOLOv8 | Real-Time Detection | Cyber Vision</p>', unsafe_allow_html=True)

# Global Confidence Slider
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    confidence = st.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05)

# Model Status
if model:
    st.markdown('<p class="status-text">🟢 SYSTEM ONLINE - Model Loaded Successfully</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="status-text">🔴 SYSTEM OFFLINE - Model not found. Please check best.pt</p>', unsafe_allow_html=True)

# =====================================================
# TABS
# =====================================================
tab1, tab2 = st.tabs(["📸 IMAGE DETECTOR", "🎬 VIDEO TRACKER"])

# =====================================================
# TAB 1: IMAGE DETECTOR
# =====================================================
with tab1:
    # Two columns for input/output
    col_left, col_right = st.columns(2)
    
    # LEFT COLUMN - Input
    with col_left:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">INPUT IMAGE</div>', unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if uploaded_image:
            img = Image.open(uploaded_image)
            img.thumbnail((250, 188))
            st.markdown('<div class="small-frame">', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-frame"><p style="text-align:center; color:#666; font-size:0.7rem; margin-top:35%;">No Image</p></div>', unsafe_allow_html=True)
        
        detect_btn = st.button("🔍 START DETECTION", key="img_detect")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # RIGHT COLUMN - Output
    with col_right:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">DETECTION RESULT</div>', unsafe_allow_html=True)
        
        if uploaded_image and detect_btn and model:
            image = Image.open(uploaded_image)
            img_np = np.array(image)
            annotated, counts, fps, total = detect_image(img_np, confidence)
            
            if annotated is not None:
                st.session_state['img_result'] = annotated
                st.session_state['img_counts'] = counts
                st.session_state['img_fps'] = fps
                st.session_state['img_total'] = total
        
        if 'img_result' in st.session_state:
            st.markdown('<div class="small-frame">', unsafe_allow_html=True)
            st.image(st.session_state['img_result'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-frame"><p style="text-align:center; color:#666; font-size:0.7rem; margin-top:35%;">Awaiting Detection</p></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analytics Section
    if 'img_result' in st.session_state and st.session_state.get('img_total', 0) > 0:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 DETECTION ANALYTICS</div>', unsafe_allow_html=True)
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{st.session_state.get('img_fps', 0)}</div>
                <div class="metric-label">FPS</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{st.session_state.get('img_total', 0)}</div>
                <div class="metric-label">TOTAL OBJECTS</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{len(st.session_state.get('img_counts', {}))}</div>
                <div class="metric-label">OBJECT CLASSES</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Object Table
        if st.session_state.get('img_counts'):
            df = pd.DataFrame(st.session_state['img_counts'].items(), columns=["OBJECT TYPE", "QUANTITY"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# TAB 2: VIDEO TRACKER
# =====================================================
with tab2:
    # Two columns for input/output
    col_left, col_right = st.columns(2)
    
    # LEFT COLUMN - Input
    with col_left:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">INPUT VIDEO</div>', unsafe_allow_html=True)
        
        uploaded_video = st.file_uploader("", type=['mp4', 'avi', 'mov'], label_visibility="collapsed", key="video_upload")
        
        if uploaded_video:
            st.markdown(f'<p style="color:#00ff41; font-size:0.6rem; text-align:center;">📹 {uploaded_video.name[:20]}</p>', unsafe_allow_html=True)
            st.markdown('<div class="small-frame"><p style="text-align:center; color:#00ff41; font-size:0.6rem; margin-top:35%;">Video Ready</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-frame"><p style="text-align:center; color:#666; font-size:0.7rem; margin-top:35%;">No Video</p></div>', unsafe_allow_html=True)
        
        process_btn = st.button("🎬 START TRACKING", key="vid_process")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # RIGHT COLUMN - Output
    with col_right:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">TRACKING RESULT</div>', unsafe_allow_html=True)
        
        if uploaded_video and process_btn and model:
            with st.spinner("Processing video..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name
                
                progress_bar = st.progress(0)
                output_path, frame_count = detect_video(video_path, confidence, progress_bar)
                
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                    st.video(video_bytes)
                
                st.markdown(f'<p class="status-text">✅ Processed {frame_count} frames</p>', unsafe_allow_html=True)
                st.session_state['video_processed'] = True
        else:
            st.markdown('<div class="small-frame"><p style="text-align:center; color:#666; font-size:0.7rem; margin-top:35%;">Awaiting Video</p></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Video Info
    if uploaded_video and process_btn:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">ℹ️ PROCESSING INFO</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#00ff41; font-size:0.7rem;">✓ Detection completed with {confidence*100:.0f}% confidence threshold</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#ff00ff; font-size:0.7rem;">✓ Output video optimized for 320x240 resolution</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.6rem; padding:0.5rem;">
    Vehicle Detection System | Powered by YOLOv8 | Real-Time Detection
</div>
""", unsafe_allow_html=True)