import streamlit as st
import requests
import os

# Configuration (Change port to 8000 if Uvicorn is running on 8000 instead of 8080)
BACKEND_URL = "http://127.0.0.1:8080" 

st.set_page_config(
    page_title="Video Analytics System",
    page_icon="🎥",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Space Grotesk', sans-serif;
    }
    .frame-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .badge-time {
        background-color: #1f6feb;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        margin-right: 5px;
    }
    .badge-score {
        background-color: #238636;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎥 Video Analytics System")
st.markdown("Upload a video and search its contents using natural language.")

# ==========================================
# SIDEBAR: UPLOAD & STATE MANAGEMENT
# ==========================================
with st.sidebar:
    st.header("1. Upload Video")
    uploaded_file = st.file_uploader("Choose an MP4 video", type=["mp4", "avi", "mov"])
    
    if st.button("Upload & Process Video"):
        if uploaded_file is not None:
            with st.spinner("Processing video..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # CRITICAL FIX: Save the ID to session state so it survives refreshes
                        st.session_state["video_id"] = data["video_id"]
                        st.success("✅ Video Processed Successfully!")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Backend Connection Error: {e}")
        else:
            st.warning("⚠️ Please select a file first.")
            
    st.divider()
    
    # VISUAL DEBUGGER: See if Streamlit remembers your video
    st.subheader("System State")
    if "video_id" in st.session_state:
        st.info(f"🟢 Active Video:\n**{st.session_state['video_id']}**")
    else:
        st.warning("🔴 No Active Video. Please upload.")

# ==========================================
# MAIN AREA: SEMANTIC SEARCH
# ==========================================
st.header("2. Semantic Video Search")
query_text = st.text_input("What are you looking for?", placeholder="e.g., 'a red car', 'someone walking'")
threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

# This block is perfectly indented. Do not change the spacing!
if st.button("Search Matching Frames"):
    # 1. Grab the ID safely from memory
    current_video_id = st.session_state.get("video_id")
    
    # 2. Sequential Safety Checks
    if not query_text:
        st.warning("⚠️ Please enter a search query first.")
    elif not current_video_id:
        st.error("⚠️ No video detected! Please upload a video in the sidebar first.")
    else:
        # 3. Proceed with API request
        with st.spinner("Searching frames..."):
            payload = {
                "query": query_text,
                "threshold": threshold,
                "video_id": current_video_id
            }
            try:
                response = requests.post(f"{BACKEND_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    st.success(data.get("message", "Search completed."))
                    
                    if results:
                        cols_per_row = 3
                        for i in range(0, len(results), cols_per_row):
                            row_cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                idx = i + j
                                if idx < len(results):
                                    res = results[idx]
                                    with row_cols[j]:
                                        st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                                        if os.path.exists(res["frame_path"]):
                                            st.image(res["frame_path"], use_container_width=True)
                                        else:
                                            st.warning(f"Image missing: {res['frame_path']}")
                                            
                                        st.markdown(
                                            f'<div style="margin-top:8px;">'
                                            f'<span class="badge-time">⏱️ {res["timestamp"]}s</span>'
                                            f'<span class="badge-score">📊 Conf: {res["similarity_score"]:.2f}</span>'
                                            f'</div>', 
                                            unsafe_allow_html=True
                                        )
                                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No matching frames found above the threshold. Try lowering the threshold or changing the query.")
                else:
                    st.error(f"Backend Error: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")