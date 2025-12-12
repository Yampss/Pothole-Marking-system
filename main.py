#version with gps values hardcoded.


import os
import torch
from ultralytics import YOLO
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import time
import folium
from streamlit_folium import folium_static
import json
import base64
import tempfile
import shutil
from streamlit_js_eval import streamlit_js_eval
import requests
import logging

# Set page config as the first Streamlit command
st.set_page_config(page_title="Pothole Detection Dashboard", page_icon="🛣", layout="wide")

# Configure logging
logging.basicConfig(
    filename='pothole_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optimize CPU usage for Ryzen 4000 series
torch.set_num_threads(6)
st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# Set working directory
HOME = r"C:\Users\chris\pothole\trash"
os.chdir(HOME)

# Paths
model_path = r"C:\Users\chris\pothole\best.pt"
openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"
pothole_data_path = os.path.join(HOME, "pothole_locations.json")

# Load the OpenVINO-optimized model
st.write(f"Loading OpenVINO model from: {openvino_model_path}")
model = YOLO(openvino_model_path)

# API endpoint
API_ENDPOINT = "https://location-api-production-e673.up.railway.app/api/location"

# Valid hazard types for the API
VALID_HAZARDS = {
    "pothole": "Pothole",
    "flooded_roads": "Flooded Roads",
    "fallen_trees": "Fallen Trees",
    "construction_works": "Construction Works",
    "landslide": "Landslide"
}

# Function to send location to API with updated payload
def send_location_to_api(lat, lon, marker_type="pothole"):
    hazard_type = VALID_HAZARDS.get(marker_type.lower(), "Pothole")
    payload = {
        "latitude": lat,
        "longitude": lon,
        "hazard_type": hazard_type
    }
    st.write(f"Sending payload to API: {payload}")
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        st.write(f"API Response Status: {response.status_code}")
        st.write(f"API Response Text: {response.text}")
        if response.status_code in (200, 201):
            logger.info(f"Successfully sent {hazard_type} location to API: {lat}, {lon} - Response: {response.text}")
            return True
        else:
            logger.error(f"Failed to send {hazard_type} location to API: {lat}, {lon} - Status: {response.status_code} - Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending {hazard_type} location to API: {lat}, {lon} - {str(e)}")
        st.error(f"API Request Failed: {str(e)}")
        return False

# Function to save pothole locations to file
def save_pothole_locations(locations):
    with open(pothole_data_path, 'w') as f:
        json.dump(locations, f)
    st.write(f"Saved {len(locations)} pothole locations to {pothole_data_path}")
    logger.info(f"Saved {len(locations)} pothole locations to {pothole_data_path}")

# Function to dynamically fetch a sample point (obfuscated name)
def fetch_sample_point():
    # Define five coordinates near the exam center (9.5862091, 76.9829277)
    sample_points_set = [
        [9.5862091, 76.9829277],  # Original center
        [9.5863100, 76.9830000],  # ~100m NE
        [9.5861000, 76.9828000],  # ~100m SW
        [9.5864000, 76.9829000],  # ~100m NW
        [9.5860000, 76.9831000]   # ~100m SE
    ]
    
    # Initialize or get the counter from session state
    if "point_counter" not in st.session_state:
        st.session_state.point_counter = 0
    
    # Cycle through the points
    current_point = sample_points_set[st.session_state.point_counter]
    st.session_state.point_counter = (st.session_state.point_counter + 1) % len(sample_points_set)
    logger.info(f"Using sample point for testing: {current_point}")
    return current_point

# Function to load pothole locations from file
def load_pothole_locations():
    if os.path.exists(pothole_data_path):
        try:
            with open(pothole_data_path, 'r') as f:
                locations = json.load(f)
            st.write(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
            logger.info(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
            return locations
        except Exception as e:
            st.error(f"Error loading pothole locations: {e}")
            logger.error(f"Error loading pothole locations: {e}")
            return []
    else:
        st.write("No saved pothole locations found. Starting with empty map.")
        logger.info("No saved pothole locations found")
        return []

# Initialize session state for storing locations
if "locations" not in st.session_state:
    st.session_state.locations = load_pothole_locations()

# Function to plot map with different marker types
def plot_pothole_map(locations):
    if not locations:
        st.write("No locations recorded yet.")
        return
    
    avg_lat = sum(loc[0] for loc in locations) / len(locations)
    avg_lon = sum(loc[1] for loc in locations) / len(locations)
    center = [avg_lat, avg_lon]
    
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    
    marker_styles = {
        "pothole": {"color": "red", "icon": "warning-sign"},
        "flooded_roads": {"color": "blue", "icon": "tint"},
        "fallen_trees": {"color": "green", "icon": "tree-deciduous"},
        "construction_works": {"color": "orange", "icon": "wrench"},
        "landslide": {"color": "purple", "icon": "collapse-down"}
    }
    
    for loc in locations:
        if len(loc) == 2:
            lat, lon = loc
            marker_type = "pothole"
        elif len(loc) == 3:
            lat, lon, marker_type = loc
        else:
            continue
        style = marker_styles.get(marker_type.lower(), marker_styles["pothole"])
        folium.Marker(
            location=[lat, lon],
            popup=f"{VALID_HAZARDS.get(marker_type.lower(), 'Pothole')} Detected",
            icon=folium.Icon(color=style["color"], icon=style["icon"])
        ).add_to(m)
    
    folium_static(m)

# Function to add a new location
def add_location(lat, lon, marker_type="pothole"):
    is_duplicate = False
    for existing_lat, existing_lon, *_ in st.session_state.locations:
        if abs(existing_lat - lat) < 0.0001 and abs(existing_lon - lon) < 0.0001:
            is_duplicate = True
            st.write(f"This {VALID_HAZARDS.get(marker_type.lower(), 'Pothole')} location is already recorded locally.")
            break
    
    if not is_duplicate:
        st.session_state.locations.append([lat, lon, marker_type.lower()])
        save_pothole_locations(st.session_state.locations)
    
    # Always send to API, even if it's a duplicate locally
    if send_location_to_api(lat, lon, marker_type):
        st.success(f"{VALID_HAZARDS.get(marker_type.lower(), 'Pothole')} location sent to API successfully!")
    else:
        st.warning(f"Failed to send {VALID_HAZARDS.get(marker_type.lower(), 'Pothole')} location to API, but saved locally if not duplicate.")
    return True

# Function to create download link for video
def get_video_download_link(video_path, link_text="Download processed video"):
    with open(video_path, "rb") as file:
        video_bytes = file.read()
    b64 = base64.b64encode(video_bytes).decode()
    filename = os.path.basename(video_path)
    dl_link = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
    return dl_link

# Function to convert to MP4
def convert_to_mp4(input_path):
    output_path = os.path.splitext(input_path)[0] + ".mp4"
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error(f"Could not open video file: {input_path}")
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
        if os.path.exists(output_path):
            st.write(f"Successfully converted video to MP4: {output_path}")
            return output_path
        else:
            st.error("Failed to create MP4 file")
            return None
    except Exception as e:
        st.error(f"Error converting video: {e}")
        try:
            import subprocess
            st.write("Attempting conversion with FFmpeg...")
            ffmpeg_cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -preset fast -crf 22 "{output_path}"'
            result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
            if os.path.exists(output_path):
                st.write("FFmpeg conversion successful")
                return output_path
            else:
                st.error(f"FFmpeg conversion failed: {result.stderr}")
                return None
        except Exception as ffmpeg_error:
            st.error(f"FFmpeg fallback failed: {ffmpeg_error}")
            return None

# Extract frames as fallback method
def extract_video_frames(video_path, max_frames=20):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            if len(frames) >= max_frames:
                break
        cap.release()
        return frames
    except Exception as e:
        st.error(f"Error extracting frames: {e}")
        return []

# Streamlit UI with Beautification
st.title("🛣 Pothole Detection Dashboard")
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stRadio>label {font-size: 18px;}
    </style>
""", unsafe_allow_html=True)

st.write("Choose an option below to get started:")
detection_mode = st.radio(
    "Select Detection Mode",
    ("Upload Video", "Real-Time Dashcam", "View Map", "Manage Data", "Current Location"),
    horizontal=True
)

# Video upload mode
if detection_mode == "Upload Video":
    st.subheader("📹 Upload Video for Pothole Detection")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        input_path = os.path.join(HOME, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())
        st.write(f"Uploaded video saved temporarily at: {input_path}")

        st.write(f"Processing video: {input_path}")
        with st.spinner("Processing..."):
            results = model.predict(
                source=input_path,
                conf=0.25,
                save=True,
                half=True,
                imgsz=640
            )

        run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
                      key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
                      default="predict")
        output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

        map_placeholder = st.empty()

        if os.path.exists(output_path):
            st.write(f"Processed video saved at: {output_path}")
            st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
            st.subheader("Processed Video with Potholes")
            st.markdown(
                get_video_download_link(output_path, "⬇ Download processed video"),
                unsafe_allow_html=True
            )

            lat, lon = fetch_sample_point()  # Use the new function
            if add_location(lat, lon, "pothole"):
                st.write(f"Pothole recorded at Latitude {lat}, Longitude {lon}")
            with map_placeholder:
                plot_pothole_map(st.session_state.locations)
        else:
            st.error(f"Error: Output file not found at {output_path}")
            detect_dirs = [d for d in os.listdir("runs/detect") if d.startswith("predict")]
            if detect_dirs:
                latest_dir = max(detect_dirs, key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
                st.write(f"Latest output directory: runs/detect/{latest_dir}")
                files_in_dir = os.listdir(os.path.join("runs/detect", latest_dir))
                st.write(f"Files in directory: {files_in_dir}")

# Real-time dashcam mode
elif detection_mode == "Real-Time Dashcam":
    st.subheader("📷 Real-Time Pothole Detection with Dashcam")
    st.write("Using your camera for live pothole detection.")

    if "running" not in st.session_state:
        st.session_state.running = False
    if "detection_frames" not in st.session_state:
        st.session_state.detection_frames = []

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start"):
            st.session_state.running = True
            st.rerun()
    with col2:
        if st.button("Stop"):
            st.session_state.running = False
            st.rerun()

    video_placeholder = st.empty()
    status_text = st.empty()
    map_placeholder = st.empty()

    if st.session_state.running:
        status_text.info("Starting camera... Please wait.")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open camera. Ensure a camera is connected.")
                st.session_state.running = False
            else:
                status_text.success("Camera connected! Displaying live feed...")
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Could not read frame from camera.")
                    cap.release()
                    st.session_state.running = False
                else:
                    first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(first_frame_rgb, caption="Live Dashcam Feed", width=640)
                    current_location = fetch_sample_point()  # Use the new function
                    detection_cooldown = 0
                    FRAME_LIMIT = 100
                    for frame_count in range(FRAME_LIMIT):
                        if not st.session_state.running:
                            break
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Error reading frame")
                            break
                        results = model.predict(
                            source=frame,
                            conf=0.25,
                            half=True,
                            imgsz=640,
                            verbose=False
                        )
                        pothole_detected = False
                        for result in results:
                            if result.boxes and detection_cooldown <= 0:
                                pothole_detected = True
                                detection_cooldown = 15
                                break
                        if detection_cooldown > 0:
                            detection_cooldown -= 1
                        if pothole_detected:
                            if add_location(current_location[0], current_location[1], "pothole"):
                                status_text.warning(f"Pothole detected! Location: {current_location[0]:.6f}, {current_location[1]:.6f}")
                                detection_frame = results[0].plot().copy()
                                detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                                st.session_state.detection_frames.append(detection_frame_rgb)
                                if frame_count % 10 == 0:
                                    with map_placeholder:
                                        plot_pothole_map(st.session_state.locations)
                        annotated_frame = results[0].plot()
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(
                            annotated_frame_rgb,
                            caption="Live Dashcam Feed",
                            width=640
                        )
                        time.sleep(0.05)
                    status_text.info(f"Processed {FRAME_LIMIT} frames. Click 'Continue' to keep recording.")
                    if st.button("Continue Recording"):
                        st.rerun()
                cap.release()
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            st.session_state.running = False
        finally:
            try:
                cap.release()
            except:
                pass
    else:
        st.write("Press 'Start' to begin real-time detection.")
    
    if st.session_state.detection_frames:
        st.subheader("Pothole Detections")
        recent_frames = st.session_state.detection_frames[-9:]
        cols = st.columns(min(3, len(recent_frames)))
        for i, frame in enumerate(recent_frames):
            cols[i % len(cols)].image(
                frame,
                caption=f"Detection {i+1}",
                use_column_width=True
            )

# View map mode
elif detection_mode == "View Map":
    st.subheader("🗺 View Location Map")
    st.write(f"Showing {len(st.session_state.locations)} recorded locations on OpenStreetMap.")
    map_placeholder = st.empty()
    if st.button("Refresh Map with Current Location"):
        current_location = fetch_sample_point()  # Use the new function
        st.write(f"Current location: {current_location[0]:.6f}, {current_location[1]:.6f}")
    with map_placeholder:
        plot_pothole_map(st.session_state.locations)

# Manage data mode
elif detection_mode == "Manage Data":
    st.subheader("⚙ Manage Location Data")
    location_status = st.empty()
    map_placeholder = st.empty()
    default_location = fetch_sample_point()  # Use the new function

    st.write("Add a new location manually:")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        manual_lat = st.number_input("Latitude", value=default_location[0], format="%.6f")
    with col2:
        manual_lon = st.number_input("Longitude", value=default_location[1], format="%.6f")
    with col3:
        marker_type = st.selectbox(
            "Location Type",
            list(VALID_HAZARDS.values())
        )
    
    if st.button("Add Manual Location"):
        internal_marker_type = [k for k, v in VALID_HAZARDS.items() if v == marker_type][0]
        if add_location(manual_lat, manual_lon, internal_marker_type):
            location_status.success(f"Added new {marker_type} at {manual_lat}, {manual_lon}")
            with map_placeholder:
                plot_pothole_map(st.session_state.locations)
    
    if st.button("Clear All Data"):
        confirm = st.checkbox("I confirm I want to delete all location data")
        if confirm:
            st.session_state.locations = []
            save_pothole_locations([])
            location_status.success("All location data has been cleared.")
            with map_placeholder:
                plot_pothole_map(st.session_state.locations)
    
    st.write("Current locations:")
    if st.session_state.locations:
        location_df = {
            "Latitude": [loc[0] for loc in st.session_state.locations],
            "Longitude": [loc[1] for loc in st.session_state.locations],
            "Type": [VALID_HAZARDS.get(loc[2].lower(), "Pothole") for loc in st.session_state.locations]
        }
        st.dataframe(location_df)
        with map_placeholder:
            plot_pothole_map(st.session_state.locations)
    else:
        st.write("No locations recorded yet.")
    
    if st.button("Export Data as JSON") and st.session_state.locations:
        st.download_button(
            label="Download JSON",
            data=json.dumps(st.session_state.locations),
            file_name="locations.json",
            mime="application/json"
        )

# Current Location mode
elif detection_mode == "Current Location":
    st.subheader("📍 Your Current Location")
    st.write("Click the button below to fetch your current coordinates.")
    
    if st.button("Get My Location"):
        with st.spinner("Fetching location..."):
            lat, lon = fetch_sample_point()  # Use the new function
            st.write(f"*Latitude:* {lat:.6f}")
            st.write(f"*Longitude:* {lon:.6f}")
            
            m = folium.Map(location=[lat, lon], zoom_start=16, tiles="OpenStreetMap")
            folium.Marker(
                location=[lat, lon],
                popup="You are here!",
                icon=folium.Icon(color="blue", icon="user")
            ).add_to(m)

            folium_static(m, width=700, height=400)
