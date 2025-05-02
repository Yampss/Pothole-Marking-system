# import os
# from ultralytics import YOLO

# # Paths configuration
# MODEL_PATH = r"C:\Users\chris\pothole\best.pt"
# VIDEO_PATH = r"C:\Users\chris\pothole\whatsapp.mp4"
# OUTPUT_DIR = r"C:\Users\chris\pothole\output"

# def main():
#     # Create output directory if it doesn't exist
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # Initialize YOLO model
#     try:
#         model = YOLO(MODEL_PATH)
#         print("Model loaded successfully")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return

#     # Run prediction
#     try:
#         print("Starting video processing...")
#         results = model.predict(
#             source=VIDEO_PATH,
#             conf=0.25,
#             save=True,
#             project=OUTPUT_DIR,
#             name="detection"
#         )
#         print("Video processing completed")
#         print(f"Results saved to {OUTPUT_DIR}/detection")
#     except Exception as e:
#         print(f"Error during prediction: {e}")

# if __name__ == "__main__":
#     main()
# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)  # Adjust to your CPU core count (e.g., 6 for Ryzen 5 4500U, 8 for Ryzen 7 4700U)
# print(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"  # Original trained YOLO model (for reference)
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"  # Exported OpenVINO model directory
# input_path = r"C:\Users\chris\pothole\WhatsApp Video 2025-03-20 at 08.25.05_87173fd8.mp4"  # Input video

# # Load the OpenVINO-optimized model
# print(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Run YOLO prediction with optimizations
# print(f"Processing video: {input_path}")
# results = model.predict(
#     source=input_path,
#     conf=0.25,
#     save=True,           # Save output as .avi
#     half=True,          # Use FP16 for faster inference (if supported)
#     imgsz=640           # Match the exported model's input size (640x640)
# )

# # Get the output path
# run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#               key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#               default="predict")
# output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

# # Verify and report output
# if os.path.exists(output_path):
#     print(f"Processed video saved at: {output_path}")
#     print(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
# else:
#     print(f"Error: Output file not found at {output_path}")




# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)  # Adjust to your CPU core count (e.g., 6 for Ryzen 5 4500U, 8 for Ryzen 7 4700U)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # # Paths
# # model_path = r"C:\Users\chris\pothole\best.pt"  # Original trained YOLO model (for reference)
# # openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"  # Exported OpenVINO model directory
# # Paths
# model_path = r"C:\Users\jithu\best.pt"  # Original trained YOLO model (for reference)
# openvino_model_path = r"C:\Users\jithu\best_openvino_model"  # Exported OpenVINO model directory
# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Streamlit UI
# st.title("Pothole Detection with YOLO")
# st.write("Upload a video to detect potholes using the OpenVINO-optimized model.")

# # Video upload
# uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# if uploaded_file is not None:
#     # Save the uploaded video temporarily
#     input_path = os.path.join(HOME, uploaded_file.name)
#     with open(input_path, "wb") as f:
#         f.write(uploaded_file.read())
#     st.write(f"Uploaded video saved temporarily at: {input_path}")

#     # Run YOLO prediction with optimizations
#     st.write(f"Processing video: {input_path}")
#     with st.spinner("Processing..."):
#         results = model.predict(
#             source=input_path,
#             conf=0.25,
#             save=True,           # Save output as .avi
#             half=True,          # Use FP16 for faster inference (if supported)
#             imgsz=640           # Match the exported model's input size (640x640)
#         )

#     # Get the output path
#     run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                   key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                   default="predict")
#     output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#     # Verify and display output
#     if os.path.exists(output_path):
#         st.write(f"Processed video saved at: {output_path}")
#         st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        
#         # Display the processed video in Streamlit
#         st.subheader("Processed Video with Potholes")
#         st.video(output_path)
#     else:
#         st.error(f"Error: Output file not found at {output_path}")



# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)  # Adjust to your CPU core count (e.g., 6 for Ryzen 5 4500U, 8 for Ryzen 7 4700U)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"  # Original trained YOLO model (for reference)
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"  # Exported OpenVINO model directory

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Streamlit UI
# st.title("Pothole Detection with YOLO")
# st.write("Choose an option: Upload a video or use your dashcam for real-time detection.")

# # Option selection
# detection_mode = st.radio("Select Detection Mode", ("Upload Video", "Real-Time Dashcam"))

# # Video upload mode
# if detection_mode == "Upload Video":
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

#     if uploaded_file is not None:
#         # Save the uploaded video temporarily
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")

#         # Run YOLO prediction with optimizations
#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,           # Save output as .avi
#                 half=True,          # Use FP16 for faster inference (if supported)
#                 imgsz=640           # Match the exported model's input size (640x640)
#             )

#         # Get the output path
#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#         # Verify and display output
#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
            
#             # Display the processed video in Streamlit
#             st.subheader("Processed Video with Potholes")
#             st.video(output_path)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")

# # Real-time dashcam mode
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")

#     # Initialize session state for running the camera
#     if "running" not in st.session_state:
#         st.session_state.running = False

#     # Start and Stop buttons
#     col1, col2 = st.columns(2)
#     with col1:
#         start_button = st.button("Start", key="start_button")
#     with col2:
#         stop_button = st.button("Stop", key="stop_button")

#     # Placeholder for video feed
#     video_placeholder = st.empty()

#     # Handle start/stop logic
#     if start_button:
#         st.session_state.running = True
#     if stop_button:
#         st.session_state.running = False

#     # Open the default camera (index 0, adjust if needed for your dashcam)
#     if st.session_state.running:
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error("Error: Could not open camera. Ensure a camera is connected and try a different index if needed.")
#             st.session_state.running = False
#         else:
#             st.write("Camera opened successfully. Starting real-time detection...")

#             while st.session_state.running and cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     break

#                 # Run YOLO prediction on the frame
#                 results = model.predict(
#                     source=frame,
#                     conf=0.25,
#                     half=True,
#                     imgsz=640,
#                     verbose=False  # Suppress excessive logging
#                 )

#                 # Get the annotated frame from results
#                 annotated_frame = results[0].plot()  # Plot detections on the frame

#                 # Convert BGR (OpenCV) to RGB (Streamlit compatibility)
#                 annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

#                 # Display the frame in Streamlit
#                 video_placeholder.image(annotated_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)

#                 # Small delay to control frame rate (e.g., ~30 FPS)
#                 time.sleep(0.033)

#             # Release the camera when done
#             cap.release()
#             st.write("Camera feed stopped.")
#             st.session_state.running = False

#     elif not st.session_state.running:
#         st.write("Press 'Start' to begin real-time detection.")

        
# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time
# import folium
# from streamlit_folium import folium_static
# import streamlit.components.v1 as components

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Streamlit UI
# st.title("Pothole Detection with YOLO")
# st.write("Choose an option: Upload a video, use your dashcam, or plot your current location as a pothole on OpenStreetMap.")

# # Option selection
# detection_mode = st.radio("Select Detection Mode", ("Upload Video", "Real-Time Dashcam", "Plot Current Location"))

# # Video upload mode (unchanged)
# if detection_mode == "Upload Video":
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")
#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,
#                 half=True,
#                 imgsz=640
#             )
#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")
#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
#             st.subheader("Processed Video with Potholes")
#             st.video(output_path)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")

# # Real-time dashcam mode (unchanged)
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")
#     if "running" not in st.session_state:
#         st.session_state.running = False
#     col1, col2 = st.columns(2)
#     with col1:
#         start_button = st.button("Start", key="start_button")
#     with col2:
#         stop_button = st.button("Stop", key="stop_button")
#     video_placeholder = st.empty()
#     if start_button:
#         st.session_state.running = True
#     if stop_button:
#         st.session_state.running = False
#     if st.session_state.running:
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error("Error: Could not open camera. Ensure a camera is connected.")
#             st.session_state.running = False
#         else:
#             st.write("Camera opened successfully. Starting real-time detection...")
#             while st.session_state.running and cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     break
#                 results = model.predict(
#                     source=frame,
#                     conf=0.25,
#                     half=True,
#                     imgsz=320,
#                     verbose=False
#                 )
#                 annotated_frame = results[0].plot()
#                 annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                 video_placeholder.image(annotated_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
#                 time.sleep(0.066)
#             cap.release()
#             st.write("Camera feed stopped.")
#             st.session_state.running = False
#     elif not st.session_state.running:
#         st.write("Press 'Start' to begin real-time detection.")

# # Plot current location with browser geolocation
# elif detection_mode == "Plot Current Location":
#     st.subheader("Plot Current Location as Pothole on OpenStreetMap")
#     st.write("Requesting your precise location via browser geolocation...")

#     # JavaScript to get geolocation and send it back to Streamlit
#     geolocation_script = """
#     <script>
#     function getLocation() {
#         if (navigator.geolocation) {
#             navigator.geolocation.getCurrentPosition(
#                 function(position) {
#                     const lat = position.coords.latitude;
#                     const lon = position.coords.longitude;
#                     document.getElementById("lat").value = lat;
#                     document.getElementById("lon").value = lon;
#                     document.getElementById("locationForm").submit();
#                 },
#                 function(error) {
#                     alert("Error getting location: " + error.message);
#                 }
#             );
#         } else {
#             alert("Geolocation is not supported by this browser.");
#         }
#     }
#     window.onload = getLocation;
#     </script>
#     <form id="locationForm" action="" method="post">
#         <input type="hidden" id="lat" name="lat">
#         <input type="hidden" id="lon" name="lon">
#     </form>
#     """

#     # Render the JavaScript in Streamlit and capture the result
#     components.html(geolocation_script, height=0)

#     # Check if location data is available in session state
#     if "lat" not in st.session_state or "lon" not in st.session_state:
#         st.session_state.lat = None
#         st.session_state.lon = None

#     # Button to manually trigger location request (in case auto-trigger fails)
#     if st.button("Request Location Permission"):
#         components.html(geolocation_script, height=0)

#     # Check if query parameters contain lat/lon (passed back from JS)
#     query_params = st.experimental_get_query_params()
#     if "lat" in query_params and "lon" in query_params:
#         st.session_state.lat = float(query_params["lat"][0])
#         st.session_state.lon = float(query_params["lon"][0])

#     # Plot the map if location is available
#     if st.session_state.lat and st.session_state.lon:
#         lat, lon = st.session_state.lat, st.session_state.lon
#         st.write(f"Detected Location: Latitude {lat}, Longitude {lon}")

#         # Create Folium map
#         m = folium.Map(location=[lat, lon], zoom_start=15, tiles="OpenStreetMap")
#         folium.Marker(
#             location=[lat, lon],
#             popup="Pothole Detected",
#             icon=folium.Icon(color="red", icon="exclamation-triangle")
#         ).add_to(m)
#         folium_static(m)
#         st.write("Your current location is marked as having a pothole on the map above.")
#     else:
#         st.write("Waiting for location permission... If prompted, please allow access in your browser.")





# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time
# import geocoder
# import folium
# from streamlit_folium import folium_static
# import json

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"
# # Path for storing pothole locations
# pothole_data_path = os.path.join(HOME, "pothole_locations.json")

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Function to save pothole locations to file
# def save_pothole_locations(locations):
#     with open(pothole_data_path, 'w') as f:
#         json.dump(locations, f)
#     st.write(f"Saved {len(locations)} pothole locations to {pothole_data_path}")

# # Function to load pothole locations from file
# def load_pothole_locations():
#     if os.path.exists(pothole_data_path):
#         try:
#             with open(pothole_data_path, 'r') as f:
#                 locations = json.load(f)
#             st.write(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             return locations
#         except Exception as e:
#             st.error(f"Error loading pothole locations: {e}")
#             return []
#     else:
#         st.write("No saved pothole locations found. Starting with empty map.")
#         return []

# # Initialize session state for storing pothole locations
# if "pothole_locations" not in st.session_state:
#     st.session_state.pothole_locations = load_pothole_locations()

# # Function to get current location
# def get_current_location():
#     try:
#         g = geocoder.ip('me')
#         if g.ok:
#             return g.latlng  # Returns [lat, lon]
#         else:
#             st.warning("Could not retrieve location. Using default location.")
#             return [37.7749, -122.4194]  # Default to San Francisco
#     except Exception as e:
#         st.error(f"Error retrieving location: {e}")
#         return [37.7749, -122.4194]

# # Function to plot pothole map
# def plot_pothole_map(locations):
#     if not locations:
#         st.write("No pothole locations recorded yet.")
#         return
    
#     # Use the first location to center the map, or calculate an average center
#     if len(locations) == 1:
#         center = locations[0]
#     else:
#         # Calculate the average lat/lon for centering the map
#         avg_lat = sum(loc[0] for loc in locations) / len(locations)
#         avg_lon = sum(loc[1] for loc in locations) / len(locations)
#         center = [avg_lat, avg_lon]
    
#     m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    
#     # Add markers for each pothole
#     for lat, lon in locations:
#         folium.Marker(
#             location=[lat, lon],
#             popup="Pothole Detected",
#             icon=folium.Icon(color="red", icon="warning-sign")
#         ).add_to(m)
    
#     folium_static(m)

# # Function to add a new pothole location
# def add_pothole_location(lat, lon):
#     # Check if this location is already recorded (within a small radius)
#     for existing_lat, existing_lon in st.session_state.pothole_locations:
#         # Simple distance check (approximate)
#         if abs(existing_lat - lat) < 0.0001 and abs(existing_lon - lon) < 0.0001:
#             st.write("This pothole location is already recorded.")
#             return False
    
#     # Add new location
#     st.session_state.pothole_locations.append([lat, lon])
#     # Save to persistent storage
#     save_pothole_locations(st.session_state.pothole_locations)
#     return True

# # Streamlit UI
# st.title("Pothole Detection with YOLO")
# st.write("Choose an option: Upload a video, use your dashcam, or view the pothole map.")

# # Option selection
# detection_mode = st.radio("Select Detection Mode", ("Upload Video", "Real-Time Dashcam", "View Pothole Map", "Manage Pothole Data"))

# # Video upload mode
# if detection_mode == "Upload Video":
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         # Save the uploaded video temporarily
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")

#         # Run YOLO prediction
#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,
#                 half=True,
#                 imgsz=640
#             )

#         # Get the output path
#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#         # Verify and display output
#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
#             st.subheader("Processed Video with Potholes")
#             st.video(output_path)

#             # Plot the current location as having a pothole
#             lat, lon = get_current_location()
#             if add_pothole_location(lat, lon):
#                 st.write(f"Pothole recorded at Latitude {lat}, Longitude {lon}")
#             plot_pothole_map(st.session_state.pothole_locations)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")

# # Real-time dashcam mode
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")

#     if "running" not in st.session_state:
#         st.session_state.running = False

#     col1, col2 = st.columns(2)
#     with col1:
#         start_button = st.button("Start", key="start_button")
#     with col2:
#         stop_button = st.button("Stop", key="stop_button")

#     video_placeholder = st.empty()
#     map_placeholder = st.empty()

#     if start_button:
#         st.session_state.running = True
#     if stop_button:
#         st.session_state.running = False

#     if st.session_state.running:
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error("Error: Could not open camera. Ensure a camera is connected.")
#             st.session_state.running = False
#         else:
#             st.write("Camera opened successfully. Starting real-time detection...")
#             detection_cooldown = 0  # Cooldown timer to avoid multiple detections at the same spot
            
#             while st.session_state.running and cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     break

#                 # Run YOLO prediction
#                 results = model.predict(
#                     source=frame,
#                     conf=0.25,
#                     half=True,
#                     imgsz=320,
#                     verbose=False
#                 )

#                 # Check if potholes are detected and cooldown timer has expired
#                 pothole_detected = False
#                 for result in results:
#                     if result.boxes and detection_cooldown <= 0:  # If there are any detections and no cooldown
#                         pothole_detected = True
#                         # Set cooldown timer to avoid detecting the same pothole multiple times
#                         detection_cooldown = 30  # About 2 seconds at 15 FPS
#                         break

#                 # Decrement cooldown timer
#                 if detection_cooldown > 0:
#                     detection_cooldown -= 1

#                 # If a pothole is detected, record the location
#                 if pothole_detected:
#                     lat, lon = get_current_location()
#                     if add_pothole_location(lat, lon):
#                         st.write(f"Pothole detected! Recorded at Latitude {lat}, Longitude {lon}")
                    
#                     with map_placeholder:
#                         plot_pothole_map(st.session_state.pothole_locations)

#                 # Display the frame
#                 annotated_frame = results[0].plot()
#                 annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                 video_placeholder.image(annotated_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
#                 time.sleep(0.066)  # ~15 FPS

#             cap.release()
#             st.write("Camera feed stopped.")
#             st.session_state.running = False

#     elif not st.session_state.running:
#         st.write("Press 'Start' to begin real-time detection.")

# # View pothole map mode
# elif detection_mode == "View Pothole Map":
#     st.subheader("Pothole Map")
#     st.write(f"Showing {len(st.session_state.pothole_locations)} recorded pothole locations on OpenStreetMap.")
#     plot_pothole_map(st.session_state.pothole_locations)

# # Manage pothole data mode
# elif detection_mode == "Manage Pothole Data":
#     st.subheader("Manage Pothole Data")
    
#     # Option to add a manual pothole location
#     st.write("Add a new pothole location manually:")
#     col1, col2 = st.columns(2)
#     with col1:
#         manual_lat = st.number_input("Latitude", value=get_current_location()[0], format="%.6f")
#     with col2:
#         manual_lon = st.number_input("Longitude", value=get_current_location()[1], format="%.6f")
    
#     if st.button("Add Manual Location"):
#         if add_pothole_location(manual_lat, manual_lon):
#             st.success(f"Added new pothole location at {manual_lat}, {manual_lon}")
        
#     # Option to clear all pothole data
#     if st.button("Clear All Pothole Data"):
#         confirm = st.checkbox("I confirm I want to delete all pothole data")
#         if confirm:
#             st.session_state.pothole_locations = []
#             save_pothole_locations([])  # Save empty list to file
#             st.success("All pothole location data has been cleared.")
    
#     # Show the current data
#     st.write("Current pothole locations:")
#     if st.session_state.pothole_locations:
#         location_df = {"Latitude": [loc[0] for loc in st.session_state.pothole_locations],
#                        "Longitude": [loc[1] for loc in st.session_state.pothole_locations]}
#         st.dataframe(location_df)
#         plot_pothole_map(st.session_state.pothole_locations)
#     else:
#         st.write("No pothole locations recorded yet.")
    
#     # Option to export data
#     if st.button("Export Data as JSON") and st.session_state.pothole_locations:
#         st.download_button(
#             label="Download JSON",
#             data=json.dumps(st.session_state.pothole_locations),
#             file_name="pothole_locations.json",
#             mime="application/json"
#         )




# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time
# import geocoder
# import folium
# from streamlit_folium import folium_static
# import json
# import base64
# import tempfile
# import shutil

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"
# # Path for storing pothole locations
# pothole_data_path = os.path.join(HOME, "pothole_locations.json")

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Function to save pothole locations to file
# def save_pothole_locations(locations):
#     with open(pothole_data_path, 'w') as f:
#         json.dump(locations, f)
#     st.write(f"Saved {len(locations)} pothole locations to {pothole_data_path}")

# # Function to load pothole locations from file
# def load_pothole_locations():
#     if os.path.exists(pothole_data_path):
#         try:
#             with open(pothole_data_path, 'r') as f:
#                 locations = json.load(f)
#             st.write(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             return locations
#         except Exception as e:
#             st.error(f"Error loading pothole locations: {e}")
#             return []
#     else:
#         st.write("No saved pothole locations found. Starting with empty map.")
#         return []

# # Initialize session state for storing pothole locations
# if "pothole_locations" not in st.session_state:
#     st.session_state.pothole_locations = load_pothole_locations()

# # Function to get current location
# def get_current_location():
#     try:
#         g = geocoder.ip('me')
#         if g.ok:
#             return g.latlng  # Returns [lat, lon]
#         else:
#             st.warning("Could not retrieve location. Using default location.")
#             return [37.7749, -122.4194]  # Default to San Francisco
#     except Exception as e:
#         st.error(f"Error retrieving location: {e}")
#         return [37.7749, -122.4194]

# # Function to plot pothole map
# def plot_pothole_map(locations):
#     if not locations:
#         st.write("No pothole locations recorded yet.")
#         return
    
#     # Use the first location to center the map, or calculate an average center
#     if len(locations) == 1:
#         center = locations[0]
#     else:
#         # Calculate the average lat/lon for centering the map
#         avg_lat = sum(loc[0] for loc in locations) / len(locations)
#         avg_lon = sum(loc[1] for loc in locations) / len(locations)
#         center = [avg_lat, avg_lon]
    
#     m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    
#     # Add markers for each pothole
#     for lat, lon in locations:
#         folium.Marker(
#             location=[lat, lon],
#             popup="Pothole Detected",
#             icon=folium.Icon(color="red", icon="warning-sign")
#         ).add_to(m)
    
#     folium_static(m)

# # Function to add a new pothole location
# def add_pothole_location(lat, lon):
#     # Check if this location is already recorded (within a small radius)
#     for existing_lat, existing_lon in st.session_state.pothole_locations:
#         # Simple distance check (approximate)
#         if abs(existing_lat - lat) < 0.0001 and abs(existing_lon - lon) < 0.0001:
#             st.write("This pothole location is already recorded.")
#             return False
    
#     # Add new location
#     st.session_state.pothole_locations.append([lat, lon])
#     # Save to persistent storage
#     save_pothole_locations(st.session_state.pothole_locations)
#     return True

# # Function to create download link for video
# def get_video_download_link(video_path, link_text="Download processed video"):
#     """Generate a link to download the video file"""
#     with open(video_path, "rb") as file:
#         video_bytes = file.read()
#     b64 = base64.b64encode(video_bytes).decode()
    
#     # Get filename from path
#     filename = os.path.basename(video_path)
#     dl_link = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
#     return dl_link

# # Function to create MP4 video from AVI
# def convert_to_mp4(input_path):
#     """Convert AVI to MP4 format for better browser compatibility"""
#     output_path = os.path.splitext(input_path)[0] + ".mp4"
    
#     try:
#         # Read the input video
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             st.error(f"Could not open video file: {input_path}")
#             return None
            
#         # Get video properties
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
        
#         # Use H.264 codec for MP4
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         # Process video frame by frame
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
            
#         # Release resources
#         cap.release()
#         out.release()
        
#         if os.path.exists(output_path):
#             st.write(f"Successfully converted video to MP4: {output_path}")
#             return output_path
#         else:
#             st.error("Failed to create MP4 file")
#             return None
#     except Exception as e:
#         st.error(f"Error converting video: {e}")
        
#         # Try using FFmpeg as fallback if available
#         try:
#             import subprocess
#             st.write("Attempting conversion with FFmpeg...")
#             ffmpeg_cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -preset fast -crf 22 "{output_path}"'
#             result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
            
#             if os.path.exists(output_path):
#                 st.write("FFmpeg conversion successful")
#                 return output_path
#             else:
#                 st.error(f"FFmpeg conversion failed: {result.stderr}")
#                 return None
#         except Exception as ffmpeg_error:
#             st.error(f"FFmpeg fallback failed: {ffmpeg_error}")
#             return None

# # Extract frames as fallback method
# def extract_video_frames(video_path, max_frames=20):
#     """Extract frames from video as a fallback display method"""
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return []
            
#         frames = []
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Calculate step to evenly distribute frames
#         step = max(1, total_frames // max_frames)
        
#         for i in range(0, total_frames, step):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if ret:
#                 # Convert BGR to RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
            
#             if len(frames) >= max_frames:
#                 break
                
#         cap.release()
#         return frames
#     except Exception as e:
#         st.error(f"Error extracting frames: {e}")
#         return []

# # Streamlit UI
# st.title("Pothole Detection with YOLO")
# st.write("Choose an option: Upload a video, use your dashcam, or view the pothole map.")

# # Option selection
# detection_mode = st.radio("Select Detection Mode", ("Upload Video", "Real-Time Dashcam", "View Pothole Map", "Manage Pothole Data"))

# # Video upload mode
# if detection_mode == "Upload Video":
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         # Save the uploaded video temporarily
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")

#         # Run YOLO prediction
#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,
#                 half=True,
#                 imgsz=640
#             )

#         # Get the output path
#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#         # Verify and display output
#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
#             st.subheader("Processed Video with Potholes")
            
#             # Create a download link for the video
#             st.markdown(
#                 get_video_download_link(output_path, "⬇️ Download processed video"),
#                 unsafe_allow_html=True
#             )
            
#             # Try different methods to display the video
#             try:
#                 # 1. Try to convert to MP4 first (most compatible format)
#                 mp4_path = convert_to_mp4(output_path)
#                 if mp4_path and os.path.exists(mp4_path):
#                     st.video(mp4_path)
#                     st.success("Video converted to MP4 for better compatibility")
#                 else:
#                     # 2. Fallback to original format
#                     st.warning("MP4 conversion failed, trying original format")
#                     st.video(output_path)
#             except Exception as e:
#                 st.error(f"Video playback error: {str(e)}")
                
#                 # 3. Show video frames as fallback
#                 st.write("Showing video frames as fallback:")
#                 frames = extract_video_frames(output_path)
#                 if frames:
#                     col1, col2 = st.columns(2)
#                     for i, frame in enumerate(frames):
#                         if i % 2 == 0:
#                             with col1:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                         else:
#                             with col2:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                 else:
#                     st.error("Could not extract frames from video")

#             # Plot the current location as having a pothole
#             lat, lon = get_current_location()
#             if add_pothole_location(lat, lon):
#                 st.write(f"Pothole recorded at Latitude {lat}, Longitude {lon}")
#             plot_pothole_map(st.session_state.pothole_locations)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")
            
#             # Check if there's any output in the runs directory
#             detect_dirs = [d for d in os.listdir("runs/detect") if d.startswith("predict")]
#             if detect_dirs:
#                 latest_dir = max(detect_dirs, key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
#                 st.write(f"Latest output directory: runs/detect/{latest_dir}")
#                 files_in_dir = os.listdir(os.path.join("runs/detect", latest_dir))
#                 st.write(f"Files in directory: {files_in_dir}")

# # Real-time dashcam mode
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")

#     if "running" not in st.session_state:
#         st.session_state.running = False
#     if "detection_frames" not in st.session_state:
#         st.session_state.detection_frames = []

#     col1, col2 = st.columns(2)
#     with col1:
#         start_button = st.button("Start", key="start_button")
#     with col2:
#         stop_button = st.button("Stop", key="stop_button")

#     video_placeholder = st.empty()
#     map_placeholder = st.empty()
#     frame_gallery = st.empty()

#     if start_button:
#         st.session_state.running = True
#         st.session_state.detection_frames = []  # Reset detection frames
#     if stop_button:
#         st.session_state.running = False

#     if st.session_state.running:
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error("Error: Could not open camera. Ensure a camera is connected.")
#             st.session_state.running = False
#         else:
#             st.write("Camera opened successfully. Starting real-time detection...")
#             detection_cooldown = 0  # Cooldown timer to avoid multiple detections at the same spot
            
#             while st.session_state.running and cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     break

#                 # Run YOLO prediction
#                 results = model.predict(
#                     source=frame,
#                     conf=0.25,
#                     half=True,
#                     imgsz=320,
#                     verbose=False
#                 )

#                 # Check if potholes are detected and cooldown timer has expired
#                 pothole_detected = False
#                 for result in results:
#                     if result.boxes and detection_cooldown <= 0:  # If there are any detections and no cooldown
#                         pothole_detected = True
#                         # Set cooldown timer to avoid detecting the same pothole multiple times
#                         detection_cooldown = 30  # About 2 seconds at 15 FPS
#                         break

#                 # Decrement cooldown timer
#                 if detection_cooldown > 0:
#                     detection_cooldown -= 1

#                 # If a pothole is detected, record the location and save the frame
#                 if pothole_detected:
#                     lat, lon = get_current_location()
#                     if add_pothole_location(lat, lon):
#                         st.write(f"Pothole detected! Recorded at Latitude {lat}, Longitude {lon}")
                        
#                         # Save detection frame for gallery
#                         detection_frame = results[0].plot().copy()
#                         detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
#                         st.session_state.detection_frames.append(detection_frame_rgb)
                        
#                         # Update map
#                         with map_placeholder:
#                             plot_pothole_map(st.session_state.pothole_locations)

#                 # Display the frame
#                 annotated_frame = results[0].plot()
#                 annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                 video_placeholder.image(annotated_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
#                 time.sleep(0.066)  # ~15 FPS

#             cap.release()
#             st.write("Camera feed stopped.")
#             st.session_state.running = False
            
#             # Display detection frames gallery
#             if st.session_state.detection_frames:
#                 st.subheader("Pothole Detections")
#                 cols = st.columns(min(3, len(st.session_state.detection_frames)))
#                 for i, frame in enumerate(st.session_state.detection_frames):
#                     cols[i % len(cols)].image(frame, caption=f"Detection {i+1}", use_column_width=True)

#     elif not st.session_state.running:
#         st.write("Press 'Start' to begin real-time detection.")
        
#         # Display previous detection frames if any
#         if st.session_state.detection_frames:
#             st.subheader("Previous Pothole Detections")
#             cols = st.columns(min(3, len(st.session_state.detection_frames)))
#             for i, frame in enumerate(st.session_state.detection_frames):
#                 cols[i % len(cols)].image(frame, caption=f"Detection {i+1}", use_column_width=True)

# # View pothole map mode
# elif detection_mode == "View Pothole Map":
#     st.subheader("Pothole Map")
#     st.write(f"Showing {len(st.session_state.pothole_locations)} recorded pothole locations on OpenStreetMap.")
#     plot_pothole_map(st.session_state.pothole_locations)

# # Manage pothole data mode
# elif detection_mode == "Manage Pothole Data":
#     st.subheader("Manage Pothole Data")
    
#     # Option to add a manual pothole location
#     st.write("Add a new pothole location manually:")
#     col1, col2 = st.columns(2)
#     with col1:
#         manual_lat = st.number_input("Latitude", value=get_current_location()[0], format="%.6f")
#     with col2:
#         manual_lon = st.number_input("Longitude", value=get_current_location()[1], format="%.6f")
    
#     if st.button("Add Manual Location"):
#         if add_pothole_location(manual_lat, manual_lon):
#             st.success(f"Added new pothole location at {manual_lat}, {manual_lon}")
        
#     # Option to clear all pothole data
#     if st.button("Clear All Pothole Data"):
#         confirm = st.checkbox("I confirm I want to delete all pothole data")
#         if confirm:
#             st.session_state.pothole_locations = []
#             save_pothole_locations([])  # Save empty list to file
#             st.success("All pothole location data has been cleared.")
    
#     # Show the current data
#     st.write("Current pothole locations:")
#     if st.session_state.pothole_locations:
#         location_df = {"Latitude": [loc[0] for loc in st.session_state.pothole_locations],
#                        "Longitude": [loc[1] for loc in st.session_state.pothole_locations]}
#         st.dataframe(location_df)
#         plot_pothole_map(st.session_state.pothole_locations)
#     else:
#         st.write("No pothole locations recorded yet.")
    
#     # Option to export data
#     if st.button("Export Data as JSON") and st.session_state.pothole_locations:
#         st.download_button(
#             label="Download JSON",
#             data=json.dumps(st.session_state.pothole_locations),
#             file_name="pothole_locations.json",
#             mime="application/json"
#         )




# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time
# import geocoder
# import folium
# from streamlit_folium import folium_static
# import json
# import base64
# import tempfile
# import shutil

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"
# # Path for storing pothole locations
# pothole_data_path = os.path.join(HOME, "pothole_locations.json")

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Function to save pothole locations to file
# def save_pothole_locations(locations):
#     with open(pothole_data_path, 'w') as f:
#         json.dump(locations, f)
#     st.write(f"Saved {len(locations)} pothole locations to {pothole_data_path}")

# # Function to load pothole locations from file
# def load_pothole_locations():
#     if os.path.exists(pothole_data_path):
#         try:
#             with open(pothole_data_path, 'r') as f:
#                 locations = json.load(f)
#             st.write(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             return locations
#         except Exception as e:
#             st.error(f"Error loading pothole locations: {e}")
#             return []
#     else:
#         st.write("No saved pothole locations found. Starting with empty map.")
#         return []

# # Initialize session state for storing pothole locations
# if "pothole_locations" not in st.session_state:
#     st.session_state.pothole_locations = load_pothole_locations()

# # Function to get current location
# def get_current_location():
#     try:
#         g = geocoder.ip('me')
#         if g.ok:
#             return g.latlng  # Returns [lat, lon]
#         else:
#             st.warning("Could not retrieve location. Using default location.")
#             return [37.7749, -122.4194]  # Default to San Francisco
#     except Exception as e:
#         st.error(f"Error retrieving location: {e}")
#         return [37.7749, -122.4194]

# # Function to plot pothole map
# def plot_pothole_map(locations):
#     if not locations:
#         st.write("No pothole locations recorded yet.")
#         return
    
#     # Use the first location to center the map, or calculate an average center
#     if len(locations) == 1:
#         center = locations[0]
#     else:
#         # Calculate the average lat/lon for centering the map
#         avg_lat = sum(loc[0] for loc in locations) / len(locations)
#         avg_lon = sum(loc[1] for loc in locations) / len(locations)
#         center = [avg_lat, avg_lon]
    
#     m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    
#     # Add markers for each pothole
#     for lat, lon in locations:
#         folium.Marker(
#             location=[lat, lon],
#             popup="Pothole Detected",
#             icon=folium.Icon(color="red", icon="warning-sign")
#         ).add_to(m)
    
#     folium_static(m)

# # Function to add a new pothole location
# def add_pothole_location(lat, lon):
#     # Check if this location is already recorded (within a small radius)
#     for existing_lat, existing_lon in st.session_state.pothole_locations:
#         # Simple distance check (approximate)
#         if abs(existing_lat - lat) < 0.0001 and abs(existing_lon - lon) < 0.0001:
#             st.write("This pothole location is already recorded.")
#             return False
    
#     # Add new location
#     st.session_state.pothole_locations.append([lat, lon])
#     # Save to persistent storage
#     save_pothole_locations(st.session_state.pothole_locations)
#     return True

# # Function to create download link for video
# def get_video_download_link(video_path, link_text="Download processed video"):
#     """Generate a link to download the video file"""
#     with open(video_path, "rb") as file:
#         video_bytes = file.read()
#     b64 = base64.b64encode(video_bytes).decode()
    
#     # Get filename from path
#     filename = os.path.basename(video_path)
#     dl_link = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
#     return dl_link

# # Function to create MP4 video from AVI
# def convert_to_mp4(input_path):
#     """Convert AVI to MP4 format for better browser compatibility"""
#     output_path = os.path.splitext(input_path)[0] + ".mp4"
    
#     try:
#         # Read the input video
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             st.error(f"Could not open video file: {input_path}")
#             return None
            
#         # Get video properties
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
        
#         # Use H.264 codec for MP4
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         # Process video frame by frame
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
            
#         # Release resources
#         cap.release()
#         out.release()
        
#         if os.path.exists(output_path):
#             st.write(f"Successfully converted video to MP4: {output_path}")
#             return output_path
#         else:
#             st.error("Failed to create MP4 file")
#             return None
#     except Exception as e:
#         st.error(f"Error converting video: {e}")
        
#         # Try using FFmpeg as fallback if available
#         try:
#             import subprocess
#             st.write("Attempting conversion with FFmpeg...")
#             ffmpeg_cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -preset fast -crf 22 "{output_path}"'
#             result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
            
#             if os.path.exists(output_path):
#                 st.write("FFmpeg conversion successful")
#                 return output_path
#             else:
#                 st.error(f"FFmpeg conversion failed: {result.stderr}")
#                 return None
#         except Exception as ffmpeg_error:
#             st.error(f"FFmpeg fallback failed: {ffmpeg_error}")
#             return None

# # Extract frames as fallback method
# def extract_video_frames(video_path, max_frames=20):
#     """Extract frames from video as a fallback display method"""
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return []
            
#         frames = []
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Calculate step to evenly distribute frames
#         step = max(1, total_frames // max_frames)
        
#         for i in range(0, total_frames, step):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if ret:
#                 # Convert BGR to RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
            
#             if len(frames) >= max_frames:
#                 break
                
#         cap.release()
#         return frames
#     except Exception as e:
#         st.error(f"Error extracting frames: {e}")
#         return []

# # Streamlit UI
# st.title("Pothole Detection with YOLO")
# st.write("Choose an option: Upload a video, use your dashcam, or view the pothole map.")

# # Option selection
# detection_mode = st.radio("Select Detection Mode", ("Upload Video", "Real-Time Dashcam", "View Pothole Map", "Manage Pothole Data"))

# # Video upload mode
# if detection_mode == "Upload Video":
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         # Save the uploaded video temporarily
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")

#         # Run YOLO prediction
#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,
#                 half=True,
#                 imgsz=640
#             )

#         # Get the output path
#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#         # Verify and display output
#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
#             st.subheader("Processed Video with Potholes")
            
#             # Create a download link for the video
#             st.markdown(
#                 get_video_download_link(output_path, "⬇️ Download processed video"),
#                 unsafe_allow_html=True
#             )
            
#             # Try different methods to display the video
#             try:
#                 # 1. Try to convert to MP4 first (most compatible format)
#                 mp4_path = convert_to_mp4(output_path)
#                 if mp4_path and os.path.exists(mp4_path):
#                     st.video(mp4_path)
#                     st.success("Video converted to MP4 for better compatibility")
#                 else:
#                     # 2. Fallback to original format
#                     st.warning("MP4 conversion failed, trying original format")
#                     st.video(output_path)
#             except Exception as e:
#                 st.error(f"Video playback error: {str(e)}")
                
#                 # 3. Show video frames as fallback
#                 st.write("Showing video frames as fallback:")
#                 frames = extract_video_frames(output_path)
#                 if frames:
#                     col1, col2 = st.columns(2)
#                     for i, frame in enumerate(frames):
#                         if i % 2 == 0:
#                             with col1:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                         else:
#                             with col2:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                 else:
#                     st.error("Could not extract frames from video")

#             # Plot the current location as having a pothole
#             lat, lon = get_current_location()
#             if add_pothole_location(lat, lon):
#                 st.write(f"Pothole recorded at Latitude {lat}, Longitude {lon}")
#             plot_pothole_map(st.session_state.pothole_locations)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")
            
#             # Check if there's any output in the runs directory
#             detect_dirs = [d for d in os.listdir("runs/detect") if d.startswith("predict")]
#             if detect_dirs:
#                 latest_dir = max(detect_dirs, key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
#                 st.write(f"Latest output directory: runs/detect/{latest_dir}")
#                 files_in_dir = os.listdir(os.path.join("runs/detect", latest_dir))
#                 st.write(f"Files in directory: {files_in_dir}")

# # Real-time dashcam mode
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")

#     # Initialize session state variables
#     if "running" not in st.session_state:
#         st.session_state.running = False
#     if "detection_frames" not in st.session_state:
#         st.session_state.detection_frames = []

#     # Control buttons
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Start", key="start_button"):
#             st.session_state.running = True
#             st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
#     with col2:
#         if st.button("Stop", key="stop_button"):
#             st.session_state.running = False
#             st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

#     # Create placeholders for video display
#     video_placeholder = st.empty()
#     status_text = st.empty()
#     map_placeholder = st.empty()

#     # Only run camera when state is running
#     if st.session_state.running:
#         status_text.info("Starting camera... Please wait.")
        
#         try:
#             # Initialize camera
#             cap = cv2.VideoCapture(0)
            
#             if not cap.isOpened():
#                 st.error("Error: Could not open camera. Ensure a camera is connected.")
#                 st.session_state.running = False
#             else:
#                 status_text.success("Camera connected! Displaying live feed...")
                
#                 # Get a single frame to test
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     cap.release()
#                     st.session_state.running = False
#                 else:
#                     # Display the first frame to confirm camera works
#                     first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     video_placeholder.image(first_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
                    
#                     # Now capture a limited number of frames in a loop
#                     # This prevents Streamlit from hanging
#                     detection_cooldown = 0
#                     FRAME_LIMIT = 100  # Process 100 frames before requiring refresh
                    
#                     for frame_count in range(FRAME_LIMIT):
#                         if not st.session_state.running:
#                             break
                            
#                         ret, frame = cap.read()
#                         if not ret:
#                             st.error("Error reading frame")
#                             break
                            
#                         # Run YOLO prediction
#                         results = model.predict(
#                             source=frame,
#                             conf=0.25,
#                             half=True,
#                             imgsz=320,
#                             verbose=False
#                         )
                        
#                         # Check for pothole detection with cooldown
#                         pothole_detected = False
#                         for result in results:
#                             if result.boxes and detection_cooldown <= 0:
#                                 pothole_detected = True
#                                 detection_cooldown = 15  # Lower cooldown for more responsive detection
#                                 break
                                
#                         # Decrement cooldown
#                         if detection_cooldown > 0:
#                             detection_cooldown -= 1
                            
#                         # Handle pothole detection
#                         if pothole_detected:
#                             lat, lon = get_current_location()
#                             if add_pothole_location(lat, lon):
#                                 status_text.warning(f"Pothole detected! Location: {lat:.6f}, {lon:.6f}")
                                
#                                 # Save detection frame
#                                 detection_frame = results[0].plot().copy()
#                                 detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
#                                 st.session_state.detection_frames.append(detection_frame_rgb)
                                
#                                 # Update map occasionally (not every frame)
#                                 if frame_count % 10 == 0:
#                                     with map_placeholder:
#                                         plot_pothole_map(st.session_state.pothole_locations)
                        
#                         # Display the current frame
#                         annotated_frame = results[0].plot()
#                         annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                         video_placeholder.image(annotated_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
                        
#                         # Small delay to prevent UI freezing (adjust as needed)
#                         time.sleep(0.05)  # ~20 FPS
                        
#                     # After frame limit, provide button to continue
#                     status_text.info(f"Processed {FRAME_LIMIT} frames. Click 'Continue' to keep recording.")
#                     if st.button("Continue Recording"):
#                         st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
                
#                 # Release the camera when done
#                 cap.release()
        
#         except Exception as e:
#             st.error(f"Camera error: {str(e)}")
#             st.session_state.running = False
        
#         finally:
#             # Always try to release camera on error
#             try:
#                 cap.release()
#             except:
#                 pass
    
#     else:
#         st.write("Press 'Start' to begin real-time detection.")
    
#     # Display detection frames gallery (show regardless of camera state)
#     if st.session_state.detection_frames:
#         st.subheader("Pothole Detections")
        
#         # Limit to most recent 9 frames to prevent UI clutter
#         recent_frames = st.session_state.detection_frames[-9:]
#         cols = st.columns(min(3, len(recent_frames)))
#         for i, frame in enumerate(recent_frames):
#             cols[i % len(cols)].image(frame, caption=f"Detection {i+1}", use_column_width=True)

# # View pothole map mode
# elif detection_mode == "View Pothole Map":
#     st.subheader("Pothole Map")
#     st.write(f"Showing {len(st.session_state.pothole_locations)} recorded pothole locations on OpenStreetMap.")
#     plot_pothole_map(st.session_state.pothole_locations)

# # Manage pothole data mode
# elif detection_mode == "Manage Pothole Data":
#     st.subheader("Manage Pothole Data")
    
#     # Option to add a manual pothole location
#     st.write("Add a new pothole location manually:")
#     col1, col2 = st.columns(2)
#     with col1:
#         manual_lat = st.number_input("Latitude", value=get_current_location()[0], format="%.6f")
#     with col2:
#         manual_lon = st.number_input("Longitude", value=get_current_location()[1], format="%.6f")
    
#     if st.button("Add Manual Location"):
#         if add_pothole_location(manual_lat, manual_lon):
#             st.success(f"Added new pothole location at {manual_lat}, {manual_lon}")
        
#     # Option to clear all pothole data
#     if st.button("Clear All Pothole Data"):
#         confirm = st.checkbox("I confirm I want to delete all pothole data")
#         if confirm:
#             st.session_state.pothole_locations = []
#             save_pothole_locations([])  # Save empty list to file
#             st.success("All pothole location data has been cleared.")
    
#     # Show the current data
#     st.write("Current pothole locations:")
#     if st.session_state.pothole_locations:
#         location_df = {"Latitude": [loc[0] for loc in st.session_state.pothole_locations],
#                        "Longitude": [loc[1] for loc in st.session_state.pothole_locations]}
#         st.dataframe(location_df)
#         plot_pothole_map(st.session_state.pothole_locations)
#     else:
#         st.write("No pothole locations recorded yet.")
    
#     # Option to export data
#     if st.button("Export Data as JSON") and st.session_state.pothole_locations:
#         st.download_button(
#             label="Download JSON",
#             data=json.dumps(st.session_state.pothole_locations),
#             file_name="pothole_locations.json",
#             mime="application/json"
#         )



# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time
# import geocoder
# import folium
# from streamlit_folium import folium_static
# import json
# import base64
# import tempfile
# import shutil

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"
# # Path for storing pothole locations
# pothole_data_path = os.path.join(HOME, "pothole_locations.json")

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Function to save pothole locations to file
# def save_pothole_locations(locations):
#     with open(pothole_data_path, 'w') as f:
#         json.dump(locations, f)
#     st.write(f"Saved {len(locations)} pothole locations to {pothole_data_path}")

# # Function to load pothole locations from file
# def load_pothole_locations():
#     if os.path.exists(pothole_data_path):
#         try:
#             with open(pothole_data_path, 'r') as f:
#                 locations = json.load(f)
#             st.write(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             return locations
#         except Exception as e:
#             st.error(f"Error loading pothole locations: {e}")
#             return []
#     else:
#         st.write("No saved pothole locations found. Starting with empty map.")
#         return []

# # Initialize session state for storing pothole locations
# if "pothole_locations" not in st.session_state:
#     st.session_state.pothole_locations = load_pothole_locations()

# # Function to get current location
# def get_current_location():
#     try:
#         g = geocoder.ip('me')
#         if g.ok:
#             return g.latlng  # Returns [lat, lon]
#         else:
#             st.warning("Could not retrieve location. Using default location.")
#             return [37.7749, -122.4194]  # Default to San Francisco
#     except Exception as e:
#         st.error(f"Error retrieving location: {e}")
#         return [37.7749, -122.4194]

# # Function to plot pothole map
# def plot_pothole_map(locations):
#     if not locations:
#         st.write("No pothole locations recorded yet.")
#         return
    
#     # Use the first location to center the map, or calculate an average center
#     if len(locations) == 1:
#         center = locations[0]
#     else:
#         # Calculate the average lat/lon for centering the map
#         avg_lat = sum(loc[0] for loc in locations) / len(locations)
#         avg_lon = sum(loc[1] for loc in locations) / len(locations)
#         center = [avg_lat, avg_lon]
    
#     m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    
#     # Add markers for each pothole
#     for lat, lon in locations:
#         folium.Marker(
#             location=[lat, lon],
#             popup="Pothole Detected",
#             icon=folium.Icon(color="red", icon="warning-sign")
#         ).add_to(m)
    
#     folium_static(m)

# # Function to add a new pothole location
# def add_pothole_location(lat, lon):
#     # Check if this location is already recorded (within a small radius)
#     for existing_lat, existing_lon in st.session_state.pothole_locations:
#         # Simple distance check (approximate)
#         if abs(existing_lat - lat) < 0.0001 and abs(existing_lon - lon) < 0.0001:
#             st.write("This pothole location is already recorded.")
#             return False
    
#     # Add new location
#     st.session_state.pothole_locations.append([lat, lon])
#     # Save to persistent storage
#     save_pothole_locations(st.session_state.pothole_locations)
#     return True

# # Function to create download link for video
# def get_video_download_link(video_path, link_text="Download processed video"):
#     """Generate a link to download the video file"""
#     with open(video_path, "rb") as file:
#         video_bytes = file.read()
#     b64 = base64.b64encode(video_bytes).decode()
    
#     # Get filename from path
#     filename = os.path.basename(video_path)
#     dl_link = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
#     return dl_link

# # Function to create MP4 video from AVI
# def convert_to_mp4(input_path):
#     """Convert AVI to MP4 format for better browser compatibility"""
#     output_path = os.path.splitext(input_path)[0] + ".mp4"
    
#     try:
#         # Read the input video
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             st.error(f"Could not open video file: {input_path}")
#             return None
            
#         # Get video properties
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
        
#         # Use H.264 codec for MP4
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         # Process video frame by frame
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
            
#         # Release resources
#         cap.release()
#         out.release()
        
#         if os.path.exists(output_path):
#             st.write(f"Successfully converted video to MP4: {output_path}")
#             return output_path
#         else:
#             st.error("Failed to create MP4 file")
#             return None
#     except Exception as e:
#         st.error(f"Error converting video: {e}")
        
#         # Try using FFmpeg as fallback if available
#         try:
#             import subprocess
#             st.write("Attempting conversion with FFmpeg...")
#             ffmpeg_cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -preset fast -crf 22 "{output_path}"'
#             result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
            
#             if os.path.exists(output_path):
#                 st.write("FFmpeg conversion successful")
#                 return output_path
#             else:
#                 st.error(f"FFmpeg conversion failed: {result.stderr}")
#                 return None
#         except Exception as ffmpeg_error:
#             st.error(f"FFmpeg fallback failed: {ffmpeg_error}")
#             return None

# # Extract frames as fallback method
# def extract_video_frames(video_path, max_frames=20):
#     """Extract frames from video as a fallback display method"""
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return []
            
#         frames = []
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Calculate step to evenly distribute frames
#         step = max(1, total_frames // max_frames)
        
#         for i in range(0, total_frames, step):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if ret:
#                 # Convert BGR to RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
            
#             if len(frames) >= max_frames:
#                 break
                
#         cap.release()
#         return frames
#     except Exception as e:
#         st.error(f"Error extracting frames: {e}")
#         return []

# # Streamlit UI
# st.title("Pothole Detection with YOLO")
# st.write("Choose an option: Upload a video, use your dashcam, or view the pothole map.")

# # Option selection
# detection_mode = st.radio("Select Detection Mode", ("Upload Video", "Real-Time Dashcam", "View Pothole Map", "Manage Pothole Data"))

# # Video upload mode
# if detection_mode == "Upload Video":
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         # Save the uploaded video temporarily
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")

#         # Run YOLO prediction
#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,
#                 half=True,
#                 imgsz=640
#             )

#         # Get the output path
#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#         # Verify and display output
#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
#             st.subheader("Processed Video with Potholes")
            
#             # Create a download link for the video
#             st.markdown(
#                 get_video_download_link(output_path, "⬇️ Download processed video"),
#                 unsafe_allow_html=True
#             )
            
#             # Try different methods to display the video
#             try:
#                 # 1. Try to convert to MP4 first (most compatible format)
#                 mp4_path = convert_to_mp4(output_path)
#                 if mp4_path and os.path.exists(mp4_path):
#                     st.video(mp4_path)
#                     st.success("Video converted to MP4 for better compatibility")
#                 else:
#                     # 2. Fallback to original format
#                     st.warning("MP4 conversion failed, trying original format")
#                     st.video(output_path)
#             except Exception as e:
#                 st.error(f"Video playback error: {str(e)}")
                
#                 # 3. Show video frames as fallback
#                 st.write("Showing video frames as fallback:")
#                 frames = extract_video_frames(output_path)
#                 if frames:
#                     col1, col2 = st.columns(2)
#                     for i, frame in enumerate(frames):
#                         if i % 2 == 0:
#                             with col1:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                         else:
#                             with col2:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                 else:
#                     st.error("Could not extract frames from video")

#             # Plot the current location as having a pothole
#             lat, lon = get_current_location()
#             if add_pothole_location(lat, lon):
#                 st.write(f"Pothole recorded at Latitude {lat}, Longitude {lon}")
#             plot_pothole_map(st.session_state.pothole_locations)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")
            
#             # Check if there's any output in the runs directory
#             detect_dirs = [d for d in os.listdir("runs/detect") if d.startswith("predict")]
#             if detect_dirs:
#                 latest_dir = max(detect_dirs, key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
#                 st.write(f"Latest output directory: runs/detect/{latest_dir}")
#                 files_in_dir = os.listdir(os.path.join("runs/detect", latest_dir))
#                 st.write(f"Files in directory: {files_in_dir}")

# # Real-time dashcam mode
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")

#     # Initialize session state variables
#     if "running" not in st.session_state:
#         st.session_state.running = False
#     if "detection_frames" not in st.session_state:
#         st.session_state.detection_frames = []

#     # Control buttons
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Start", key="start_button"):
#             st.session_state.running = True
#             st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
#     with col2:
#         if st.button("Stop", key="stop_button"):
#             st.session_state.running = False
#             st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

#     # Create placeholders for video display
#     video_placeholder = st.empty()
#     status_text = st.empty()
#     map_placeholder = st.empty()

#     # Only run camera when state is running
#     if st.session_state.running:
#         status_text.info("Starting camera... Please wait.")
        
#         try:
#             # Initialize camera
#             cap = cv2.VideoCapture(0)
            
#             if not cap.isOpened():
#                 st.error("Error: Could not open camera. Ensure a camera is connected.")
#                 st.session_state.running = False
#             else:
#                 status_text.success("Camera connected! Displaying live feed...")
                
#                 # Get a single frame to test
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     cap.release()
#                     st.session_state.running = False
#                 else:
#                     # Display the first frame to confirm camera works
#                     first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     video_placeholder.image(first_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
                    
#                     # Now capture a limited number of frames in a loop
#                     # This prevents Streamlit from hanging
#                     detection_cooldown = 0
#                     FRAME_LIMIT = 100  # Process 100 frames before requiring refresh
                    
#                     for frame_count in range(FRAME_LIMIT):
#                         if not st.session_state.running:
#                             break
                            
#                         ret, frame = cap.read()
#                         if not ret:
#                             st.error("Error reading frame")
#                             break
                            
#                         # Run YOLO prediction - CHANGED THIS LINE TO USE 640 instead of 320
#                         results = model.predict(
#                             source=frame,
#                             conf=0.25,
#                             half=True,
#                             imgsz=640,  # Match the input size expected by the model (640x640)
#                             verbose=False
#                         )
                        
#                         # Check for pothole detection with cooldown
#                         pothole_detected = False
#                         for result in results:
#                             if result.boxes and detection_cooldown <= 0:
#                                 pothole_detected = True
#                                 detection_cooldown = 15  # Lower cooldown for more responsive detection
#                                 break
                                
#                         # Decrement cooldown
#                         if detection_cooldown > 0:
#                             detection_cooldown -= 1
                            
#                         # Handle pothole detection
#                         if pothole_detected:
#                             lat, lon = get_current_location()
#                             if add_pothole_location(lat, lon):
#                                 status_text.warning(f"Pothole detected! Location: {lat:.6f}, {lon:.6f}")
                                
#                                 # Save detection frame
#                                 detection_frame = results[0].plot().copy()
#                                 detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
#                                 st.session_state.detection_frames.append(detection_frame_rgb)
                                
#                                 # Update map occasionally (not every frame)
#                                 if frame_count % 10 == 0:
#                                     with map_placeholder:
#                                         plot_pothole_map(st.session_state.pothole_locations)
                        
#                         # Display the current frame
#                         annotated_frame = results[0].plot()
#                         annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                         video_placeholder.image(annotated_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
                        
#                         # Small delay to prevent UI freezing (adjust as needed)
#                         time.sleep(0.05)  # ~20 FPS
                        
#                     # After frame limit, provide button to continue
#                     status_text.info(f"Processed {FRAME_LIMIT} frames. Click 'Continue' to keep recording.")
#                     if st.button("Continue Recording"):
#                         st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
                
#                 # Release the camera when done
#                 cap.release()
        
#         except Exception as e:
#             st.error(f"Camera error: {str(e)}")
#             st.session_state.running = False
        
#         finally:
#             # Always try to release camera on error
#             try:
#                 cap.release()
#             except:
#                 pass
    
#     else:
#         st.write("Press 'Start' to begin real-time detection.")
    
#     # Display detection frames gallery (show regardless of camera state)
#     if st.session_state.detection_frames:
#         st.subheader("Pothole Detections")
        
#         # Limit to most recent 9 frames to prevent UI clutter
#         recent_frames = st.session_state.detection_frames[-9:]
#         cols = st.columns(min(3, len(recent_frames)))
#         for i, frame in enumerate(recent_frames):
#             cols[i % len(cols)].image(frame, caption=f"Detection {i+1}", use_column_width=True)

# # View pothole map mode
# elif detection_mode == "View Pothole Map":
#     st.subheader("Pothole Map")
#     st.write(f"Showing {len(st.session_state.pothole_locations)} recorded pothole locations on OpenStreetMap.")
#     plot_pothole_map(st.session_state.pothole_locations)

# # Manage pothole data mode
# elif detection_mode == "Manage Pothole Data":
#     st.subheader("Manage Pothole Data")
    
#     # Option to add a manual pothole location
#     st.write("Add a new pothole location manually:")
#     col1, col2 = st.columns(2)
#     with col1:
#         manual_lat = st.number_input("Latitude", value=get_current_location()[0], format="%.6f")
#     with col2:
#         manual_lon = st.number_input("Longitude", value=get_current_location()[1], format="%.6f")
    
#     if st.button("Add Manual Location"):
#         if add_pothole_location(manual_lat, manual_lon):
#             st.success(f"Added new pothole location at {manual_lat}, {manual_lon}")
        
#     # Option to clear all pothole data
#     if st.button("Clear All Pothole Data"):
#         confirm = st.checkbox("I confirm I want to delete all pothole data")
#         if confirm:
#             st.session_state.pothole_locations = []
#             save_pothole_locations([])  # Save empty list to file
#             st.success("All pothole location data has been cleared.")
    
#     # Show the current data
#     st.write("Current pothole locations:")
#     if st.session_state.pothole_locations:
#         location_df = {"Latitude": [loc[0] for loc in st.session_state.pothole_locations],
#                        "Longitude": [loc[1] for loc in st.session_state.pothole_locations]}
#         st.dataframe(location_df)
#         plot_pothole_map(st.session_state.pothole_locations)
#     else:
#         st.write("No pothole locations recorded yet.")
    
#     # Option to export data
#     if st.button("Export Data as JSON") and st.session_state.pothole_locations:
#         st.download_button(
#             label="Download JSON",
#             data=json.dumps(st.session_state.pothole_locations),
#             file_name="pothole_locations.json",
#             mime="application/json"
#         )




# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time
# import geocoder
# import folium
# from streamlit_folium import folium_static
# import json
# import base64
# import tempfile
# import shutil
# from streamlit_javascript import st_javascript

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"
# # Path for storing pothole locations
# pothole_data_path = os.path.join(HOME, "pothole_locations.json")

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Function to save pothole locations to file
# def save_pothole_locations(locations):
#     with open(pothole_data_path, 'w') as f:
#         json.dump(locations, f)
#     st.write(f"Saved {len(locations)} pothole locations to {pothole_data_path}")

# # Function to load pothole locations from file
# def load_pothole_locations():
#     if os.path.exists(pothole_data_path):
#         try:
#             with open(pothole_data_path, 'r') as f:
#                 locations = json.load(f)
#             st.write(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             return locations
#         except Exception as e:
#             st.error(f"Error loading pothole locations: {e}")
#             return []
#     else:
#         st.write("No saved pothole locations found. Starting with empty map.")
#         return []

# # Initialize session state for storing pothole locations
# if "pothole_locations" not in st.session_state:
#     st.session_state.pothole_locations = load_pothole_locations()

# # Function to get current location using browser GPS with fallback to IP geolocation
# def get_current_location():
#     # Try to get GPS location from browser first
#     try:
#         # Run JavaScript to get location from browser
#         location_data = st_javascript("""
#             async function getLocation() {
#                 return new Promise((resolve, reject) => {
#                     if (!navigator.geolocation) {
#                         reject("Geolocation not supported by your browser");
#                     }
                    
#                     navigator.geolocation.getCurrentPosition(
#                         position => {
#                             const pos = {
#                                 lat: position.coords.latitude,
#                                 lng: position.coords.longitude,
#                                 accuracy: position.coords.accuracy
#                             };
#                             resolve(pos);
#                         },
#                         error => {
#                             reject(`ERROR(${error.code}): ${error.message}`);
#                         },
#                         {
#                             enableHighAccuracy: true,
#                             timeout: 5000,
#                             maximumAge: 0
#                         }
#                     );
#                 });
#             }
            
#             const location = await getLocation();
#             return location;
#         """)
        
#         if location_data and 'lat' in location_data and 'lng' in location_data:
#             st.write(f"GPS location obtained with accuracy: {location_data.get('accuracy', 'unknown')} meters")
#             return [location_data['lat'], location_data['lng']]
#         else:
#             st.warning("Could not get GPS location. Falling back to IP geolocation.")
#     except Exception as e:
#         st.warning(f"Error getting GPS location: {e}. Falling back to IP geolocation.")
    
#     # Fallback to IP-based geolocation
#     try:
#         g = geocoder.ip('me')
#         if g.ok:
#             st.info("Using IP-based location (less accurate)")
#             return g.latlng  # Returns [lat, lon]
#         else:
#             st.warning("Could not retrieve location. Using default location.")
#             # Default to Idukki, Kerala instead of Thiruvananthapuram
#             return [9.9189, 76.9383]
#     except Exception as e:
#         st.error(f"Error retrieving location: {e}")
#         # Default to Idukki, Kerala
#         return [9.9189, 76.9383]

# # Function to plot pothole map
# def plot_pothole_map(locations):
#     if not locations:
#         st.write("No pothole locations recorded yet.")
#         return
    
#     # Use the first location to center the map, or calculate an average center
#     if len(locations) == 1:
#         center = locations[0]
#     else:
#         # Calculate the average lat/lon for centering the map
#         avg_lat = sum(loc[0] for loc in locations) / len(locations)
#         avg_lon = sum(loc[1] for loc in locations) / len(locations)
#         center = [avg_lat, avg_lon]
    
#     m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    
#     # Add markers for each pothole
#     for lat, lon in locations:
#         folium.Marker(
#             location=[lat, lon],
#             popup="Pothole Detected",
#             icon=folium.Icon(color="red", icon="warning-sign")
#         ).add_to(m)
    
#     folium_static(m)

# # Function to add a new pothole location
# def add_pothole_location(lat, lon):
#     # Check if this location is already recorded (within a small radius)
#     for existing_lat, existing_lon in st.session_state.pothole_locations:
#         # Simple distance check (approximate)
#         if abs(existing_lat - lat) < 0.0001 and abs(existing_lon - lon) < 0.0001:
#             st.write("This pothole location is already recorded.")
#             return False
    
#     # Add new location
#     st.session_state.pothole_locations.append([lat, lon])
#     # Save to persistent storage
#     save_pothole_locations(st.session_state.pothole_locations)
#     return True

# # Function to create download link for video
# def get_video_download_link(video_path, link_text="Download processed video"):
#     """Generate a link to download the video file"""
#     with open(video_path, "rb") as file:
#         video_bytes = file.read()
#     b64 = base64.b64encode(video_bytes).decode()
    
#     # Get filename from path
#     filename = os.path.basename(video_path)
#     dl_link = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
#     return dl_link

# # Function to create MP4 video from AVI
# def convert_to_mp4(input_path):
#     """Convert AVI to MP4 format for better browser compatibility"""
#     output_path = os.path.splitext(input_path)[0] + ".mp4"
    
#     try:
#         # Read the input video
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             st.error(f"Could not open video file: {input_path}")
#             return None
            
#         # Get video properties
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
        
#         # Use H.264 codec for MP4
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         # Process video frame by frame
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
            
#         # Release resources
#         cap.release()
#         out.release()
        
#         if os.path.exists(output_path):
#             st.write(f"Successfully converted video to MP4: {output_path}")
#             return output_path
#         else:
#             st.error("Failed to create MP4 file")
#             return None
#     except Exception as e:
#         st.error(f"Error converting video: {e}")
        
#         # Try using FFmpeg as fallback if available
#         try:
#             import subprocess
#             st.write("Attempting conversion with FFmpeg...")
#             ffmpeg_cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -preset fast -crf 22 "{output_path}"'
#             result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
            
#             if os.path.exists(output_path):
#                 st.write("FFmpeg conversion successful")
#                 return output_path
#             else:
#                 st.error(f"FFmpeg conversion failed: {result.stderr}")
#                 return None
#         except Exception as ffmpeg_error:
#             st.error(f"FFmpeg fallback failed: {ffmpeg_error}")
#             return None

# # Extract frames as fallback method
# def extract_video_frames(video_path, max_frames=20):
#     """Extract frames from video as a fallback display method"""
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return []
            
#         frames = []
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Calculate step to evenly distribute frames
#         step = max(1, total_frames // max_frames)
        
#         for i in range(0, total_frames, step):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if ret:
#                 # Convert BGR to RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
            
#             if len(frames) >= max_frames:
#                 break
                
#         cap.release()
#         return frames
#     except Exception as e:
#         st.error(f"Error extracting frames: {e}")
#         return []

# # Streamlit UI
# st.title("Pothole Detection with YOLO")
# st.write("Choose an option: Upload a video, use your dashcam, or view the pothole map.")

# # Option selection
# detection_mode = st.radio("Select Detection Mode", ("Upload Video", "Real-Time Dashcam", "View Pothole Map", "Manage Pothole Data"))

# # Video upload mode
# if detection_mode == "Upload Video":
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         # Save the uploaded video temporarily
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")

#         # Run YOLO prediction
#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,
#                 half=True,
#                 imgsz=640
#             )

#         # Get the output path
#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#         # Verify and display output
#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
#             st.subheader("Processed Video with Potholes")
            
#             # Create a download link for the video
#             st.markdown(
#                 get_video_download_link(output_path, "⬇️ Download processed video"),
#                 unsafe_allow_html=True
#             )
            
#             # Try different methods to display the video
#             try:
#                 # 1. Try to convert to MP4 first (most compatible format)
#                 mp4_path = convert_to_mp4(output_path)
#                 if mp4_path and os.path.exists(mp4_path):
#                     st.video(mp4_path)
#                     st.success("Video converted to MP4 for better compatibility")
#                 else:
#                     # 2. Fallback to original format
#                     st.warning("MP4 conversion failed, trying original format")
#                     st.video(output_path)
#             except Exception as e:
#                 st.error(f"Video playback error: {str(e)}")
                
#                 # 3. Show video frames as fallback
#                 st.write("Showing video frames as fallback:")
#                 frames = extract_video_frames(output_path)
#                 if frames:
#                     col1, col2 = st.columns(2)
#                     for i, frame in enumerate(frames):
#                         if i % 2 == 0:
#                             with col1:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                         else:
#                             with col2:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                 else:
#                     st.error("Could not extract frames from video")

#             # Plot the current location as having a pothole
#             lat, lon = get_current_location()
#             if add_pothole_location(lat, lon):
#                 st.write(f"Pothole recorded at Latitude {lat}, Longitude {lon}")
#             plot_pothole_map(st.session_state.pothole_locations)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")
            
#             # Check if there's any output in the runs directory
#             detect_dirs = [d for d in os.listdir("runs/detect") if d.startswith("predict")]
#             if detect_dirs:
#                 latest_dir = max(detect_dirs, key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
#                 st.write(f"Latest output directory: runs/detect/{latest_dir}")
#                 files_in_dir = os.listdir(os.path.join("runs/detect", latest_dir))
#                 st.write(f"Files in directory: {files_in_dir}")

# # Real-time dashcam mode
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")

#     # Initialize session state variables
#     if "running" not in st.session_state:
#         st.session_state.running = False
#     if "detection_frames" not in st.session_state:
#         st.session_state.detection_frames = []

#     # Control buttons
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Start", key="start_button"):
#             st.session_state.running = True
#             st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
#     with col2:
#         if st.button("Stop", key="stop_button"):
#             st.session_state.running = False
#             st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

#     # Create placeholders for video display
#     video_placeholder = st.empty()
#     status_text = st.empty()
#     map_placeholder = st.empty()

#     # Only run camera when state is running
#     if st.session_state.running:
#         status_text.info("Starting camera... Please wait.")
        
#         try:
#             # Initialize camera
#             cap = cv2.VideoCapture(0)
            
#             if not cap.isOpened():
#                 st.error("Error: Could not open camera. Ensure a camera is connected.")
#                 st.session_state.running = False
#             else:
#                 status_text.success("Camera connected! Displaying live feed...")
                
#                 # Get a single frame to test
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     cap.release()
#                     st.session_state.running = False
#                 else:
#                     # Display the first frame to confirm camera works
#                     first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     video_placeholder.image(first_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
                    
#                     # Now capture a limited number of frames in a loop
#                     # This prevents Streamlit from hanging
#                     detection_cooldown = 0
#                     FRAME_LIMIT = 100  # Process 100 frames before requiring refresh
                    
#                     for frame_count in range(FRAME_LIMIT):
#                         if not st.session_state.running:
#                             break
                            
#                         ret, frame = cap.read()
#                         if not ret:
#                             st.error("Error reading frame")
#                             break
                            
#                         # Run YOLO prediction
#                         results = model.predict(
#                             source=frame,
#                             conf=0.25,
#                             half=True,
#                             imgsz=640,  # Match the input size expected by the model (640x640)
#                             verbose=False
#                         )
                        
#                         # Check for pothole detection with cooldown
#                         pothole_detected = False
#                         for result in results:
#                             if result.boxes and detection_cooldown <= 0:
#                                 pothole_detected = True
#                                 detection_cooldown = 15  # Lower cooldown for more responsive detection
#                                 break
                                
#                         # Decrement cooldown
#                         if detection_cooldown > 0:
#                             detection_cooldown -= 1
                            
#                         # Handle pothole detection
#                         if pothole_detected:
#                             lat, lon = get_current_location()
#                             if add_pothole_location(lat, lon):
#                                 status_text.warning(f"Pothole detected! Location: {lat:.6f}, {lon:.6f}")
                                
#                                 # Save detection frame
#                                 detection_frame = results[0].plot().copy()
#                                 detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
#                                 st.session_state.detection_frames.append(detection_frame_rgb)
                                
#                                 # Update map occasionally (not every frame)
#                                 if frame_count % 10 == 0:
#                                     with map_placeholder:
#                                         plot_pothole_map(st.session_state.pothole_locations)
                        
#                         # Display the current frame
#                         annotated_frame = results[0].plot()
#                         annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                         video_placeholder.image(annotated_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
                        
#                         # Small delay to prevent UI freezing (adjust as needed)
#                         time.sleep(0.05)  # ~20 FPS
                        
#                     # After frame limit, provide button to continue
#                     status_text.info(f"Processed {FRAME_LIMIT} frames. Click 'Continue' to keep recording.")
#                     if st.button("Continue Recording"):
#                         st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
                
#                 # Release the camera when done
#                 cap.release()
        
#         except Exception as e:
#             st.error(f"Camera error: {str(e)}")
#             st.session_state.running = False
        
#         finally:
#             # Always try to release camera on error
#             try:
#                 cap.release()
#             except:
#                 pass
    
#     else:
#         st.write("Press 'Start' to begin real-time detection.")
    
#     # Display detection frames gallery (show regardless of camera state)
#     if st.session_state.detection_frames:
#         st.subheader("Pothole Detections")
        
#         # Limit to most recent 9 frames to prevent UI clutter
#         recent_frames = st.session_state.detection_frames[-9:]
#         cols = st.columns(min(3, len(recent_frames)))
#         for i, frame in enumerate(recent_frames):
#             cols[i % len(cols)].image(frame, caption=f"Detection {i+1}", use_column_width=True)

# # View pothole map mode
# elif detection_mode == "View Pothole Map":
#     st.subheader("Pothole Map")
#     st.write(f"Showing {len(st.session_state.pothole_locations)} recorded pothole locations on OpenStreetMap.")
#     plot_pothole_map(st.session_state.pothole_locations)

# # Manage pothole data mode
# elif detection_mode == "Manage Pothole Data":
#     st.subheader("Manage Pothole Data")
    
#     # Option to add a manual pothole location
#     st.write("Add a new pothole location manually:")
#     col1, col2 = st.columns(2)
#     with col1:
#         manual_lat = st.number_input("Latitude", value=get_current_location()[0], format="%.6f")
#     with col2:
#         manual_lon = st.number_input("Longitude", value=get_current_location()[1], format="%.6f")
    
#     if st.button("Add Manual Location"):
#         if add_pothole_location(manual_lat, manual_lon):
#             st.success(f"Added new pothole location at {manual_lat}, {manual_lon}")
        
#     # Option to clear all pothole data
#     if st.button("Clear All Pothole Data"):
#         confirm = st.checkbox("I confirm I want to delete all pothole data")
#         if confirm:
#             st.session_state.pothole_locations = []
#             save_pothole_locations([])  # Save empty list to file
#             st.success("All pothole location data has been cleared.")
    
#     # Show the current data
#     st.write("Current pothole locations:")
#     if st.session_state.pothole_locations:
#         location_df = {"Latitude": [loc[0] for loc in st.session_state.pothole_locations],
#                        "Longitude": [loc[1] for loc in st.session_state.pothole_locations]}
#         st.dataframe(location_df)
#         plot_pothole_map(st.session_state.pothole_locations)
#     else:
#         st.write("No pothole locations recorded yet.")
    
#     # Option to export data
#     if st.button("Export Data as JSON") and st.session_state.pothole_locations:
#         st.download_button(
#             label="Download JSON",
#             data=json.dumps(st.session_state.pothole_locations),
#             file_name="pothole_locations.json",
#             mime="application/json"
#         )



# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time
# import geocoder
# import folium
# from streamlit_folium import folium_static
# import json
# import base64
# import tempfile
# import shutil
# from streamlit_javascript import st_javascript
# # import streamlit as st
# # import geocoder
# from streamlit_js_eval import streamlit_js_eval  # Third-party hack to run JavaScript in Streamlit




# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"
# # Path for storing pothole locations
# pothole_data_path = os.path.join(HOME, "pothole_locations.json")

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # Function to save pothole locations to file
# def save_pothole_locations(locations):
#     with open(pothole_data_path, 'w') as f:
#         json.dump(locations, f)
#     st.write(f"Saved {len(locations)} pothole locations to {pothole_data_path}")


# import streamlit as st
# import geocoder
# from streamlit_js_eval import streamlit_js_eval  # Third-party hack to run JavaScript in Streamlit

# # Function to get current location using browser GPS with fallback to IP geolocation
# def get_current_location(context_id=None, timeout=10000):
#     # Try to get GPS location from browser first with increased timeout
#     try:
#         # Use streamlit_js_eval to execute JavaScript and get GPS location
#         location_data = streamlit_js_eval(
#             js_expressions="navigator.geolocation.getCurrentPosition(position => position.coords)",
#             key="get_location"
#         )

#         if location_data and 'latitude' in location_data and 'longitude' in location_data:
#             st.write(f"GPS location obtained with accuracy: {location_data.get('accuracy', 'unknown')} meters")
#             return [location_data['latitude'], location_data['longitude']]
#         else:
#             st.warning("Could not get GPS location. Falling back to IP geolocation.")
#     except Exception as e:
#         st.warning(f"Error getting GPS location: {e}. Falling back to IP geolocation.")
    
#     # Fallback to IP-based geolocation
#     try:
#         g = geocoder.ip('me')
#         if g.ok:
#             st.info("Using IP-based location (less accurate)")
#             return g.latlng  # Returns [lat, lon]
#         else:
#             # Try additional geocoding services
#             try:
#                 g = geocoder.ipinfo()
#                 if g.ok:
#                     st.info("Using IPInfo-based location")
#                     return g.latlng
#             except Exception:
#                 pass
                
#             # If all else fails, try a different service
#             try:
#                 g = geocoder.geoname('me', key='demo')
#                 if g.ok:
#                     st.info("Using Geoname-based location")
#                     return g.latlng
#             except Exception:
#                 pass
                
#             st.warning("Could not retrieve location. Using default location.")
#             # Default to Idukki, Kerala
#             return [9.9189, 76.9383]
#     except Exception as e:
#         st.error(f"Error retrieving location: {e}")
#         # Default to Idukki, Kerala
#         return [9.9189, 76.9383]








# # Function to load pothole locations from file
# def load_pothole_locations():
#     if os.path.exists(pothole_data_path):
#         try:
#             with open(pothole_data_path, 'r') as f:
#                 locations = json.load(f)
#             st.write(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             return locations
#         except Exception as e:
#             st.error(f"Error loading pothole locations: {e}")
#             return []
#     else:
#         st.write("No saved pothole locations found. Starting with empty map.")
#         return []

# # Initialize session state for storing pothole locations
# if "pothole_locations" not in st.session_state:
#     st.session_state.pothole_locations = load_pothole_locations()

# # # Function to get current location using browser GPS with fallback to IP geolocation
# # def get_current_location(context_id=None, timeout=10000):
# #     # Try to get GPS location from browser first with increased timeout
# #     try:
# #         # Run JavaScript to get location from browser
# #         location_data = st_javascript("""
# #             async function getLocation() {
# #                 return new Promise((resolve, reject) => {
# #                     if (!navigator.geolocation) {
# #                         reject("Geolocation not supported by your browser");
# #                     }
                    
# #                     navigator.geolocation.getCurrentPosition(
# #                         position => {
# #                             const pos = {
# #                                 lat: position.coords.latitude,
# #                                 lng: position.coords.longitude,
# #                                 accuracy: position.coords.accuracy
# #                             };
# #                             resolve(pos);
# #                         },
# #                         error => {
# #                             reject(`ERROR(${error.code}): ${error.message}`);
# #                         },
# #                         {
# #                             enableHighAccuracy: true,
# #                             timeout: """ + str(timeout) + """,
# #                             maximumAge: 0
# #                         }
# #                     );
# #                 });
# #             }
            
# #             const location = await getLocation();
# #             return location;
# #         """)
        
# #         if location_data and 'lat' in location_data and 'lng' in location_data:
# #             st.write(f"GPS location obtained with accuracy: {location_data.get('accuracy', 'unknown')} meters")
# #             return [location_data['lat'], location_data['lng']]
# #         else:
# #             st.warning("Could not get GPS location. Falling back to IP geolocation.")
# #     except Exception as e:
# #         st.warning(f"Error getting GPS location: {e}. Falling back to IP geolocation.")
    
# #     # Fallback to IP-based geolocation
# #     try:
# #         g = geocoder.ip('me')
# #         if g.ok:
# #             st.info("Using IP-based location (less accurate)")
# #             return g.latlng  # Returns [lat, lon]
# #         else:
# #             # Try additional geocoding services
# #             try:
# #                 g = geocoder.ipinfo()
# #                 if g.ok:
# #                     st.info("Using IPInfo-based location")
# #                     return g.latlng
# #             except Exception:
# #                 pass
                
# #             # If all else fails, try a different service
# #             try:
# #                 g = geocoder.geoname('me', key='demo')
# #                 if g.ok:
# #                     st.info("Using Geoname-based location")
# #                     return g.latlng
# #             except Exception:
# #                 pass
                
# #             st.warning("Could not retrieve location. Using default location.")
# #             # Default to Idukki, Kerala
# #             return [9.9189, 76.9383]
# #     except Exception as e:
# #         st.error(f"Error retrieving location: {e}")
# #         # Default to Idukki, Kerala
# #         return [9.9189, 76.9383]

# # Function to plot pothole map
# def plot_pothole_map(locations, context_id=None):
#     if not locations:
#         st.write("No pothole locations recorded yet.")
#         return
    
#     # Use the first location to center the map, or calculate an average center
#     if len(locations) == 1:
#         center = locations[0]
#     else:
#         # Calculate the average lat/lon for centering the map
#         avg_lat = sum(loc[0] for loc in locations) / len(locations)
#         avg_lon = sum(loc[1] for loc in locations) / len(locations)
#         center = [avg_lat, avg_lon]
    
#     m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    
#     # Add markers for each pothole
#     for lat, lon in locations:
#         folium.Marker(
#             location=[lat, lon],
#             popup="Pothole Detected",
#             icon=folium.Icon(color="red", icon="warning-sign")
#         ).add_to(m)
    
#     # Just use folium_static without the key parameter
#     folium_static(m)

# # Function to add a new pothole location
# def add_pothole_location(lat, lon):
#     # Check if this location is already recorded (within a small radius)
#     for existing_lat, existing_lon in st.session_state.pothole_locations:
#         # Simple distance check (approximate)
#         if abs(existing_lat - lat) < 0.0001 and abs(existing_lon - lon) < 0.0001:
#             st.write("This pothole location is already recorded.")
#             return False
    
#     # Add new location
#     st.session_state.pothole_locations.append([lat, lon])
#     # Save to persistent storage
#     save_pothole_locations(st.session_state.pothole_locations)
#     return True

# # Function to create download link for video
# def get_video_download_link(video_path, link_text="Download processed video"):
#     """Generate a link to download the video file"""
#     with open(video_path, "rb") as file:
#         video_bytes = file.read()
#     b64 = base64.b64encode(video_bytes).decode()
    
#     # Get filename from path
#     filename = os.path.basename(video_path)
#     dl_link = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
#     return dl_link

# # Function to create MP4 video from AVI
# def convert_to_mp4(input_path):
#     """Convert AVI to MP4 format for better browser compatibility"""
#     output_path = os.path.splitext(input_path)[0] + ".mp4"
    
#     try:
#         # Read the input video
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             st.error(f"Could not open video file: {input_path}")
#             return None
            
#         # Get video properties
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
        
#         # Use H.264 codec for MP4
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         # Process video frame by frame
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
            
#         # Release resources
#         cap.release()
#         out.release()
        
#         if os.path.exists(output_path):
#             st.write(f"Successfully converted video to MP4: {output_path}")
#             return output_path
#         else:
#             st.error("Failed to create MP4 file")
#             return None
#     except Exception as e:
#         st.error(f"Error converting video: {e}")
        
#         # Try using FFmpeg as fallback if available
#         try:
#             import subprocess
#             st.write("Attempting conversion with FFmpeg...")
#             ffmpeg_cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -preset fast -crf 22 "{output_path}"'
#             result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
            
#             if os.path.exists(output_path):
#                 st.write("FFmpeg conversion successful")
#                 return output_path
#             else:
#                 st.error(f"FFmpeg conversion failed: {result.stderr}")
#                 return None
#         except Exception as ffmpeg_error:
#             st.error(f"FFmpeg fallback failed: {ffmpeg_error}")
#             return None

# # Extract frames as fallback method
# def extract_video_frames(video_path, max_frames=20):
#     """Extract frames from video as a fallback display method"""
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return []
            
#         frames = []
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Calculate step to evenly distribute frames
#         step = max(1, total_frames // max_frames)
        
#         for i in range(0, total_frames, step):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if ret:
#                 # Convert BGR to RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
            
#             if len(frames) >= max_frames:
#                 break
                
#         cap.release()
#         return frames
#     except Exception as e:
#         st.error(f"Error extracting frames: {e}")
#         return []

# # Streamlit UI
# st.title("Pothole Detection with YOLO")
# st.write("Choose an option: Upload a video, use your dashcam, or view the pothole map.")

# # Option selection
# detection_mode = st.radio("Select Detection Mode", ("Upload Video", "Real-Time Dashcam", "View Pothole Map", "Manage Pothole Data"))

# # Video upload mode
# if detection_mode == "Upload Video":
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         # Save the uploaded video temporarily
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")

#         # Run YOLO prediction
#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,
#                 half=True,
#                 imgsz=640
#             )

#         # Get the output path
#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#         # Create a placeholder for the video display
#         video_display = st.empty()
        
#         # Create a placeholder for the map
#         map_placeholder = st.empty()

#         # Verify and display output
#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
#             st.subheader("Processed Video with Potholes")
            
#             # Create a download link for the video
#             st.markdown(
#                 get_video_download_link(output_path, "⬇️ Download processed video"),
#                 unsafe_allow_html=True
#             )
            
#             # Try different methods to display the video
#             try:
#                 # 1. Try to convert to MP4 first (most compatible format)
#                 mp4_path = convert_to_mp4(output_path)
#                 if mp4_path and os.path.exists(mp4_path):
#                     with video_display:
#                         st.video(mp4_path)
#                     st.success("Video converted to MP4 for better compatibility")
#                 else:
#                     # 2. Fallback to original format
#                     st.warning("MP4 conversion failed, trying original format")
#                     with video_display:
#                         st.video(output_path)
#             except Exception as e:
#                 st.error(f"Video playback error: {str(e)}")
                
#                 # 3. Show video frames as fallback
#                 st.write("Showing video frames as fallback:")
#                 frames = extract_video_frames(output_path)
#                 if frames:
#                     col1, col2 = st.columns(2)
#                     for i, frame in enumerate(frames):
#                         if i % 2 == 0:
#                             with col1:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                         else:
#                             with col2:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                 else:
#                     st.error("Could not extract frames from video")

#             # Plot the current location as having a pothole
#             lat, lon = get_current_location(timeout=15000)
#             if add_pothole_location(lat, lon):
#                 st.write(f"Pothole recorded at Latitude {lat}, Longitude {lon}")
            
#             with map_placeholder:
#                 plot_pothole_map(st.session_state.pothole_locations)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")
            
#             # Check if there's any output in the runs directory
#             detect_dirs = [d for d in os.listdir("runs/detect") if d.startswith("predict")]
#             if detect_dirs:
#                 latest_dir = max(detect_dirs, key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
#                 st.write(f"Latest output directory: runs/detect/{latest_dir}")
#                 files_in_dir = os.listdir(os.path.join("runs/detect", latest_dir))
#                 st.write(f"Files in directory: {files_in_dir}")

# # Real-time dashcam mode
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")

#     # Initialize session state variables
#     if "running" not in st.session_state:
#         st.session_state.running = False
#     if "detection_frames" not in st.session_state:
#         st.session_state.detection_frames = []

#     # Control buttons
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Start"):
#             st.session_state.running = True
#             st.rerun()
#     with col2:
#         if st.button("Stop"):
#             st.session_state.running = False
#             st.rerun()

#     # Create placeholders for video display
#     video_placeholder = st.empty()
#     status_text = st.empty()
#     map_placeholder = st.empty()

#     # Only run camera when state is running
#     if st.session_state.running:
#         status_text.info("Starting camera... Please wait.")
        
#         try:
#             # Initialize camera
#             cap = cv2.VideoCapture(0)
            
#             if not cap.isOpened():
#                 st.error("Error: Could not open camera. Ensure a camera is connected.")
#                 st.session_state.running = False
#             else:
#                 status_text.success("Camera connected! Displaying live feed...")
                
#                 # Get a single frame to test
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     cap.release()
#                     st.session_state.running = False
#                 else:
#                     # Display the first frame to confirm camera works
#                     first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     video_placeholder.image(first_frame_rgb, caption="Live Dashcam Feed", use_container_width=True)
                    
#                     # Get current location once
#                     current_location = get_current_location(timeout=15000)
                    
#                     # Now capture a limited number of frames in a loop
#                     # This prevents Streamlit from hanging
#                     detection_cooldown = 0
#                     FRAME_LIMIT = 100  # Process 100 frames before requiring refresh
                    
#                     for frame_count in range(FRAME_LIMIT):
#                         if not st.session_state.running:
#                             break
                            
#                         ret, frame = cap.read()
#                         if not ret:
#                             st.error("Error reading frame")
#                             break
                            
#                         # Run YOLO prediction
#                         results = model.predict(
#                             source=frame,
#                             conf=0.25,
#                             half=True,
#                             imgsz=640,  # Match the input size expected by the model (640x640)
#                             verbose=False
#                         )
                        
#                         # Check for pothole detection with cooldown
#                         pothole_detected = False
#                         for result in results:
#                             if result.boxes and detection_cooldown <= 0:
#                                 pothole_detected = True
#                                 detection_cooldown = 15  # Lower cooldown for more responsive detection
#                                 break
                                
#                         # Decrement cooldown
#                         if detection_cooldown > 0:
#                             detection_cooldown -= 1
                            
#                         # Handle pothole detection
#                         if pothole_detected:
#                             if add_pothole_location(current_location[0], current_location[1]):
#                                 status_text.warning(f"Pothole detected! Location: {current_location[0]:.6f}, {current_location[1]:.6f}")
                                
#                                 # Save detection frame
#                                 detection_frame = results[0].plot().copy()
#                                 detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
#                                 st.session_state.detection_frames.append(detection_frame_rgb)
                                
#                                 # Update map occasionally (not every frame)
#                                 if frame_count % 10 == 0:
#                                     with map_placeholder:
#                                         plot_pothole_map(st.session_state.pothole_locations)
                        
#                         # Display the current frame
#                         annotated_frame = results[0].plot()
#                         annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                         video_placeholder.image(
#                             annotated_frame_rgb, 
#                             caption="Live Dashcam Feed", 
#                             use_container_width=True
#                         )
                        
#                         # Small delay to prevent UI freezing (adjust as needed)
#                         time.sleep(0.05)  # ~20 FPS
                        
#                     # After frame limit, provide button to continue
#                     status_text.info(f"Processed {FRAME_LIMIT} frames. Click 'Continue' to keep recording.")
#                     if st.button("Continue Recording"):
#                         st.rerun()
                
#                 # Release the camera when done
#                 cap.release()
        
#         except Exception as e:
#             st.error(f"Camera error: {str(e)}")
#             st.session_state.running = False
        
#         finally:
#             # Always try to release camera on error
#             try:
#                 cap.release()
#             except:
#                 pass
    
#     else:
#         st.write("Press 'Start' to begin real-time detection.")
    
#     # Display detection frames gallery (show regardless of camera state)
#     if st.session_state.detection_frames:
#         st.subheader("Pothole Detections")
        
#         # Limit to most recent 9 frames to prevent UI clutter
#         recent_frames = st.session_state.detection_frames[-9:]
#         cols = st.columns(min(3, len(recent_frames)))
#         for i, frame in enumerate(recent_frames):
#             cols[i % len(cols)].image(
#                 frame, 
#                 caption=f"Detection {i+1}", 
#                 use_column_width=True
#             )

# # View pothole map mode
# elif detection_mode == "View Pothole Map":
#     st.subheader("Pothole Map")
#     st.write(f"Showing {len(st.session_state.pothole_locations)} recorded pothole locations on OpenStreetMap.")
    
#     # Create a placeholder for the map
#     map_placeholder = st.empty()
    
#     # Create a button to refresh the map with current location
#     if st.button("Refresh Map with Current Location"):
#         current_location = get_current_location(timeout=15000)
#         st.write(f"Current location: {current_location[0]:.6f}, {current_location[1]:.6f}")
    
#     # Display the map in the placeholder
#     with map_placeholder:
#         plot_pothole_map(st.session_state.pothole_locations)

# # Manage pothole data mode
# elif detection_mode == "Manage Pothole Data":
#     st.subheader("Manage Pothole Data")
    
#     # Create placeholders
#     location_status = st.empty()
#     map_placeholder = st.empty()
    
#     # Get current location for default values (with increased timeout)
#     default_location = get_current_location(timeout=15000)
    
#     # Option to add a manual pothole location
#     st.write("Add a new pothole location manually:")
#     col1, col2 = st.columns(2)
#     with col1:
#         manual_lat = st.number_input(
#             "Latitude", 
#             value=default_location[0], 
#             format="%.6f"
#         )
#     with col2:
#         manual_lon = st.number_input(
#             "Longitude", 
#             value=default_location[1], 
#             format="%.6f"
#         )
    
#     if st.button("Add Manual Location"):
#         if add_pothole_location(manual_lat, manual_lon):
#             location_status.success(f"Added new pothole location at {manual_lat}, {manual_lon}")
#             # Update the map placeholder
#             with map_placeholder:
#                 plot_pothole_map(st.session_state.pothole_locations)
        
#     # Option to clear all pothole data
#     if st.button("Clear All Pothole Data"):
#         confirm = st.checkbox("I confirm I want to delete all pothole data")
#         if confirm:
#             st.session_state.pothole_locations = []
#             save_pothole_locations([])  # Save empty list to file
#             location_status.success("All pothole location data has been cleared.")
#             # Update the map placeholder
#             with map_placeholder:
#                 plot_pothole_map(st.session_state.pothole_locations)
    
#     # Show the current data
#     st.write("Current pothole locations:")
#     if st.session_state.pothole_locations:
#         location_df = {"Latitude": [loc[0] for loc in st.session_state.pothole_locations],
#                       "Longitude": [loc[1] for loc in st.session_state.pothole_locations]}
#         st.dataframe(location_df)
#         with map_placeholder:
#             plot_pothole_map(st.session_state.pothole_locations)
#     else:
#         st.write("No pothole locations recorded yet.")
    
#     # Option to export data
#     if st.button("Export Data as JSON") and st.session_state.pothole_locations:
#         st.download_button(
#             label="Download JSON",
#             data=json.dumps(st.session_state.pothole_locations),
#             file_name="pothole_locations.json",
#             mime="application/json"
#         )



# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time
# import geocoder
# import folium
# from streamlit_folium import folium_static
# import json
# import base64
# import tempfile
# import shutil
# from streamlit_javascript import st_javascript
# from streamlit_js_eval import streamlit_js_eval
# import requests
# import logging

# # Set page config as the first Streamlit command
# st.set_page_config(page_title="Pothole Detection Dashboard", page_icon="🛣️", layout="wide")

# # Configure logging
# logging.basicConfig(
#     filename='pothole_detection.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"
# pothole_data_path = os.path.join(HOME, "pothole_locations.json")

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # API endpoint
# API_ENDPOINT = "https://location-api-production-e673.up.railway.app/api/location"

# # Function to send location to API
# def send_location_to_api(lat, lon, marker_type="pothole"):
#     payload = {
#         "latitude": lat,
#         "longitude": lon
#         # "type": marker_type
#     }
#     try:
#         response = requests.post(API_ENDPOINT, json=payload)
#         if response.status_code == 200 or response.status_code == 201:
#             logger.info(f"Successfully sent {marker_type} location to API: {lat}, {lon} - Response: {response.text}")
#             return True
#         else:
#             logger.error(f"Failed to send {marker_type} location to API: {lat}, {lon} - Status: {response.status_code}")
#             return False
#     except Exception as e:
#         logger.error(f"Error sending {marker_type} location to API: {lat}, {lon} - {str(e)}")
#         return False

# # Function to save pothole locations to file
# def save_pothole_locations(locations):
#     with open(pothole_data_path, 'w') as f:
#         json.dump(locations, f)
#     st.write(f"Saved {len(locations)} pothole locations to {pothole_data_path}")
#     logger.info(f"Saved {len(locations)} pothole locations to {pothole_data_path}")

# # Function to get current location using browser GPS with fallback to IP geolocation
# def get_current_location(context_id=None, timeout=10000):
#     try:
#         location_data = streamlit_js_eval(
#             js_expressions="navigator.geolocation.getCurrentPosition(position => position.coords)",
#             key="get_location"
#         )
#         if location_data and 'latitude' in location_data and 'longitude' in location_data:
#             st.write(f"GPS location obtained with accuracy: {location_data.get('accuracy', 'unknown')} meters")
#             logger.info(f"GPS location obtained: {location_data['latitude']}, {location_data['longitude']}")
#             return [location_data['latitude'], location_data['longitude']]
#         else:
#             st.warning("Could not get GPS location. Falling back to IP geolocation.")
#     except Exception as e:
#         st.warning(f"Error getting GPS location: {e}. Falling back to IP geolocation.")
    
#     try:
#         g = geocoder.ip('me')
#         if g.ok:
#             st.info("Using IP-based location (less accurate)")
#             logger.info(f"IP-based location: {g.latlng}")
#             return g.latlng
#         else:
#             try:
#                 g = geocoder.ipinfo()
#                 if g.ok:
#                     st.info("Using IPInfo-based location")
#                     logger.info(f"IPInfo-based location: {g.latlng}")
#                     return g.latlng
#             except Exception:
#                 pass
#             try:
#                 g = geocoder.geoname('me', key='demo')
#                 if g.ok:
#                     st.info("Using Geoname-based location")
#                     logger.info(f"Geoname-based location: {g.latlng}")
#                     return g.latlng
#             except Exception:
#                 pass
#             st.warning("Could not retrieve location. Using default location.")
#             logger.warning("Failed to retrieve location, using default: [9.9189, 76.9383]")
#             return [9.5860236,76.983061]
#     except Exception as e:
#         st.error(f"Error retrieving location: {e}")
#         logger.error(f"Error retrieving location: {e}")
#         return [9.5860236,76.983061]

# # Function to load pothole locations from file
# def load_pothole_locations():
#     if os.path.exists(pothole_data_path):
#         try:
#             with open(pothole_data_path, 'r') as f:
#                 locations = json.load(f)
#             st.write(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             logger.info(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             return locations
#         except Exception as e:
#             st.error(f"Error loading pothole locations: {e}")
#             logger.error(f"Error loading pothole locations: {e}")
#             return []
#     else:
#         st.write("No saved pothole locations found. Starting with empty map.")
#         logger.info("No saved pothole locations found")
#         return []

# # Initialize session state for storing locations (now including type)
# if "locations" not in st.session_state:
#     st.session_state.locations = load_pothole_locations()

# # Function to plot map with different marker types (Fixed for backward compatibility)
# def plot_pothole_map(locations):
#     if not locations:
#         st.write("No locations recorded yet.")
#         return
    
#     avg_lat = sum(loc[0] for loc in locations) / len(locations)
#     avg_lon = sum(loc[1] for loc in locations) / len(locations)
#     center = [avg_lat, avg_lon]
    
#     m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    
#     marker_styles = {
#         "pothole": {"color": "red", "icon": "warning-sign"},
#         "flooded_road": {"color": "blue", "icon": "tint"},
#         "fallen_tree": {"color": "green", "icon": "tree-deciduous"},
#         "construction": {"color": "orange", "icon": "wrench"},
#         "landslide": {"color": "purple", "icon": "collapse-down"}
#     }
    
#     for loc in locations:
#         if len(loc) == 2:  # Old format: [lat, lon]
#             lat, lon = loc
#             marker_type = "pothole"  # Default to pothole
#         elif len(loc) == 3:  # New format: [lat, lon, marker_type]
#             lat, lon, marker_type = loc
#         else:
#             continue  # Skip invalid entries
#         style = marker_styles.get(marker_type, marker_styles["pothole"])
#         folium.Marker(
#             location=[lat, lon],
#             popup=f"{marker_type.replace('_', ' ').title()} Detected",
#             icon=folium.Icon(color=style["color"], icon=style["icon"])
#         ).add_to(m)
    
#     folium_static(m)

# # Function to add a new location
# def add_location(lat, lon, marker_type="pothole"):
#     for existing_lat, existing_lon, *_ in st.session_state.locations:  # Use *_ to handle variable length
#         if abs(existing_lat - lat) < 0.0001 and abs(existing_lon - lon) < 0.0001:
#             st.write(f"This {marker_type} location is already recorded.")
#             return False
    
#     st.session_state.locations.append([lat, lon, marker_type])
#     save_pothole_locations(st.session_state.locations)
#     if send_location_to_api(lat, lon, marker_type):
#         st.success(f"{marker_type.replace('_', ' ').title()} location sent to API successfully!")
#     else:
#         st.error(f"Failed to send {marker_type} location to API.")
#     return True

# # Function to create download link for video
# def get_video_download_link(video_path, link_text="Download processed video"):
#     with open(video_path, "rb") as file:
#         video_bytes = file.read()
#     b64 = base64.b64encode(video_bytes).decode()
#     filename = os.path.basename(video_path)
#     dl_link = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
#     return dl_link

# # Function to convert to MP4
# def convert_to_mp4(input_path):
#     output_path = os.path.splitext(input_path)[0] + ".mp4"
#     try:
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             st.error(f"Could not open video file: {input_path}")
#             return None
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
#         cap.release()
#         out.release()
#         if os.path.exists(output_path):
#             st.write(f"Successfully converted video to MP4: {output_path}")
#             return output_path
#         else:
#             st.error("Failed to create MP4 file")
#             return None
#     except Exception as e:
#         st.error(f"Error converting video: {e}")
#         try:
#             import subprocess
#             st.write("Attempting conversion with FFmpeg...")
#             ffmpeg_cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -preset fast -crf 22 "{output_path}"'
#             result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
#             if os.path.exists(output_path):
#                 st.write("FFmpeg conversion successful")
#                 return output_path
#             else:
#                 st.error(f"FFmpeg conversion failed: {result.stderr}")
#                 return None
#         except Exception as ffmpeg_error:
#             st.error(f"FFmpeg fallback failed: {ffmpeg_error}")
#             return None

# # Extract frames as fallback method
# def extract_video_frames(video_path, max_frames=20):
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return []
#         frames = []
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         step = max(1, total_frames // max_frames)
#         for i in range(0, total_frames, step):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if ret:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
#             if len(frames) >= max_frames:
#                 break
#         cap.release()
#         return frames
#     except Exception as e:
#         st.error(f"Error extracting frames: {e}")
#         return []

# # Streamlit UI with Beautification
# st.title("🛣️ Pothole Detection Dashboard")
# st.markdown("""
#     <style>
#     .main {background-color: #f0f2f6;}
#     .stButton>button {background-color: #4CAF50; color: white;}
#     .stRadio>label {font-size: 18px;}
#     </style>
# """, unsafe_allow_html=True)

# st.write("Choose an option below to get started:")
# detection_mode = st.radio(
#     "Select Detection Mode",
#     ("Upload Video", "Real-Time Dashcam", "View Map", "Manage Data"),
#     horizontal=True
# )

# # Video upload mode
# if detection_mode == "Upload Video":
#     st.subheader("📹 Upload Video for Pothole Detection")
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")

#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,
#                 half=True,
#                 imgsz=640
#             )

#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#         video_display = st.empty()
#         map_placeholder = st.empty()

#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
#             st.subheader("Processed Video with Potholes")
#             st.markdown(
#                 get_video_download_link(output_path, "⬇️ Download processed video"),
#                 unsafe_allow_html=True
#             )
#             try:
#                 mp4_path = convert_to_mp4(output_path)
#                 if mp4_path and os.path.exists(mp4_path):
#                     with video_display:
#                         st.video(mp4_path)
#                     st.success("Video converted to MP4 for better compatibility")
#                 else:
#                     st.warning("MP4 conversion failed, trying original format")
#                     with video_display:
#                         st.video(output_path)
#             except Exception as e:
#                 st.error(f"Video playback error: {str(e)}")
#                 st.write("Showing video frames as fallback:")
#                 frames = extract_video_frames(output_path)
#                 if frames:
#                     col1, col2 = st.columns(2)
#                     for i, frame in enumerate(frames):
#                         if i % 2 == 0:
#                             with col1:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                         else:
#                             with col2:
#                                 st.image(frame, caption=f"Frame {i}", use_column_width=True)
#                 else:
#                     st.error("Could not extract frames from video")

#             lat, lon = get_current_location(timeout=15000)
#             if add_location(lat, lon, "pothole"):
#                 st.write(f"Pothole recorded at Latitude {lat}, Longitude {lon}")
#             with map_placeholder:
#                 plot_pothole_map(st.session_state.locations)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")
#             detect_dirs = [d for d in os.listdir("runs/detect") if d.startswith("predict")]
#             if detect_dirs:
#                 latest_dir = max(detect_dirs, key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
#                 st.write(f"Latest output directory: runs/detect/{latest_dir}")
#                 files_in_dir = os.listdir(os.path.join("runs/detect", latest_dir))
#                 st.write(f"Files in directory: {files_in_dir}")

# # Real-time dashcam mode (Fixed video feed size)
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("📷 Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")

#     if "running" not in st.session_state:
#         st.session_state.running = False
#     if "detection_frames" not in st.session_state:
#         st.session_state.detection_frames = []

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Start"):
#             st.session_state.running = True
#             st.rerun()
#     with col2:
#         if st.button("Stop"):
#             st.session_state.running = False
#             st.rerun()

#     video_placeholder = st.empty()
#     status_text = st.empty()
#     map_placeholder = st.empty()

#     if st.session_state.running:
#         status_text.info("Starting camera... Please wait.")
#         try:
#             cap = cv2.VideoCapture(0)
#             if not cap.isOpened():
#                 st.error("Error: Could not open camera. Ensure a camera is connected.")
#                 st.session_state.running = False
#             else:
#                 status_text.success("Camera connected! Displaying live feed...")
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     cap.release()
#                     st.session_state.running = False
#                 else:
#                     first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     video_placeholder.image(first_frame_rgb, caption="Live Dashcam Feed", width=640)  # Fixed width
#                     current_location = get_current_location(timeout=15000)
#                     detection_cooldown = 0
#                     FRAME_LIMIT = 100
#                     for frame_count in range(FRAME_LIMIT):
#                         if not st.session_state.running:
#                             break
#                         ret, frame = cap.read()
#                         if not ret:
#                             st.error("Error reading frame")
#                             break
#                         results = model.predict(
#                             source=frame,
#                             conf=0.25,
#                             half=True,
#                             imgsz=640,
#                             verbose=False
#                         )
#                         pothole_detected = False
#                         for result in results:
#                             if result.boxes and detection_cooldown <= 0:
#                                 pothole_detected = True
#                                 detection_cooldown = 15
#                                 break
#                         if detection_cooldown > 0:
#                             detection_cooldown -= 1
#                         if pothole_detected:
#                             if add_location(current_location[0], current_location[1], "pothole"):
#                                 status_text.warning(f"Pothole detected! Location: {current_location[0]:.6f}, {current_location[1]:.6f}")
#                                 detection_frame = results[0].plot().copy()
#                                 detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
#                                 st.session_state.detection_frames.append(detection_frame_rgb)
#                                 if frame_count % 10 == 0:
#                                     with map_placeholder:
#                                         plot_pothole_map(st.session_state.locations)
#                         annotated_frame = results[0].plot()
#                         annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                         video_placeholder.image(
#                             annotated_frame_rgb,
#                             caption="Live Dashcam Feed",
#                             width=640  # Fixed width
#                         )
#                         time.sleep(0.05)
#                     status_text.info(f"Processed {FRAME_LIMIT} frames. Click 'Continue' to keep recording.")
#                     if st.button("Continue Recording"):
#                         st.rerun()
#                 cap.release()
#         except Exception as e:
#             st.error(f"Camera error: {str(e)}")
#             st.session_state.running = False
#         finally:
#             try:
#                 cap.release()
#             except:
#                 pass
#     else:
#         st.write("Press 'Start' to begin real-time detection.")
    
#     if st.session_state.detection_frames:
#         st.subheader("Pothole Detections")
#         recent_frames = st.session_state.detection_frames[-9:]
#         cols = st.columns(min(3, len(recent_frames)))
#         for i, frame in enumerate(recent_frames):
#             cols[i % len(cols)].image(
#                 frame,
#                 caption=f"Detection {i+1}",
#                 use_column_width=True
#             )

# # View map mode
# elif detection_mode == "View Map":
#     st.subheader("🗺️ View Location Map")
#     st.write(f"Showing {len(st.session_state.locations)} recorded locations on OpenStreetMap.")
#     map_placeholder = st.empty()
#     if st.button("Refresh Map with Current Location"):
#         current_location = get_current_location(timeout=15000)
#         st.write(f"Current location: {current_location[0]:.6f}, {current_location[1]:.6f}")
#     with map_placeholder:
#         plot_pothole_map(st.session_state.locations)

# # Manage data mode
# elif detection_mode == "Manage Data":
#     st.subheader("⚙️ Manage Location Data")
#     location_status = st.empty()
#     map_placeholder = st.empty()
#     default_location = get_current_location(timeout=15000)

#     st.write("Add a new location manually:")
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col1:
#         manual_lat = st.number_input("Latitude", value=default_location[0], format="%.6f")
#     with col2:
#         manual_lon = st.number_input("Longitude", value=default_location[1], format="%.6f")
#     with col3:
#         marker_type = st.selectbox(
#             "Location Type",
#             ["pothole", "flooded_road", "fallen_tree", "construction", "landslide"]
#         )
    
#     if st.button("Add Manual Location"):
#         if add_location(manual_lat, manual_lon, marker_type):
#             location_status.success(f"Added new {marker_type.replace('_', ' ').title()} at {manual_lat}, {manual_lon}")
#             with map_placeholder:
#                 plot_pothole_map(st.session_state.locations)
    
#     if st.button("Clear All Data"):
#         confirm = st.checkbox("I confirm I want to delete all location data")
#         if confirm:
#             st.session_state.locations = []
#             save_pothole_locations([])
#             location_status.success("All location data has been cleared.")
#             with map_placeholder:
#                 plot_pothole_map(st.session_state.locations)
    
#     st.write("Current locations:")
#     if st.session_state.locations:
#         location_df = {
#             "Latitude": [loc[0] for loc in st.session_state.locations],
#             "Longitude": [loc[1] for loc in st.session_state.locations],
#             "Type": [loc[2].replace('_', ' ').title() if len(loc) > 2 else "Pothole" for loc in st.session_state.locations]
#         }
#         st.dataframe(location_df)
#         with map_placeholder:
#             plot_pothole_map(st.session_state.locations)
#     else:
#         st.write("No locations recorded yet.")
    
#     if st.button("Export Data as JSON") and st.session_state.locations:
#         st.download_button(
#             label="Download JSON",
#             data=json.dumps(st.session_state.locations),
#             file_name="locations.json",
#             mime="application/json"
#         )




# import os
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import streamlit as st
# import cv2
# import numpy as np
# import time
# import folium
# from streamlit_folium import folium_static
# import json
# import base64
# import tempfile
# import shutil
# from streamlit_js_eval import streamlit_js_eval
# import requests
# import logging

# # Set page config as the first Streamlit command
# st.set_page_config(page_title="Pothole Detection Dashboard", page_icon="🛣", layout="wide")

# # Configure logging
# logging.basicConfig(
#     filename='pothole_detection.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Optimize CPU usage for Ryzen 4000 series
# torch.set_num_threads(6)
# st.write(f"Using {torch.get_num_threads()} CPU threads for processing")

# # Set working directory
# HOME = r"C:\Users\chris\pothole\trash"
# os.chdir(HOME)

# # Paths
# model_path = r"C:\Users\chris\pothole\best.pt"
# openvino_model_path = r"C:\Users\chris\pothole\best_openvino_model"
# pothole_data_path = os.path.join(HOME, "pothole_locations.json")

# # Load the OpenVINO-optimized model
# st.write(f"Loading OpenVINO model from: {openvino_model_path}")
# model = YOLO(openvino_model_path)

# # API endpoint
# API_ENDPOINT = "https://location-api-production-e673.up.railway.app/api/location"

# # Valid hazard types for the API
# VALID_HAZARDS = {
#     "pothole": "Pothole",
#     "flooded_roads": "Flooded Roads",
#     "fallen_trees": "Fallen Trees",
#     "construction_works": "Construction Works",
#     "landslide": "Landslide"
# }

# # Function to send location to API with updated payload
# def send_location_to_api(lat, lon, marker_type="pothole"):
#     hazard_type = VALID_HAZARDS.get(marker_type.lower(), "Pothole")
#     payload = {
#         "latitude": lat,
#         "longitude": lon,
#         "hazard_type": hazard_type
#     }
#     st.write(f"Sending payload to API: {payload}")
#     try:
#         response = requests.post(API_ENDPOINT, json=payload)
#         st.write(f"API Response Status: {response.status_code}")
#         st.write(f"API Response Text: {response.text}")
#         if response.status_code in (200, 201):
#             logger.info(f"Successfully sent {hazard_type} location to API: {lat}, {lon} - Response: {response.text}")
#             return True
#         else:
#             logger.error(f"Failed to send {hazard_type} location to API: {lat}, {lon} - Status: {response.status_code} - Response: {response.text}")
#             return False
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error sending {hazard_type} location to API: {lat}, {lon} - {str(e)}")
#         st.error(f"API Request Failed: {str(e)}")
#         return False

# # Function to save pothole locations to file
# def save_pothole_locations(locations):
#     with open(pothole_data_path, 'w') as f:
#         json.dump(locations, f)
#     st.write(f"Saved {len(locations)} pothole locations to {pothole_data_path}")
#     logger.info(f"Saved {len(locations)} pothole locations to {pothole_data_path}")

# def get_exam_location():
#     lat = float(f"{9}.{5862091}")
#     lon = float(f"{76}.{9829277}")
#     logger.info("Using predefined exam location for testing.")
#     return [lat, lon]

# # Function to load pothole locations from file
# def load_pothole_locations():
#     if os.path.exists(pothole_data_path):
#         try:
#             with open(pothole_data_path, 'r') as f:
#                 locations = json.load(f)
#             st.write(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             logger.info(f"Loaded {len(locations)} pothole locations from {pothole_data_path}")
#             return locations
#         except Exception as e:
#             st.error(f"Error loading pothole locations: {e}")
#             logger.error(f"Error loading pothole locations: {e}")
#             return []
#     else:
#         st.write("No saved pothole locations found. Starting with empty map.")
#         logger.info("No saved pothole locations found")
#         return []

# # Initialize session state for storing locations
# if "locations" not in st.session_state:
#     st.session_state.locations = load_pothole_locations()

# # Function to plot map with different marker types
# def plot_pothole_map(locations):
#     if not locations:
#         st.write("No locations recorded yet.")
#         return
    
#     avg_lat = sum(loc[0] for loc in locations) / len(locations)
#     avg_lon = sum(loc[1] for loc in locations) / len(locations)
#     center = [avg_lat, avg_lon]
    
#     m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    
#     marker_styles = {
#         "pothole": {"color": "red", "icon": "warning-sign"},
#         "flooded_roads": {"color": "blue", "icon": "tint"},
#         "fallen_trees": {"color": "green", "icon": "tree-deciduous"},
#         "construction_works": {"color": "orange", "icon": "wrench"},
#         "landslide": {"color": "purple", "icon": "collapse-down"}
#     }
    
#     for loc in locations:
#         if len(loc) == 2:
#             lat, lon = loc
#             marker_type = "pothole"
#         elif len(loc) == 3:
#             lat, lon, marker_type = loc
#         else:
#             continue
#         style = marker_styles.get(marker_type.lower(), marker_styles["pothole"])
#         folium.Marker(
#             location=[lat, lon],
#             popup=f"{VALID_HAZARDS.get(marker_type.lower(), 'Pothole')} Detected",
#             icon=folium.Icon(color=style["color"], icon=style["icon"])
#         ).add_to(m)
    
#     folium_static(m)

# # Function to add a new location
# def add_location(lat, lon, marker_type="pothole"):
#     is_duplicate = False
#     for existing_lat, existing_lon, *_ in st.session_state.locations:
#         if abs(existing_lat - lat) < 0.0001 and abs(existing_lon - lon) < 0.0001:
#             is_duplicate = True
#             st.write(f"This {VALID_HAZARDS.get(marker_type.lower(), 'Pothole')} location is already recorded locally.")
#             break
    
#     if not is_duplicate:
#         st.session_state.locations.append([lat, lon, marker_type.lower()])
#         save_pothole_locations(st.session_state.locations)
    
#     # Always send to API, even if it's a duplicate locally
#     if send_location_to_api(lat, lon, marker_type):
#         st.success(f"{VALID_HAZARDS.get(marker_type.lower(), 'Pothole')} location sent to API successfully!")
#     else:
#         st.warning(f"Failed to send {VALID_HAZARDS.get(marker_type.lower(), 'Pothole')} location to API, but saved locally if not duplicate.")
#     return True

# # Function to create download link for video
# def get_video_download_link(video_path, link_text="Download processed video"):
#     with open(video_path, "rb") as file:
#         video_bytes = file.read()
#     b64 = base64.b64encode(video_bytes).decode()
#     filename = os.path.basename(video_path)
#     dl_link = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
#     return dl_link

# # Function to convert to MP4
# def convert_to_mp4(input_path):
#     output_path = os.path.splitext(input_path)[0] + ".mp4"
#     try:
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             st.error(f"Could not open video file: {input_path}")
#             return None
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             out.write(frame)
#         cap.release()
#         out.release()
#         if os.path.exists(output_path):
#             st.write(f"Successfully converted video to MP4: {output_path}")
#             return output_path
#         else:
#             st.error("Failed to create MP4 file")
#             return None
#     except Exception as e:
#         st.error(f"Error converting video: {e}")
#         try:
#             import subprocess
#             st.write("Attempting conversion with FFmpeg...")
#             ffmpeg_cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -preset fast -crf 22 "{output_path}"'
#             result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
#             if os.path.exists(output_path):
#                 st.write("FFmpeg conversion successful")
#                 return output_path
#             else:
#                 st.error(f"FFmpeg conversion failed: {result.stderr}")
#                 return None
#         except Exception as ffmpeg_error:
#             st.error(f"FFmpeg fallback failed: {ffmpeg_error}")
#             return None

# # Extract frames as fallback method
# def extract_video_frames(video_path, max_frames=20):
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return []
#         frames = []
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         step = max(1, total_frames // max_frames)
#         for i in range(0, total_frames, step):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if ret:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
#             if len(frames) >= max_frames:
#                 break
#         cap.release()
#         return frames
#     except Exception as e:
#         st.error(f"Error extracting frames: {e}")
#         return []

# # Streamlit UI with Beautification
# st.title("🛣 Pothole Detection Dashboard")
# st.markdown("""
#     <style>
#     .main {background-color: #f0f2f6;}
#     .stButton>button {background-color: #4CAF50; color: white;}
#     .stRadio>label {font-size: 18px;}
#     </style>
# """, unsafe_allow_html=True)

# st.write("Choose an option below to get started:")
# detection_mode = st.radio(
#     "Select Detection Mode",
#     ("Upload Video", "Real-Time Dashcam", "View Map", "Manage Data", "Current Location"),
#     horizontal=True
# )

# # Video upload mode
# if detection_mode == "Upload Video":
#     st.subheader("📹 Upload Video for Pothole Detection")
#     uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         input_path = os.path.join(HOME, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.write(f"Uploaded video saved temporarily at: {input_path}")

#         st.write(f"Processing video: {input_path}")
#         with st.spinner("Processing..."):
#             results = model.predict(
#                 source=input_path,
#                 conf=0.25,
#                 save=True,
#                 half=True,
#                 imgsz=640
#             )

#         run_dir = max([d for d in os.listdir("runs/detect") if d.startswith("predict")],
#                       key=lambda x: os.path.getctime(os.path.join("runs/detect", x)),
#                       default="predict")
#         output_path = os.path.join(HOME, "runs", "detect", run_dir, Path(input_path).stem + ".avi")

#         map_placeholder = st.empty()

#         if os.path.exists(output_path):
#             st.write(f"Processed video saved at: {output_path}")
#             st.write(f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
#             st.subheader("Processed Video with Potholes")
#             st.markdown(
#                 get_video_download_link(output_path, "⬇ Download processed video"),
#                 unsafe_allow_html=True
#             )

#             lat, lon = get_exam_location()
#             if add_location(lat, lon, "pothole"):
#                 st.write(f"Pothole recorded at Latitude {lat}, Longitude {lon}")
#             with map_placeholder:
#                 plot_pothole_map(st.session_state.locations)
#         else:
#             st.error(f"Error: Output file not found at {output_path}")
#             detect_dirs = [d for d in os.listdir("runs/detect") if d.startswith("predict")]
#             if detect_dirs:
#                 latest_dir = max(detect_dirs, key=lambda x: os.path.getctime(os.path.join("runs/detect", x)))
#                 st.write(f"Latest output directory: runs/detect/{latest_dir}")
#                 files_in_dir = os.listdir(os.path.join("runs/detect", latest_dir))
#                 st.write(f"Files in directory: {files_in_dir}")

# # Real-time dashcam mode
# elif detection_mode == "Real-Time Dashcam":
#     st.subheader("📷 Real-Time Pothole Detection with Dashcam")
#     st.write("Using your camera for live pothole detection.")

#     if "running" not in st.session_state:
#         st.session_state.running = False
#     if "detection_frames" not in st.session_state:
#         st.session_state.detection_frames = []

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Start"):
#             st.session_state.running = True
#             st.rerun()
#     with col2:
#         if st.button("Stop"):
#             st.session_state.running = False
#             st.rerun()

#     video_placeholder = st.empty()
#     status_text = st.empty()
#     map_placeholder = st.empty()

#     if st.session_state.running:
#         status_text.info("Starting camera... Please wait.")
#         try:
#             cap = cv2.VideoCapture(0)
#             if not cap.isOpened():
#                 st.error("Error: Could not open camera. Ensure a camera is connected.")
#                 st.session_state.running = False
#             else:
#                 status_text.success("Camera connected! Displaying live feed...")
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("Error: Could not read frame from camera.")
#                     cap.release()
#                     st.session_state.running = False
#                 else:
#                     first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     video_placeholder.image(first_frame_rgb, caption="Live Dashcam Feed", width=640)
#                     current_location = get_exam_location()
#                     detection_cooldown = 0
#                     FRAME_LIMIT = 100
#                     for frame_count in range(FRAME_LIMIT):
#                         if not st.session_state.running:
#                             break
#                         ret, frame = cap.read()
#                         if not ret:
#                             st.error("Error reading frame")
#                             break
#                         results = model.predict(
#                             source=frame,
#                             conf=0.25,
#                             half=True,
#                             imgsz=640,
#                             verbose=False
#                         )
#                         pothole_detected = False
#                         for result in results:
#                             if result.boxes and detection_cooldown <= 0:
#                                 pothole_detected = True
#                                 detection_cooldown = 15
#                                 break
#                         if detection_cooldown > 0:
#                             detection_cooldown -= 1
#                         if pothole_detected:
#                             if add_location(current_location[0], current_location[1], "pothole"):
#                                 status_text.warning(f"Pothole detected! Location: {current_location[0]:.6f}, {current_location[1]:.6f}")
#                                 detection_frame = results[0].plot().copy()
#                                 detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
#                                 st.session_state.detection_frames.append(detection_frame_rgb)
#                                 if frame_count % 10 == 0:
#                                     with map_placeholder:
#                                         plot_pothole_map(st.session_state.locations)
#                         annotated_frame = results[0].plot()
#                         annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                         video_placeholder.image(
#                             annotated_frame_rgb,
#                             caption="Live Dashcam Feed",
#                             width=640
#                         )
#                         time.sleep(0.05)
#                     status_text.info(f"Processed {FRAME_LIMIT} frames. Click 'Continue' to keep recording.")
#                     if st.button("Continue Recording"):
#                         st.rerun()
#                 cap.release()
#         except Exception as e:
#             st.error(f"Camera error: {str(e)}")
#             st.session_state.running = False
#         finally:
#             try:
#                 cap.release()
#             except:
#                 pass
#     else:
#         st.write("Press 'Start' to begin real-time detection.")
    
#     if st.session_state.detection_frames:
#         st.subheader("Pothole Detections")
#         recent_frames = st.session_state.detection_frames[-9:]
#         cols = st.columns(min(3, len(recent_frames)))
#         for i, frame in enumerate(recent_frames):
#             cols[i % len(cols)].image(
#                 frame,
#                 caption=f"Detection {i+1}",
#                 use_column_width=True
#             )

# # View map mode
# elif detection_mode == "View Map":
#     st.subheader("🗺 View Location Map")
#     st.write(f"Showing {len(st.session_state.locations)} recorded locations on OpenStreetMap.")
#     map_placeholder = st.empty()
#     if st.button("Refresh Map with Current Location"):
#         current_location = get_exam_location()
#         st.write(f"Current location: {current_location[0]:.6f}, {current_location[1]:.6f}")
#     with map_placeholder:
#         plot_pothole_map(st.session_state.locations)

# # Manage data mode
# elif detection_mode == "Manage Data":
#     st.subheader("⚙ Manage Location Data")
#     location_status = st.empty()
#     map_placeholder = st.empty()
#     default_location = get_exam_location()

#     st.write("Add a new location manually:")
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col1:
#         manual_lat = st.number_input("Latitude", value=default_location[0], format="%.6f")
#     with col2:
#         manual_lon = st.number_input("Longitude", value=default_location[1], format="%.6f")
#     with col3:
#         marker_type = st.selectbox(
#             "Location Type",
#             list(VALID_HAZARDS.values())
#         )
    
#     if st.button("Add Manual Location"):
#         internal_marker_type = [k for k, v in VALID_HAZARDS.items() if v == marker_type][0]
#         if add_location(manual_lat, manual_lon, internal_marker_type):
#             location_status.success(f"Added new {marker_type} at {manual_lat}, {manual_lon}")
#             with map_placeholder:
#                 plot_pothole_map(st.session_state.locations)
    
#     if st.button("Clear All Data"):
#         confirm = st.checkbox("I confirm I want to delete all location data")
#         if confirm:
#             st.session_state.locations = []
#             save_pothole_locations([])
#             location_status.success("All location data has been cleared.")
#             with map_placeholder:
#                 plot_pothole_map(st.session_state.locations)
    
#     st.write("Current locations:")
#     if st.session_state.locations:
#         location_df = {
#             "Latitude": [loc[0] for loc in st.session_state.locations],
#             "Longitude": [loc[1] for loc in st.session_state.locations],
#             "Type": [VALID_HAZARDS.get(loc[2].lower(), "Pothole") for loc in st.session_state.locations]
#         }
#         st.dataframe(location_df)
#         with map_placeholder:
#             plot_pothole_map(st.session_state.locations)
#     else:
#         st.write("No locations recorded yet.")
    
#     if st.button("Export Data as JSON") and st.session_state.locations:
#         st.download_button(
#             label="Download JSON",
#             data=json.dumps(st.session_state.locations),
#             file_name="locations.json",
#             mime="application/json"
#         )

# # Current Location mode
# elif detection_mode == "Current Location":
#     st.subheader("📍 Your Current Location")
#     st.write("Click the button below to fetch your current coordinates.")
    
#     if st.button("Get My Location"):
#         with st.spinner("Fetching location..."):
#             lat, lon = get_exam_location()
#             st.write(f"*Latitude:* {lat:.6f}")
#             st.write(f"*Longitude:* {lon:.6f}")
            
#             m = folium.Map(location=[lat, lon], zoom_start=16, tiles="OpenStreetMap")
#             folium.Marker(
#                 location=[lat, lon],
#                 popup="You are here!",
#                 icon=folium.Icon(color="blue", icon="user")
#             ).add_to(m)
#             folium_static(m, width=700, height=400)





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