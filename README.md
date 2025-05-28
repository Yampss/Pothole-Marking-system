# Real-Time Pothole Detection and Mapping System
## Overview
This project is a real-time pothole detection and mapping system designed to identify potholes in video footage or live dashcam feeds, map their locations on OpenStreetMap, and log them for infrastructure monitoring. Leveraging AI-driven object detection with YOLOv8 and OpenVINO optimization, the system provides an interactive Streamlit dashboard for video uploads, live detection, and manual location management.
Features

###### Real-Time Detection: Detects potholes in live dashcam feeds with a confidence threshold of 0.25.
###### Video Processing: Processes uploaded videos to identify and annotate potholes, with downloadable output.
###### Interactive Mapping: Visualizes detected pothole locations on OpenStreetMap using Folium.
###### Manual Location Management: Allows users to add, view, and export pothole locations via a user-friendly interface.
###### API Integration: Designed to send location data to a remote API for scalability (currently in development).
###### Optimized Performance: Uses OpenVINO for efficient inference on CPU and supports multi-threading for Ryzen 4000 series.

## Technologies Used

Python: Core programming language.
Streamlit: For the interactive web dashboard.
YOLOv8 (Ultralytics): For pothole detection.
OpenVINO: For model optimization.
OpenCV: For video and image processing.
Folium: For mapping on OpenStreetMap.
PyTorch (Torch): For model operations.
Requests: For API integration.
Logging: For debugging and performance tracking.

## Installation
Prerequisites

Python 3.8 or higher
A 16 or 32 GB RAM.
pip install the necessary dependencies. 
A webcam (for real-time detection)

## Steps

Clone the repository:
```bash
git clone https://github.com/Yampss/pothole-marking-system.git
cd pothole-detection-system
```

Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

#### Download the pre-trained YOLO model (best.pt) and OpenVINO-optimized model (best_openvino_model) into the project directory. (These are not included due to size constraints; replace with your trained models.)

Usage
```bash
streamlit run main.py
```

Open the provided URL (typically http://localhost:8501) in your browser.
Choose a detection mode:
Upload Video: Upload a video file to detect potholes and download the processed output.
Real-Time Dashcam: Use a webcam for live pothole detection.
View Map: See detected pothole locations on a map.
Manage Data: Manually add or export pothole locations.
Current Location: View a sample location on the map.



#### Project Structure

main.py: Main application script.
pothole_detection.log: Log file for debugging.
runs/detect/: Directory for processed video outputs.
pothole_locations.json: Stores detected pothole coordinates.

#### Future Improvements

Resolve API connectivity for real-time data syncing.
Add support for additional hazard types beyond potholes.
Implement GPS integration for accurate location tracking.
Enhance UI with more visualization options.

#### Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with your changes.
License
This project is licensed under the MIT License.
Contact
For questions or collaboration, reach out to [chrissattasseril16@gmail.com] or open an issue on GitHub.
