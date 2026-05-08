# ===============================================================
# PROJECT: Automated Road Condition Monitoring
# ===============================================================



import os
import torch
from ultralytics import YOLO
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import json
import base64
import tempfile
import shutil
import logging
from PIL import Image
import subprocess



st.set_page_config(
    page_title="Pothole Detection Dashboard",
    page_icon="🛣️",
    layout="wide"
)


logging.basicConfig(
    filename="pothole_detection_mid.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)




try:
    cpu_cores = os.cpu_count()
    threads_to_use = max(1, cpu_cores - 2 if cpu_cores else 4)
    torch.set_num_threads(threads_to_use)
    st.write(f"Using {torch.get_num_threads()} CPU threads for processing")
    logger.info(f"Using {torch.get_num_threads()} CPU threads for PyTorch")

except Exception as e:
    st.warning(f"Could not set PyTorch threads: {e}")
    logger.warning(f"Could not set PyTorch threads: {e}")



SCRIPT_DIR = Path(__file__).resolve().parent
HOME = SCRIPT_DIR / "trash"

try:
    os.makedirs(HOME, exist_ok=True)
    st.write(f"Using directory for temporary outputs: {HOME}")
    logger.info(f"Output/temp directory set to: {HOME}")
except Exception as e:
    st.error(f"Error creating output directory {HOME}: {e}")
    st.stop()


openvino_model_path = SCRIPT_DIR / "best_openvino_model"

st.write(f"Attempting to load OpenVINO model from: {openvino_model_path}")



try:
    xml_path = openvino_model_path / "best.xml"
    bin_path = openvino_model_path / "best.bin"

    if not xml_path.exists() or not bin_path.exists():
        st.error("Error: Missing OpenVINO model files")
        logger.error("Model files missing in best_openvino_model/")
        st.stop()

    model = YOLO(openvino_model_path)
    st.success("OpenVINO model loaded successfully!")
    logger.info("Model loaded successfully")

except Exception as e:
    st.error(f"Error loading OpenVINO model: {e}")
    st.stop()




def get_video_download_link(video_path, link_text="Download processed video"):
    try:
        with open(video_path, "rb") as file:
            video_bytes = file.read()

        b64 = base64.b64encode(video_bytes).decode()
        filename = Path(video_path).name
        mime_type = "video/avi" if Path(video_path).suffix.lower() == ".avi" else "video/mp4"

        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'

    except Exception as e:
        st.error(f"Error creating download link: {e}")
        return "Link error."




def convert_to_mp4(input_path_str):
    input_path = Path(input_path_str)
    output_path = input_path.with_suffix(".mp4")

    if not input_path.exists():
        st.error(f"Input not found: {input_path}")
        return None

    ffmpeg_exe = shutil.which("ffmpeg")

    if ffmpeg_exe:
        try:
            st.write("Attempting FFmpeg conversion...")
            cmd = [
                ffmpeg_exe, "-y", "-i", str(input_path),
                "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-c:a", "aac", "-b:a", "128k",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and output_path.stat().st_size > 0:
                return str(output_path)

        except Exception as e_ff:
            st.error(f"FFmpeg error: {e_ff}")

    # ------------------ OpenCV fallback (no audio) ------------------
    try:
        cap = cv2.VideoCapture(str(input_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        out = cv2.VideoWriter(str(output_path),
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        if output_path.stat().st_size > 0:
            return str(output_path)

    except Exception as e_cv:
        st.error(f"OpenCV error: {e_cv}")

    return None




def extract_video_frames(video_path, max_frames=20):
    frames = []

    try:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            return frames

        step = max(1, total // max_frames)

        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, f = cap.read()

            if ret:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    except Exception as e:
        st.error(f"Frame extraction error: {e}")

    finally:
        if cap.isOpened():
            cap.release()

    return frames



st.title("🛣️ Road Condition Monitoring")

st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {
        background-color: #0068c9;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0050a0;
    }
    .stFileUploader label {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)




st.write("Choose an option below to demonstrate detection:")
detection_mode = st.radio(
    "Select Mode",
    ("Upload Image", "Upload Video"),
    horizontal=True,
    label_visibility="collapsed",
)



if detection_mode == "Upload Image":

    st.subheader("🖼️ Upload Image for Analysis")
    uploaded_image_file = st.file_uploader(
        "Choose an image file (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image_file:
        try:
            image_pil = Image.open(uploaded_image_file)
            img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            with st.spinner("Analyzing image..."):
                results = model.predict(source=img_cv, conf=0.25, save=False)

                if results and results[0].boxes:
                    annotated_image = results[0].plot()
                else:
                    annotated_image = img_cv
                    st.info("No potholes detected.")

            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                     caption="Processed Image",
                     use_column_width=True)

        except Exception as e:
            st.error(f"Image processing error: {e}")




elif detection_mode == "Upload Video":

    st.subheader("📹 Upload Video for Analysis")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:

        temp_upload_dir = HOME / "uploads"
        os.makedirs(temp_upload_dir, exist_ok=True)

        input_path = temp_upload_dir / uploaded_file.name

        try:
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write(f"Video saved at: {input_path}")

        except Exception as e:
            st.error(f"Save failed: {e}")
            st.stop()

        output_path = None
        output_dir = HOME / "runs" / "detect"

        with st.spinner("Analyzing video..."):
            try:
                results = model.predict(
                    source=str(input_path),
                    conf=0.25,
                    save=True,
                    project=str(output_dir),
                    name="predict_upload",
                    exist_ok=True,
                )

                all_runs = sorted(
                    [d for d in output_dir.iterdir() if d.is_dir()],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

                latest_run = all_runs[0]
                expected_name = input_path.stem + ".avi"
                output_path = latest_run / expected_name

                if not output_path.exists():
                    avi_files = list(latest_run.glob("*.avi"))
                    if avi_files:
                        output_path = avi_files[0]

            except Exception as e:
                st.error(f"Prediction error: {e}")

        if output_path and output_path.exists():
            st.success("Processing complete!")

            st.markdown(
                get_video_download_link(str(output_path), "⬇️ Download (.avi)"),
                unsafe_allow_html=True
            )

            mp4_version = convert_to_mp4(str(output_path))

            if mp4_version:
                st.video(mp4_version)
            else:
                st.video(str(output_path))

        else:
            st.error("Output video not found.")

        try:
            input_path.unlink()
        except:
            pass
