import os
import argparse
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from moviepy.editor import VideoFileClip
from mtcnn import MTCNN
import cv2
import numpy as np
from tqdm import tqdm
import absl
from tensorflow.keras import mixed_precision
import mediapipe as mp

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def pixelate_faces(input_file, output_file, method, pixelation_level):
    # Load the video and audio
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio

    # Initialize the face detector based on the chosen method
    if method == 'mtcnn':
        detector = MTCNN()
    elif method == 'mediapipe':
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    else:
        raise ValueError("Unsupported face detection method. Choose 'mtcnn' or 'mediapipe'.")

    # Prepare to save the video output
    temp_video_path = 'temp_video_no_audio.mp4'
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_clip.fps, (video_clip.w, video_clip.h))

    # Get total number of frames for the progress bar
    total_frames = int(video_clip.fps * video_clip.duration)

    # Process each frame with a progress bar
    for frame in tqdm(video_clip.iter_frames(fps=video_clip.fps), total=total_frames, desc="Processing frames", unit="frame"):
        # Convert the frame to a mutable (writable) format
        frame = np.array(frame)

        if method == 'mtcnn':
            # Detect faces using MTCNN
            faces = detector.detect_faces(frame)
            for face in faces:
                x, y, w, h = face['box']
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                face_region = frame[y:y + h, x:x + w]
                face_region_small = cv2.resize(face_region, (w // pixelation_level, h // pixelation_level), interpolation=cv2.INTER_LINEAR)
                face_region_pixelated = cv2.resize(face_region_small, (w, h), interpolation=cv2.INTER_NEAREST)
                frame[y:y + h, x:x + w] = face_region_pixelated
        elif method == 'mediapipe':
            # Detect faces using MediaPipe
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                        continue
                    face_region = frame[y:y + h, x:x + w]
                    face_region_small = cv2.resize(face_region, (w // 10, h // 10), interpolation=cv2.INTER_LINEAR)
                    face_region_pixelated = cv2.resize(face_region_small, (w, h), interpolation=cv2.INTER_NEAREST)
                    frame[y:y + h, x:x + w] = face_region_pixelated

        # Write the frame to the output video
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release the video object
    out.release()

    # Combine the original audio with the pixelated video using ffmpeg
    final_output_path = output_file
    escaped_input_file = f'"{input_file}"'
    escaped_temp_video = f'"{temp_video_path}"'
    escaped_final_output = f'"{final_output_path}"'

    os.system(f"ffmpeg -y -i {escaped_temp_video} -i {escaped_input_file} -c:v copy -c:a copy {escaped_final_output}")

    # Clean up the temporary video file
    os.remove(temp_video_path)

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Pixelate faces in a video and preserve the original audio.")
    parser.add_argument("input_file", help="Path to the input video file.")
    parser.add_argument("output_file", help="Path to the output video file.")
    parser.add_argument("--method", choices=['mtcnn', 'mediapipe'], default='mtcnn', help="Face detection method to use (default: mtcnn)")
    parser.add_argument("--pixelation_level", type=int, default=10, help="Pixelation level (default: 10)")

    args = parser.parse_args()

    pixelate_faces(args.input_file, args.output_file, args.method, args.pixelation_level)
    pixelate_faces(args.input_file, args.output_file, args.method)

def check_gpu():
    # Check if GPU is available and configure TensorFlow to use it
    if False:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Using GPU for TensorFlow.")
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU detected, using CPU.")