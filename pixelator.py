import os
import argparse,logging
# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress everything except errors

import tensorflow as tf
from moviepy.editor import VideoFileClip
from mtcnn import MTCNN
import cv2
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
import absl
from tensorflow.keras import mixed_precision

tf.get_logger().setLevel('ERROR')
# Suppress Abseil logging (used internally by TensorFlow)
absl.logging.set_verbosity(absl.logging.ERROR)

# Suppress Python's logging from TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Disable XLA
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Enable mixed precision for better GPU performance on Apple Silicon
# mixed_precision.set_global_policy('mixed_float16')


def pixelate_faces(input_file, output_file):
    # Load the video and audio
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio

    # Initialize MTCNN face detector (TensorFlow will automatically use GPU if available)
    detector = MTCNN()

    # Prepare to save the video output
    temp_video_path = 'temp_video_no_audio.mp4'  # Temporary video file without audio
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_clip.fps, (video_clip.w, video_clip.h))

    # Get total number of frames for the progress bar
    total_frames = int(video_clip.fps * video_clip.duration)

    # Process each frame with a progress bar
    for frame in tqdm(video_clip.iter_frames(fps=video_clip.fps), total=total_frames, desc="Processing frames", unit="frame"):
        # Convert the frame to a mutable (writable) format
        frame = np.array(frame)

        # Detect faces using MTCNN (this will use the GPU if available)
        faces = detector.detect_faces(frame)

        # Pixelate each detected face
        for face in faces:
            x, y, w, h = face['box']

            # Ensure face dimensions are within frame limits
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue

            # Extract and pixelate the face region
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

    # Escape file paths with quotes
    final_output_path = output_file
    escaped_input_file = f'"{input_file}"'
    escaped_temp_video = f'"{temp_video_path}"'
    escaped_final_output = f'"{final_output_path}"'

    # Combine the original audio with the pixelated video using ffmpeg
    os.system(f"ffmpeg -y -i {escaped_temp_video} -i {escaped_input_file} -c:v copy -c:a copy {escaped_final_output}")

    # Clean up the temporary video file
    os.remove(temp_video_path)

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Pixelate faces in a video and preserve the original audio.")
    parser.add_argument("input_file", help="Path to the input video file.")
    parser.add_argument("output_file", help="Path to the output video file.")

    args = parser.parse_args()

    # Run the pixelation function with command line arguments
    pixelate_faces(args.input_file, args.output_file)



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
