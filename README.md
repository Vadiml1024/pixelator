# Pixelator

The `pixelator.py` script processes a video to pixelate faces while keeping the original audio intact.

This project focuses on video processing and analysis using machine learning and computer vision techniques. It leverages TensorFlow for deep learning, moviepy for video editing, opencv-python for computer vision tasks, and mtcnn for face detection.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Vadiml1024/pixelator.git
    cd pixelator
    ```

2. Create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage:

python pixelator.py [-h] input_file output_file

Pixelate faces in a video and preserve the original audio.

positional arguments:
  input_file   Path to the input video file.
  output_file  Path to the output video file.

options:
  -h, --help   show this help message and exit

## License

MIT
