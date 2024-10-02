import gradio as gr
import tempfile
import os
from pixelator2 import obfuscate_faces

def process_video(input_file, output_file_name, method, obfuscation, pixelation_level, min_detection_confidence, range_selection):
    # Set default output file name if not provided
    if not output_file_name:
        output_file_name = "Obfuscated.mp4"
    else:
        output_file_name = output_file_name if output_file_name.endswith(".mp4") else output_file_name + ".mp4"

    # Run the obfuscation function
    obfuscate_faces(input_file.name, output_file_name, method, obfuscation, pixelation_level, min_detection_confidence, range_selection)

    return output_file_name

def update_visibility(method):
    if method == 'mediapipe':
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)

# Define the Gradio interface
with gr.Blocks() as iface:
    input_file = gr.File(label="Input Video", type="filepath")
    output_file_name = gr.Textbox(label="Output File Name", placeholder="Obfuscated.mp4")
    method = gr.Radio(choices=['mtcnn', 'mediapipe'], label="Face Detection Method", value='mtcnn')
    obfuscation = gr.Radio(choices=['pixelate', 'blur'], label="Obfuscation Method", value='pixelate')
    pixelation_level = gr.Slider(minimum=1, maximum=50, value=10, label="Pixelation Level")
    min_detection_confidence = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, label="Minimum Detection Confidence", visible=False)
    range_selection = gr.Radio(choices=['near', 'far'], label="Detection Range", value='near', visible=False)
    output_video = gr.File(label="Output Video")

    method.change(fn=update_visibility, inputs=method, outputs=[min_detection_confidence, range_selection])

    submit_button = gr.Button("Submit")
    submit_button.click(fn=process_video, inputs=[input_file, output_file_name, method, obfuscation, pixelation_level, min_detection_confidence, range_selection], outputs=output_video)

# Launch the Gradio interface
iface.launch()