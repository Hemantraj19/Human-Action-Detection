from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
import cv2
import numpy as np
import tensorflow as tf
from moviepy.editor import *
from collections import deque
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

# Define the constants, models, and predict_on_video function here
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["Diving", "Punch", "Typing", "PushUps"]
SEQUENCE_LENGTH = 20
LRCN_MODEL_PATH = 'LRCN_model___Date_Time_2023_05_04__06_55_19___Loss_0.14662007987499237___Accuracy_0.970802903175354.h5'
CONVLSTM_MODEL_PATH = 'convlstm_model___Date_Time_2023_05_04__06_53_53___Loss_0.3467928469181061___Accuracy_0.9197080135345459.h5'

# Load the pre-trained models.
LRCN_model = tf.keras.models.load_model(LRCN_MODEL_PATH)
convlstm_model = tf.keras.models.load_model(CONVLSTM_MODEL_PATH)

def frame_generator(video_reader):
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        yield frame

def predict_on_video(video_file_path, output_file_path):
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('H', '2', '6', '4'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a deque to store video frames.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Initialize a ThreadPoolExecutor to parallelize the prediction process.
    with ThreadPoolExecutor(max_workers=5) as executor:

        predicted_class_name = ''

        # Use a generator to read video frames.
        for frame in frame_generator(video_reader):

            # Resize the Frame to fixed Dimensions.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
            normalized_frame = resized_frame / 255

            # Appending the pre-processed frame into the frames queue.
            frames_queue.append(normalized_frame)

            # Check if the number of frames in the queue are equal to the fixed sequence length.
            if len(frames_queue) == SEQUENCE_LENGTH:

                # Pass the normalized frames to the model and get the predicted probabilities.
                future = executor.submit(lambda: (LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0],
                                                  convlstm_model.predict(np.expand_dims(frames_queue, axis=0))[0]))
                predicted_labels_probabilities_LRCN, predicted_labels_probabilities_convlstm = future.result()

                # Assign weights to each model based on their performance.
                LRCN_weight = 0.6
                convlstm_weight = 0.4

                # Calculate the weighted average of the predicted probabilities.
                predicted_labels_probabilities = (predicted_labels_probabilities_LRCN * LRCN_weight) + (predicted_labels_probabilities_convlstm * convlstm_weight)

                # Get the index of the class with the highest probability.
                predicted_label = np.argmax(predicted_labels_probabilities)

                # Get the class name using the retrieved index.
                predicted_class_name = CLASSES_LIST[predicted_label]

                # Remove the oldest frame from the queue.
                frames_queue.popleft()

            # Write predicted class name on top of the frame.
            cv2.putText(frame, predicted_class_name, (int(frame.shape[1]/2)-100, 53),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Write The frame into the disk using the VideoWriter Object.
            video_writer.write(frame)

    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()

# Define the Flask app and its configurations
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], str(
                uuid.uuid4()) + '.' + filename.rsplit('.', 1)[1].lower())
            file.save(input_video_path)
            output_video_path = os.path.join(
                app.config['OUTPUT_FOLDER'], str(uuid.uuid4()) + '.mp4')
            predict_on_video(input_video_path, output_video_path)
            os.remove(input_video_path)
            ovp = output_video_path.split(os.path.sep)
            return render_template('result.html', output_video=ovp[1])
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])
    app.run(debug=True)
