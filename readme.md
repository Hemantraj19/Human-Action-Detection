# Video Action Recognition

This repository contains scripts to train an action recognition model on videos and a Flask application to make predictions using the trained model.

You can view some output videos at - https://shorturl.at/kuvOT

Ensure all dependencies are installed before executing the scripts.

```bash
$ pip install -r requirements.txt
```

## Files Description:

1. **app.py**: A Flask application to upload videos, predict actions using pre-trained models, and display the processed video with predicted actions.
2. **train.py**: Script to train a Long-term Recurrent Convolutional Networks (LRCN) model on video data for action recognition.

## Actions Recognized:
- Diving
- Punch
- Typing
- PushUps

## Instructions:

### 1. Training the Model:
To train the model, execute the `train.py` script. This script will:
- Load video data.
- Preprocess and extract frames.
- Train the LRCN and Convlstm models.
- Save the trained models. 

```bash
$ python train.py
```

### 2. Running the Flask Application:
To run the Flask application, execute the `app.py` script. Once the application is running, navigate to the provided URL in your browser to access the web interface.

```bash
$ python app.py
```

Upload a video and click the 'Predict' button. The application will process the video, make predictions on the actions, and display the video with the predicted actions overlaid.

## Dependencies:
- Flask
- TensorFlow
- OpenCV
- MoviePy
- scikit-learn
- Keras

---

**Note**: The dataset used for training the model in `train.py` is expected to be in the `/content/UCF-101` directory. Adjust the paths accordingly based on your dataset location. Also, adjust the actions according to the need , you can add as many actions you need but it should be present in UCF Dataset.
