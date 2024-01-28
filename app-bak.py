# app.py
from flask import Flask, render_template, request
import os
from cnn_predictions import cnn_predictions, load_cnn_model
from deeplearning import object_detection, yolo_predictions

app = Flask(__name__)
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

# Replace 'path/to/your/cnn_model.h5' with the actual path to your trained CNN model file
cnn_model = load_cnn_model('./static/models/pest_detection_model.h5') 

@ app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)

        # Check which algorithm button is clicked
        algorithm = request.form.get('algorithm', 'ssd')

        if algorithm == 'cnn':
            cnn_result, cnn_accuracy = cnn_predictions(path_save, cnn_model)
            return render_template('index.html', upload=True, upload_image=filename, text=cnn_result, no=f'CNN Prediction - Accuracy: {cnn_accuracy:.2%}')

        elif algorithm == 'yolov3':
            yolov3_result, no_detection =  object_detection(path_save,filename)
            return render_template('index.html', upload=True, upload_image=filename, text=yolov3_result, no=no_detection)

        else:
            # Continue with other algorithms (e.g., SSD)
            no_detection, label = object_detection(path_save, filename)
            if label == '':
                label = 'Not Detected'

            return render_template('index.html', upload=True, upload_image=filename, text=label, no=no_detection)

    return render_template('index.html', upload=False)

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8009)
