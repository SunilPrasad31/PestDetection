import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Assuming 10 output classes, adjust as needed

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
from tensorflow.keras.models import load_model

def load_cnn_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading CNN model: {str(e)}")
        return None

# Example usage:
cnn_model_path = './static/models/pest_detection_model.h5'
cnn_model = load_cnn_model(cnn_model_path)

if cnn_model is not None:
    cnn_model = load_model('./static/models/pest_detection_model.h5')
else:
    print("Failed to load the CNN model.")

# Load CNN model
 # Replace with the actual path to your CNN model file
def cnn_predictions(img_path, cnn_model):
    # Load and preprocess the image
    image = load_img(img_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]

    # Perform prediction
    predictions = cnn_model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0, predicted_class_index] * 100

    # Map the predicted class index to a label (adjust as per your class labels)
    class_labels = {
        0: 'aphids',
        1: 'armyworm',
        2: 'beetle',
        3: 'bollworm',
        4: 'grasshopper',
        5: 'mites',
        6: 'mosquito',
        7: 'sawfly',
        8: 'stem_borer',
        9: 'No_Bug'
    }
    predicted_class_label = class_labels.get(predicted_class_index, 'Unknown')

    return predicted_class_label, confidence

def object_detection(path, filename):
    # Read image
    image = cv2.imread(path)
    image = np.array(image, dtype=np.uint8)

    cnn_model = load_model('./static/models/pest_detection_model.h5') 
    # Perform CNN prediction
    label, confidence = cnn_predictions(path, cnn_model)

    # Optionally, you can perform additional processing based on the CNN prediction
    # ...

    # Save or display the result as needed
    # ...

    return label, confidence

# The rest of your code remains the same, just use the object_detection function for CNN predictions
# ...
