import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3, 4]:
        sys.exit("Usage: python traffic.py data_directory [model.h5] [image_path]")

    # If mode has been trained, load model
    data_dir = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) >= 3 else None
    image_path = sys.argv[3] if len(sys.argv) == 4 else None

    # Get image arrays and labels for all image files
    images, labels = load_data(data_dir)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # If model file is provided and exists, load the model
    if model_file and os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file)
        print(f"Model loaded from {model_file}")
    else:
        # Get a compiled neural network
        model = get_model()

        # Fit model on training data
        history = model.fit(x_train, y_train, epochs=EPOCHS)

        # Evaluate neural network performance
        model.evaluate(x_test, y_test, verbose=2)

        # Plot
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('Model Accuracy and Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.legend()
        plt.show()    

        # Save model to file if a filename is provided
        if model_file:
            model.save(model_file)
            print(f"Model saved to {model_file}.")

    # If an image path is provided, predict the category of the image
    if image_path:
        predict_image(model, image_path)


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    # Path to data folder
    data_path = os.path.join(data_dir)

    # Number of subdirectories/labels
    number_of_labels = 0
    
    for i in os.listdir(data_path):
        number_of_labels += 1

    # Loop through the subdirectories
    for sub in range(number_of_labels):
        sub_folder = os.path.join(data_path, str(sub))

        images_in_subfolder = []

        for image in os.listdir(sub_folder):
            images_in_subfolder.append(image)

        # Open each image 
            image_path = os.path.join(sub_folder, image) 
            img = Image.open(image_path)
        
            # Add Label
            labels.append(sub)

            # Resize and Add Image
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img = np.array(img)
            images.append(img)
            
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([

        # Convolutional layers and Max-pooling layers
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten units
        tf.keras.layers.Flatten(),


        # Hidden Layers
        tf.keras.layers.Dense(128, activation="relu"),

        
        # Dropout
        tf.keras.layers.Dropout(0.5),


              
        # Output layer with output units for all digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def predict_image(model, image_path):
    """
    Predict the category of the given image using the trained model.

    Parameters:
    - model: The trained Keras model.
    - image_path: Path to the image file to be predicted.
    """
    img = Image.open(image_path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_category = np.argmax(prediction)

    print(f"Predicted category: {predicted_category}")


if __name__ == "__main__":
    main()