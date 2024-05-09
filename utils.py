import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.io import imread
import numpy as np

def load_image(img_path):
    """
    Load and preprocess an image.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    H = 480
    W = 480

    # Read Image and Preprocess
    read_img = imread(img_path)
    crop_img = read_img[:W, :H]
    norm_img = crop_img / 255.0
    preprocess_img = norm_img.astype(np.float32)

    return preprocess_img

def colorful_mask(pred_mask):
    """
    Convert a grayscale mask to a colorful mask.

    Parameters:
        pred_mask (numpy.ndarray): Grayscale predicted mask.

    Returns:
        numpy.ndarray: Colorful predicted mask.
    """
    color_map = {0: [0, 0, 0],       # Black
                 1: [128, 0, 128],   # Purple
                 2: [65, 105, 225],  # Royal Blue
                 3: [255, 215, 0]}   # Gold

    colored_image = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

    for i in range(pred_mask.shape[0]):
        for j in range(pred_mask.shape[1]):
            colored_image[i, j] = color_map[pred_mask[i, j]]

    return colored_image

def predict(preprocess_img, model):
    """
    Make prediction using a model.

    Parameters:
        preprocess_img (numpy.ndarray): Preprocessed image.
        model: Trained model.

    Returns:
        numpy.ndarray: Predicted mask.
    """
    pred_mask = model.predict(np.expand_dims(preprocess_img, axis=0))
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0]

    return pred_mask

def lunar_model(model_path):
    """
    Load a pre-trained Lunar Model.

    Parameters:
        model_path (str): Path to the Lunar Model file.

    Returns:
        tensorflow.keras.Model: Loaded Lunar Model.
    """
    return load_model(model_path)
