import cv2


def load_image(image_path):
    """
    Load image using OpenCV.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image


def convert_bgr_to_rgb(image):
    """
    Convert OpenCV BGR image to RGB.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)