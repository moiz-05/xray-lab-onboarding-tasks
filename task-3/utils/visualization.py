import os
import cv2


def save_visualization(output_path, image):
    """
    Save visualization image.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, image)

    print(f"Saved Output: {output_path}")