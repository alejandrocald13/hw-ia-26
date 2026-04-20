import cv2
from PIL import Image, ImageTk


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def cv_to_tk(image):
    rgb = bgr_to_rgb(image)
    pil_image = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_image)