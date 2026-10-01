from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def draw_black_text(frame, text, position, font_size=24, color_bgr=(0, 255, 0)):
    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Black.ttf"
    return draw_text(frame, text, position, font_size, color_bgr, font_path=font_path)


def draw_bold_text(frame, text, position, font_size=24, color_bgr=(0, 255, 0)):
    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Bold.ttf"
    return draw_text(frame, text, position, font_size, color_bgr, font_path=font_path)


def draw_heavy_text(frame, text, position, font_size=24, color_bgr=(0, 255, 0)):
    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Heavy.ttf"
    return draw_text(frame, text, position, font_size, color_bgr, font_path=font_path)


def draw_light_text(frame, text, position, font_size=24, color_bgr=(0, 255, 0)):
    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Light.ttf"
    return draw_text(frame, text, position, font_size, color_bgr, font_path=font_path)


def draw_medium_text(frame, text, position, font_size=24, color_bgr=(0, 255, 0)):
    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Medium.ttf"
    return draw_text(frame, text, position, font_size, color_bgr, font_path=font_path)


def draw_regular_text(frame, text, position, font_size=24, color_bgr=(0, 255, 0)):
    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Regular.ttf"
    return draw_text(frame, text, position, font_size, color_bgr, font_path=font_path)


def draw_semibold_text(frame, text, position, font_size=24, color_bgr=(0, 255, 0)):
    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Semibold.ttf"
    return draw_text(frame, text, position, font_size, color_bgr, font_path=font_path)


def draw_thin_text(frame, text, position, font_size=24, color_bgr=(0, 255, 0)):
    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Thin.ttf"
    return draw_text(frame, text, position, font_size, color_bgr, font_path=font_path)


def draw_text(
    frame, text, position, font_size=24, color_bgr=(0, 255, 0), font_path=None
):
    """
    Renders custom TrueType (.ttf) text onto an OpenCV frame.

    Parameters:
        frame: OpenCV BGR image frame
        text: String to display
        position: Tuple of (x, y) coordinates
        font_size: Size of the font
        color_bgr: Tuple of (Blue, Green, Red) for text color
        font_path: Path to the TrueType font file (.ttf)
    Return:
        Frame with text rendered
    """

    if font_path is None:
        font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Regular.ttf"

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except IOError:
        font = ImageFont.load_default()

    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    draw.text(position, text, font=font, fill=color_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
