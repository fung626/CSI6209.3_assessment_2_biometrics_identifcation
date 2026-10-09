from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import COLOR_DARK_CHARCOAL_RGB, COLOR_GRAY_RGB, COLOR_WHITE_RGB

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import datetime


def draw_timestamp_badge(frame):
    """
    Renders a professional timestamp badge with rounded corners
    and SF Pro typography on the bottom-left corner.
    """
    height, width, _ = frame.shape

    x = 20
    w = 170
    h = 36
    y = height - h - 20
    corner_radius = 16

    bg_color_rgb = COLOR_DARK_CHARCOAL_RGB  # Dark charcoal background
    text_color_rgb = COLOR_WHITE_RGB  # White text

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    draw.rounded_rectangle(
        [(x, y), (x + w, y + h)],
        radius=corner_radius,
        fill=bg_color_rgb,
        width=1,
    )

    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Semibold.ttf"
    try:
        font = ImageFont.truetype(str(font_path), 13)
    except Exception:
        font = ImageFont.load_default()

    current_time = datetime.datetime.now().strftime("%d-%m-%Y • %H:%M:%S")
    timestamp_text = f"{current_time}"

    bbox = draw.textbbox((0, 0), timestamp_text, font=font)
    text_h = bbox[3] - bbox[1]

    text_x = x + 14
    text_y = y + (h - text_h) // 2 - bbox[1]

    draw.text((text_x, text_y), timestamp_text, font=font, fill=text_color_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_live_badge(frame):
    x, y, w, h = 16, 16, 138, 38
    corner_radius = 16

    bg_color_rgb = COLOR_DARK_CHARCOAL_RGB  # Dark charcoal background
    border_color_rgb = COLOR_GRAY_RGB  # Subtle gray border
    text_color_rgb = COLOR_WHITE_RGB  # White text

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    draw.rounded_rectangle(
        [(x, y), (x + w, y + h)],
        radius=corner_radius,
        fill=bg_color_rgb,
        outline=border_color_rgb,
        width=1,
    )

    icon_path = PROJECT_ROOT / "assets" / "icons" / "camera.png"
    if icon_path.exists():
        try:
            icon_img = Image.open(icon_path).convert("RGBA")
            icon_img = icon_img.resize((18, 18), Image.Resampling.LANCZOS)
            icon = Image.new("RGBA", icon_img.size, (255, 0, 0, 255))
            icon.putalpha(icon_img.split()[3])
            icon_x = x + 12
            icon_y = y + (h - 18) // 2
            pil_img.paste(icon, (icon_x, icon_y), icon)
        except Exception as e:
            print(f"Error loading camera icon: {e}")

    font_path = PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Bold.ttf"
    try:
        font = ImageFont.truetype(str(font_path), 15)
    except IOError:
        font = ImageFont.load_default()

    text_pos = (x + 36, y + 9)
    draw.text(text_pos, "Live Camera", font=font, fill=text_color_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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
