import numpy as np
from PIL import ImageDraw, Image

from env.utils_v1 import load_map_normalized


def save_map_with_tx(filepath: str, pixel_map: np.ndarray, mark_size: int,
                     target_locs: list, curr_locs: list) -> None:
    """Save building map array as a black-white image and mark TX locations.

    Args:
        filepath: Path of the image.
        pixel_map: Building map array.
        mark_size: Size of the marker.
        target_locs: Coordinates of target markers.
        curr_locs: Coordinates of current markers.

    Returns:

    """
    # convert the binary array to an image
    image_from_array = Image.fromarray((255 * pixel_map).astype(np.uint8), mode='L')
    image_from_array = image_from_array.convert('RGB')
    # Create an ImageDraw object to draw on the image
    draw = ImageDraw.Draw(image_from_array)

    for loc in target_locs:
        x, y = loc[1], loc[0]
        map_size = pixel_map.shape[0]
        top_left = (max(0, x - (mark_size - 1) // 2), max(0, y - (mark_size - 1) // 2))
        bottom_right = (min(map_size, x + mark_size // 2 + 1), min(map_size, y + mark_size // 2 + 1))
        # Draw the red point
        draw.rectangle((top_left, bottom_right), fill="red")

    for loc in curr_locs:
        x, y = loc[1], loc[0]
        map_size = pixel_map.shape[0]
        top_left = (max(0, x - (mark_size - 1) // 2), max(0, y - (mark_size - 1) // 2))
        bottom_right = (min(map_size, x + mark_size // 2 + 1), min(map_size, y + mark_size // 2 + 1))
        # draw.point(xy, fill='blue')
        draw.rectangle((top_left, bottom_right), fill="blue")

    image_from_array.save(filepath)

if __name__ == '__main__':
    save_map_with_tx('./test.png', pixel_map=load_map_normalized('../resource/usc_old_sparse/map/1.png'),
                     mark_size=5, target_locs=[[12, 120], [35, 37]], curr_locs=[[90,90], [200,200]])
