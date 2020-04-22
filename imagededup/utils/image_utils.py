from pathlib import PurePath
from typing import List, Union, Tuple

import cv2
import numpy as np

from imagededup.utils.logger import return_logger

IMG_FORMATS = ['JPEG', 'PNG', 'BMP', 'WEBP', 'MPO', 'PPM', 'TIFF', 'GIF']
logger = return_logger(__name__)


def preprocess_image(
        image, target_size: Tuple[int, int] = None, grayscale: bool = False
) -> np.ndarray:
    """
    Take as input an image as numpy array or opencv mat format. Returns an array version of optionally resized and grayed
    image.

    Args:
        image: numpy array or a opencv mat.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.

    Returns:
        A numpy array of the processed image.
    """
    if isinstance(image, np.ndarray):
        image = image.astype('uint8')

    if target_size:
        image = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_CUBIC)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return np.array(image).astype('uint8')


def load_image(
        image_file: Union[PurePath, str],
        target_size: Tuple[int, int] = None,
        grayscale: bool = False,
        img_formats: List[str] = IMG_FORMATS,
) -> np.ndarray:
    """
    Load an image given its path. Returns an array version of optionally resized and grayed image. Only allows images
    of types described by img_formats argument.

    Args:
        image_file: Path to the image file.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.
        img_formats: List of allowed image formats that can be loaded.
    """
    try:
        img = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img, target_size=target_size, grayscale=grayscale)
        return img

    except Exception as e:
        logger.warning(f'Invalid image file {image_file}:\n{e}')
        return None
