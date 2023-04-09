import requests
import PIL
from PIL import ImageFile, Image
import pathlib
import requests
from io import BytesIO
from typing import Tuple, Union
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True


def stream_image(url: str, threshold_size: int = 512, debug: bool = True, user_agent: str = None) -> Image:
    r"""
    Stream an image from the internet.

    Args:
        url (str): The URL of the image to stream.
        rescale_size (tuple or list): The width and height target of the rescaled image.
        threshold_size (int, optional, default=512): The minimum resolution of the image.
        debug (bool, optional, default=False): A flag to toggle debug printing.
        user_agent (str, optional, default=None): The User-Agent header string to include in the request.

    Returns:
        A PIL.Image object if the image was successfully streamed, otherwise None.
    """

    try:
        # get images from the internet
        headers = {}
        if user_agent:
            headers['User-Agent'] = user_agent

        s = requests.Session()
        s.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        response = s.get(url, timeout=1, headers=headers)

        # Open the image using the Pillow library
        try:
            image = Image.open(BytesIO(response.content))

            # Check if the image is large enough
            # if true then return PIL image object
            if image.size[0] >= threshold_size and image.size[1] >= threshold_size:
                return image
            else:
                if debug:
                    print(f"Image {url} is too small, skipping.")
                pass

        except Exception as e:
            if debug:
                print(f"Error opening image {url}: {e}")
            pass

    except Exception as e:
        if debug:
            print(f"Error retrieving image {url}: {e}")
        pass


def rescale_image(image: Union[str, Image.Image], target_resolution: int) -> Image.Image:
    """
    Rescales an image to a target resolution while maintaining aspect ratio based on the shorter axis.

    Args:
        image (Union[str, Image.Image]):
            Either a path to the input image file, or an Image object.
        target_resolution (int):
            The length of the shorter axis of the target resolution.

    Returns:
        An Image object representing the rescaled image.
    """
    # Load the image if it's a file path
    if isinstance(image, str):
        with Image.open(image) as image_file:
            image = image_file.copy()

    # Get the current resolution
    current_resolution = image.size

    # Determine the shorter axis
    shorter_axis = min(current_resolution)

    # Calculate the new resolution while maintaining aspect ratio
    ratio = target_resolution / shorter_axis
    new_resolution = (
        int(current_resolution[0] * ratio), int(current_resolution[1] * ratio))

    # Rescale the image using Lanczos algorithm
    rescaled_image = image.resize(new_resolution, resample=Image.LANCZOS)

    return rescaled_image


def save_webp_without_alpha(image: Image.Image, filepath: str, quality: Union[int, None] = None):
    """
    Saves a PIL Image object as a WebP file, stripping any alpha channel if it exists.

    Args:
        image (PIL.Image.Image): The image to save.
        filepath (str): The file path to save the image to.
        quality (int, optional): The quality of the WebP compression. Must be a value between 0 and 100,
            where higher values indicate higher quality and larger file sizes. If None, the default quality
            of 80 will be used. Defaults to None.
    """
    # Remove alpha channel
    image = image.convert('RGB')

    # Save the image as WebP format
    image.save(filepath, format='WebP', quality=quality)


def split_dataframe(df: pd.DataFrame, split_count: int) -> pd.DataFrame:
    """
    Split a pandas DataFrame into multiple sub-DataFrames as evenly as possible, and store them in a list.

    Args:
        df (pandas.DataFrame): The DataFrame to split.
        split_count (int): The number of splits to create.

    Returns:
        A list of sub-DataFrames. The length of the list will be equal to `split_count`. Each sub-DataFrame will have
        approximately the same number of rows, with any remainder rows being concatenated to the last sub-DataFrame.
    """
    # Determine the number of rows in each split
    num_rows = len(df)
    rows_per_split = num_rows // split_count
    remainder = num_rows % split_count

    # Split the dataframe
    split_dfs = []
    start = 0
    for i in range(split_count):
        end = start + rows_per_split
        if i < remainder:
            end += 1
        split_dfs.append(df.iloc[start:end])
        start = end

    return split_dfs
