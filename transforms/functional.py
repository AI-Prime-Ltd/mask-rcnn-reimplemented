import cv2
import numpy as np


"""
Most functions in this file assume input and output image is np.ndarray of shape NxHxWxC, or HxWxC or HxW,
The array can be of type uint8 in range [0, 255], or floating point in range [0, 1] or [0, 255].
except for:
- `normalize` and `denormalize`, which perform transforms between normalized float64 image and denormalized
float64 image in range [0, 1].
- `posterize` accept uint8 image only.

"""


def bgr2rgb(img: np.ndarray):
    """
    Convert a BGR image to RGB image.
    Args:
        img (ndarray): The input image.
    Returns:
        ndarray: The converted RGB image.
    """
    assert img.ndim >= 3 and img.shape[-1] == 3
    return np.array(img[..., ::-1])


def rgb2bgr(img: np.ndarray):
    """
    Convert a RGB image to BGR image.
    Args:
        img (ndarray): The input image.
    Returns:
        ndarray: The converted BGR image.
    """
    return bgr2rgb(img)


def rgb2gray(img: np.ndarray):
    """
    Convert a RGB image to grayscale image.
    Args:
        img (ndarray): The input image.
    Returns:
        ndarray: The converted grayscale image of shape NxHxWx1 or HxW.
    """
    assert img.ndim >= 3 and img.shape[-1] == 3
    out_img = np.sum(img * [0.2989, 0.5870, 0.1140], axis=-1, keepdims=True, dtype=img.dtype)
    return out_img


def gray2rgb(img: np.ndarray):
    """
    Convert a grayscale image to RGB image.
    Args:
        img (ndarray): The input image of shape HxW or NxHxWx1.
    Returns:
        ndarray: The converted BGR image of shape NxHxWx3 or HxWx3.
    """
    assert img.ndim == 2 or img.shape[-1] == 1
    out_img = np.array(img)
    if img.ndim == 2:   # HxW
        out_img = out_img[..., None]
    out_img = np.tile(out_img, (1, ) * (out_img.ndim - 1) + (3, ))
    return out_img


def gray2bgr(img):
    """
    Convert a grayscale image to RGB image.
    Args:
        img (ndarray): The input image.
    Returns:
        ndarray: The converted RGB image.
    """
    return gray2rgb(img)


def normalize(img, mean, std):
    """
    Normalize an image with mean and std. Perform `(img - mean) / std`.
    Args:
        img (ndarray): Image to be normalized, float in range [0, 1] or uint8 in [0, 255].
        mean (ndarray): The mean to be substracted, should be either a float or array-like channel-wise float mean.
        std (ndarray): The std to be devided, should be either a number or array-like channel-wise float std.
        to_rgb (bool): Whether to convert to rgb.
    Returns:
        ndarray: The normalized float64 image.
    """
    stdinv = 1. / np.float64(std)
    return (img - mean) * stdinv


def denormalize(img, mean, std):
    """
    Denormalize an image with mean and std. Perform `img * std + mean`.
    Args:
        img (ndarray): Image to be denormalized, float64.
        mean (ndarray): The mean substracted, should be either a float or array-like channel-wise float mean.
        std (ndarray): The std devided, should be either a number or array-like channel-wise float std.
    Returns:
        ndarray: The denormalized float64 image in range [0, 1].
    """

    return np.clip(np.float64(img) * std + mean, a_min=0., a_max=1.)


def posterize(img, bits):
    """
    Posterize an image (reduce the number of bits for each color channel)
    Args:
        img (ndarray): Image to be posterized, uint8.
        bits (int): Number of bits (1 to 8) to use for posterizing.
    Returns:
        ndarray: The posterized image.
    """
    shift = 8 - bits
    img = np.left_shift(np.right_shift(img, shift), shift)
    return img


def adjust_color(img, alpha=1, beta=None, gamma=0):
    """It blends the source image and its gray image:
    ``output = img * alpha + gray_img * beta + gamma``
    Args:
        img (ndarray): The input source uint8 RGB image.
        alpha (int | float): Weight for the source image. Default 1.
        beta (int | float): Weight for the converted gray image.
            If None, it's assigned the value (1 - `alpha`).
        gamma (int | float): Scalar added to each sum.
            Same as :func:`cv2.addWeighted`. Default 0.
    Returns:
        ndarray: Colored image which has the same size and dtype as input.
    """
    gray_img = rgb2gray(img)
    gray_img = np.tile(gray_img, (1, ) * (gray_img.ndim - 1) + (3, ))
    if beta is None:
        beta = 1 - alpha
    colored_img = cv2.addWeighted(img, alpha, gray_img, beta, gamma)
    if not colored_img.dtype == np.uint8:
        colored_img = np.clip(colored_img, 0., 1.)
    return colored_img


def equalize(img):
    """
    Equalize the image histogram.
    This function applies a non-linear mapping to the input image,
    in order to create a uniform distribution of grayscale values
    in the output image.
    Args:
        img (ndarray): Image to be equalized.
    Returns:
        ndarray: The equalized image.
    """

    def _scale_channel(im, c):
        """Scale the data in the corresponding channel."""
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = np.histogram(im, 256, (0, 255))[0]
        # For computing the step, filter out the nonzeros.
        nonzero_histo = histo[histo > 0]
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        if not step:
            lut = np.array(range(256))
        else:
            # Compute the cumulative sum, shifted by step // 2
            # and then normalized by step.
            lut = (np.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = np.concatenate([[0], lut[:-1]], 0)
        # If step is zero, return the original image.
        # Otherwise, index from lut.
        return np.where(np.equal(step, 0), im, lut[im])

    # Scales each channel independently and then stacks
    # the result.
    s1 = _scale_channel(img, 0)
    s2 = _scale_channel(img, 1)
    s3 = _scale_channel(img, 2)
    equalized_img = np.stack([s1, s2, s3], axis=-1)
    return equalized_img


def adjust_brightness(img: np.ndarray, factor=1.):
    """Adjust image brightness.
    This function controls the brightness of an image. An
    enhancement factor of 0.0 gives a black image.
    A factor of 1.0 gives the original image. This function
    blends the source image and the degenerated black image:
    ``output = img * factor + degenerated * (1 - factor)``
    Args:
        img (ndarray): uint8 RGB image to be brightened.
        factor (float): A value controls the enhancement.
            Factor 1.0 returns the original image, lower
            factors mean less color (brightness, contrast,
            etc), and higher values more. Default 1.
    Returns:
        ndarray: The brightened image.
    """
    degenerated = np.zeros_like(img)
    # Note manually convert the dtype to np.float32, to
    # achieve as close results as PIL.ImageEnhance.Brightness.
    # Set beta=1-factor, and gamma=0
    brightened_img = cv2.addWeighted(img.astype(np.float32), factor, degenerated.astype(np.float32), 1 - factor, 0)
    return brightened_img.astype(img.dtype)


def adjust_contrast(img: np.ndarray, factor=1.):
    """Adjust image contrast.
    This function controls the contrast of an image. An
    enhancement factor of 0.0 gives a solid grey
    image. A factor of 1.0 gives the original image. It
    blends the source image and the degenerated mean image:
    ``output = img * factor + degenerated * (1 - factor)``
    Args:
        img (ndarray): uint8 RGB image.
        factor (float): Same as :func:`mmcv.adjust_brightness`.
    Returns:
        ndarray: The contrasted image.
    """
    gray_img = rgb2gray(img)
    hist = np.histogram(gray_img, 256, (0, 255 if img.dtype == np.uint8 else 1.))[0]
    mean = round(np.sum(gray_img) / np.sum(hist))
    degenerated = (np.ones_like(img[..., 0]) * mean).astype(img.dtype)
    degenerated = gray2rgb(degenerated)
    contrasted_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32),
        1 - factor, 0)
    return contrasted_img.astype(img.dtype)
