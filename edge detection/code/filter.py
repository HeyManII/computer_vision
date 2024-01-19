from PIL import Image  # pillow package
import numpy as np
from scipy import ndimage


def read_img_as_array(file):
    """read image and convert it to numpy array with dtype np.float64"""
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr


def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min) / (max - min) * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)


def show_array_as_img(arr, rescale="minmax"):
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min) / (max - min) * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()


def rgb2gray(arr):
    R = arr[:, :, 0]  # red channel
    G = arr[:, :, 1]  # green channel
    B = arr[:, :, 2]  # blue channel
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return gray


def sharpen(img, sigma, alpha):
    """Sharpen the image. 'sigma' is the standard deviation of Gaussian filter. 'alpha' controls how much details to add."""
    smoothed_img = ndimage.gaussian_filter(img, sigma=sigma)
    # show_array_as_img(smoothed_img)
    detailed_img = img - smoothed_img
    # show_array_as_img(detailed_img)
    arr = img + alpha * detailed_img
    for z in range(arr.shape[2]):
        for y in range(arr.shape[1]):
            for x in range(arr.shape[0]):
                if arr[x][y][z] > 255:
                    arr[x][y][z] = 255
                if arr[x][y][z] < 0:
                    arr[x][y][z] = 0
    return arr


def median_filter(img, s):
    """Perform median filter of size s x s to image 'arr', and return the filtered image."""
    arr = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    half_s = int(s / 2)
    for z in range(img.shape[2]):
        # add padding for catering the border pixels
        img_with_padding = np.pad(
            img[:, :, z],
            ((half_s, half_s), (half_s, half_s)),
            "constant",
            constant_values=(0, 0),
        )
        for y in range(half_s, img.shape[1]):
            for x in range(half_s, img.shape[0]):
                temp = []
                if s % 2 == 0:
                    for i in range(x - half_s, x + half_s):
                        for j in range(y - half_s, y + half_s):
                            temp.append(img_with_padding[i, j])
                    temp.sort()
                    # for even length of array, take the average of the middle two values and divided by 2
                    arr[x, y, z] = round(
                        (temp[int((s * s) / 2)] + temp[int(((s * s) / 2) - 1)]) / 2
                    )
                elif s % 2 == 1:
                    for i in range(x - half_s, x + half_s + 1):
                        for j in range(y - half_s, y + half_s + 1):
                            temp.append(img_with_padding[i, j])
                    temp.sort()
                    arr[x, y, z] = temp[int((s * s) / 2)]
    return arr


if __name__ == "__main__":
    input_path = "./data/rain.jpeg"
    img = read_img_as_array(input_path)
    # show_array_as_img(img)
    sharpen_img = sharpen(img, [3, 3, 0], 1.2)
    # show_array_as_img(sharpen_img)
    save_array_as_img(sharpen_img, "./data/1.1_sharpened.jpg")
    remove_noise = median_filter(img, 5)
    # show_array_as_img(remove_noise)
    save_array_as_img(remove_noise, "./data/1.2_derained.jpg")
