from PIL import Image, ImageDraw  # pillow package
import numpy as np
from scipy import ndimage


def read_img_as_array(file):
    """read image and convert it to numpy array with dtype np.float64"""
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr


def save_array_as_img(arr, file):
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min) / (max - min) * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)


def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
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



def sobel(arr):
    """Apply sobel operator on arr and return the result."""
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = ndimage.convolve(arr, x_filter)
    Gy = ndimage.convolve(arr, y_filter)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    return G, Gx, Gy


def nonmax_suppress(G, Gx, Gy):
    """Suppress non-max value along direction perpendicular to the edge."""
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    supressed_G = np.zeros((G.shape[0], G.shape[1]))
    for x in range(G.shape[0] - 1):
        for y in range(G.shape[1] - 1):
            if (
                (-22.5 <= theta[x, y] < 22.5)
                or (157.5 <= theta[x, y] <= 180)
                or (-180 <= theta[x, y] <= -157.5)
            ):
                N1 = G[x, y + 1]
                N2 = G[x, y - 1]
            elif (22.5 <= theta[x, y] < 67.5) or (-157.5 <= theta[x, y] < -112.5):
                N1 = G[x + 1, y + 1]
                N2 = G[x - 1, y - 1]
            elif (67.5 <= theta[x, y] < 112.5) or (-112.5 <= theta[x, y] < -67.5):
                N1 = G[x + 1, y]
                N2 = G[x - 1, y]
            elif (112.5 <= theta[x, y] < 157.5) or (-67.5 <= theta[x, y] < -22.5):
                N1 = G[x - 1, y + 1]
                N2 = G[x + 1, y - 1]
            if (G[x, y] < N1) or (G[x, y] < N2):
                supressed_G[x, y] = 0
            else:
                supressed_G[x, y] = G[x, y]
    return supressed_G


def thresholding(G, t):
    """Binarize G according threshold t"""
    G_binary = G.copy()
    G_binary[G <= t] = 0
    # show_array_as_img(G_binary)
    return G_binary


def hysteresis_thresholding(G, low, high):
    """Binarize G according threshold low and high"""
    G_low = thresholding(G, low)
    G_high = thresholding(G, high)
    # Set the G_hyst equals to the strong edge first
    G_hyst = G_high.copy()
    for x in range(G.shape[0] - 1):
        for y in range(G.shape[1] - 1):
            # check the pixels which are between low and high boundary
            if (G_low[x, y] != 0) and (G_hyst[x, y] == 0):
                # if the surrounding 8 pixels are strong edge or checked it as connected to strong edge in previous loops
                # assign the value to the pixel, if not, assign 0 to it
                if (
                    (G_hyst[x + 1, y] != 0)
                    or (G_hyst[x + 1, y + 1] != 0)
                    or (G_hyst[x, y + 1] != 0)
                    or (G_hyst[x - 1, y] != 0)
                    or (G_hyst[x - 1, y - 1] != 0)
                    or (G_hyst[x, y - 1] != 0)
                    or (G_hyst[x + 1, y - 1] != 0)
                    or (G_hyst[x - 1, y + 1] != 0)
                ):
                    G_hyst[x, y] = G_low[x, y]
                else:
                    G_hyst[x, y] = 0
    return G_low, G_high, G_hyst


def hough(G_hyst):
    """Return Hough transform of G"""
    d = int(np.sqrt(np.square(G_hyst.shape[0]) + np.square(G_hyst.shape[1])))
    num_rhos = d * 2 + 1
    num_thetas = 1000
    thetas = np.deg2rad(np.linspace(0, 180, num=num_thetas))
    accumulator = np.zeros([num_rhos, num_thetas], dtype=np.int16)
    for x in range(G_hyst.shape[0]):
        for y in range(G_hyst.shape[1]):
            if G_hyst[x, y] > 0:
                for theta_index, theta in enumerate(thetas):
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    # add d to the value of rho, so that for min. rho = -d -> 0
                    rho_index = int(rho) + d
                    accumulator[rho_index, theta_index] = (
                        accumulator[rho_index, theta_index] + 1
                    )
    # find the top 10 [rho, theta] pairs index
    top_10_index = np.argpartition(accumulator.flatten(), -10)[-10:]
    top_10_pairs = []
    for index in top_10_index:
        rho_index, theta_index = np.unravel_index(index, accumulator.shape)
        # rho_index minus d equals to the actual valude of rho
        rho = rho_index - d
        # also find the actual value of radian theta in the thetas array
        theta = thetas[theta_index]
        top_10_pairs.append([rho, theta])
    return accumulator, top_10_pairs


if __name__ == "__main__":
    input_path = "data/road.jpeg"
    img = read_img_as_array(input_path)
    gray = rgb2gray(img)
    save_path = "./data/2.1_gray.jpg"
    # show_array_as_img(gray)
    save_array_as_img(gray, save_path)

    smoothed_img = ndimage.gaussian_filter(gray, sigma=2)
    # show_array_as_img(smoothed_img)
    save_array_as_img(smoothed_img, "./data/2.2_gauss.jpg")

    G, Gx, Gy = sobel(smoothed_img)
    # show_array_as_img(G)
    save_array_as_img(G, "./data/2.3_G.jpg")
    save_array_as_img(Gx, "./data/2.3_G_x.jpg")
    save_array_as_img(Gy, "./data/2.3_G_y.jpg")

    supress = nonmax_suppress(G, Gx, Gy)
    # show_array_as_img(supress)
    save_array_as_img(supress, "./data/2.4_supress.jpg")

    G_low, G_high, G_hyst = hysteresis_thresholding(supress, 10, 60)
    # show_array_as_img(G_hyst)
    save_array_as_img(G_low, "./data/2.5_edgemap_low.jpg")
    save_array_as_img(G_high, "./data/2.5_edgemap_high.jpg")
    save_array_as_img(G_hyst, "./data/2.5_edgemap.jpg")

    hough_img, top_10_pairs = hough(G_hyst)
    save_array_as_img(hough_img, "./data/2.6_hough.jpg")
    original_image = Image.open(input_path)
    draw = ImageDraw.Draw(original_image)
    for x in range(G_hyst.shape[0]):
        for y in range(G_hyst.shape[1]):
            for i in range(10):
                rho = top_10_pairs[i][0]
                theta = top_10_pairs[i][1]
                if rho == int(x * np.cos(theta) + y * np.sin(theta)):
                    draw.point((y, x), fill="#FF0000")
    # original_image.show()
    original_image.save("./data/2.7_detection_result.jpg", "JPEG")
