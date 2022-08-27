import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
# import math
# from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, dimensions):
    mask = np.zeros_like(img)
    # defining weather a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask_color = (100,) * channel_count
    else:
        mask_color = 100
    mask = cv2.fillPoly(mask, dimensions, (100, 100, 100))
    masked_edges = cv2.bitwise_and(img, mask)
    return masked_edges


def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=10)
    return img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def weighted_img(img, initial_img, a, b, c):
    return cv2.addWeighted(initial_img, a, img, b, c)


def make_coordinates(img, line_parameters):
    slope = line_parameters[0]
    intercept = line_parameters[1]
    y1 = img.shape[0]
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intersect = parameters[1]
        if slope < 0:
            left_fit.append((slope, intersect))
        elif slope > 0:
            right_fit.append((slope, intersect))
        print(slope, "-", intersect)
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(img, left_fit_average)
    right_line = make_coordinates(img, right_fit_average)
    return np.array([left_line, right_line])


# converting image to an array of numbers
image = mpimg.imread('solidYellowLeft.jpg')
img_copy = np.copy(image)
gray_img = grayscale(img_copy)
blur_img = gaussian_blur(gray_img, 5)
canny_img = canny(blur_img, 50, 150)
im_shape = image.shape
vertices = np.array([[(100, 560), (480, 310), (900, 560)]], dtype=np.int32)
region_image = region_of_interest(canny_img, vertices)
hough_lines_ = hough_lines(region_image, 2, np.pi / 180, 15, 40, 20)
line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
hough_lines_image = draw_lines(line_img, hough_lines_)
average_image = average_slope_intercept(img_copy, hough_lines_)
final_average_image = draw_lines(img_copy, average_image)
Weighted_image = weighted_img(final_average_image, image, 0.4, 0.8, 1)
plt.imshow(region_image)
# plt.show()
# mpimg.im_save('Solid_white_right-after.jpg', Weighted_image)

cap = cv2.VideoCapture('solidWhiteRight.mp4')
while cap.isOpened():
    video_frames = cap.read()
    frame = video_frames[1]
    img_copy = np.copy(frame)
    gray_img = grayscale(img_copy)
    blur_img = gaussian_blur(gray_img, 5)
    canny_img = canny(blur_img, 50, 150)
    im_shape = frame.shape
    vertices = np.array([[(100, 560), (480, 310), (900, 560)]], dtype=np.int32)
    region_image = region_of_interest(canny_img, vertices)
    hough_lines_ = hough_lines(region_image, 1, np.pi / 180, 25, 20, 20)
    line_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    hough_lines_image = draw_lines(line_img, hough_lines_)
    average_image = average_slope_intercept(img_copy, hough_lines_)
    final_average_image = draw_lines(img_copy, average_image)
    Weighted_image = weighted_img(final_average_image, frame, 0.4, 0.8, 1)
    cv2.imshow("result", Weighted_image)
    cv2.waitKey(40)
    # plt.show()
