#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ScriptName  : lane_detection.py
@Project     : lane_detection
@Author      : Meng Peng
@Date        : 18-07-2020
@Description : the basic lane detection method based on OpenCV
"""
import os
import cv2
import math
import numpy as np


def get_base_path(relative):
    base_path = os.path.abspath("..")
    return os.path.join(base_path, relative)


def get_image_path(filename):
    return get_base_path(os.path.join('resource', filename))


def get_image_name(path):
    (image_path, image_full_name) = os.path.split(path)
    (image_name, image_type) = os.path.splitext(image_full_name)
    return image_name


def set_save_path(name, suffix):
    save_dir = os.path.join(get_base_path('result'), 'detail')
    os.makedirs(save_dir, exist_ok=True)
    save_name = name + suffix + '.jpg'
    save_path = os.path.join(save_dir, save_name)
    return save_path


# ########## full lane detection functions ##########
def grayscale(img, rgb=False):
    if rgb:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


def hls_color_space(img, rgb=False):
    if rgb:
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def color_mask(img, low_thresh, high_thresh):
    assert(0 <= low_thresh.all() <= 255)
    assert(0 <= high_thresh.all() <= 255)
    return cv2.inRange(img, low_thresh, high_thresh)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def roi_mask(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by vertices with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def get_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def draw_lines(img, lines, color=None, thickness=2):
    color = [0, 255, 0] if color is None else color
    # create an image filled with zero intensities
    none_img = np.zeros_like(img)
    # draw lines
    lines_img = draw_lines_on_image(none_img, lines, color, thickness)
    return lines_img


def draw_lines_on_image(img, lines, color=None, thickness=2):
    color = [0, 255, 0] if color is None else color
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def get_lanes(img, lines, bottom=None, top=None):
    # empty arrays for slope and intercept of the left and right lines
    left_params = []
    right_params = []

    # set threshold of excepted slope
    slope_min = 0.3
    slope_max = 3.3

    for line in lines:
        slope, intercept = get_line_slope_intercept(line)
        if slope == math.inf:
            continue

        if slope_min < np.absolute(slope) < slope_max:
            # if slope is negative, the line is to the left of the lane,
            # otherwise, the line is to the right of the lane
            if slope < 0:
                left_params.append((slope, intercept))
            else:
                right_params.append((slope, intercept))
    # average params into a single slope and intercept value for each line
    left_params_avg = np.average(left_params, axis=0)
    right_params_avg = np.average(right_params, axis=0)
    # calculate the x1, y1, x2, y2 coordinates for the left and right lane
    left_lane = calculate_coordinates(img, left_params_avg, bottom, top)
    right_lane = calculate_coordinates(img, right_params_avg, bottom, top)

    return np.array([[left_lane, right_lane]])


def get_line_slope_intercept(line):
    for x1, y1, x2, y2 in line:
        if x1 == x2:
            slope = math.inf
            intercept = 0
            return slope, intercept
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept


def calculate_coordinates(img, params, bottom, top):
    try:
        slope, intercept = params
        # set initial y-coordinate
        y1 = img.shape[0] if bottom is None else bottom
        # set final y-coordinate
        y2 = int(img.shape[0] / 2) if top is None else top
        # set initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
        x1 = int((y1 - intercept) / slope)
        # set final x-coordinate as (y2 - b) / m since y2 = mx2 + b
        x2 = int((y2 - intercept) / slope)
    except TypeError:
        return np.array([], dtype=np.int32)
    return np.array([x1, y1, x2, y2], dtype=np.int32)


# ##### full processes of lane detection #####
def lane_detection(img, rgb=False, show=False, save=False, name=None):
    # get original image
    original_img = np.copy(img)
    if show:
        cv2.imshow('image original', original_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_original'), original_img)

    # convert to grayscale
    gray_img = grayscale(img, rgb)
    if show:
        cv2.imshow('image grayscale', gray_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_gray'), gray_img)

    # darken the grayscale
    darkened_img = adjust_gamma(gray_img, 0.6)
    if show:
        cv2.imshow('image dark', darkened_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_dark'), darkened_img)

    # convert original image to HLS color space
    hls_img = hls_color_space(img, rgb)
    if show:
        cv2.imshow('image HLS', hls_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_HLS'), hls_img)

    # color selection
    # select white
    white_lower = np.array([0, 200, 0], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = color_mask(hls_img, white_lower, white_upper)
    if show:
        cv2.imshow('image white mask', white_mask)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_white_mask'), white_mask)

    # select yellow
    yellow_lower = np.array([15, 0, 100], dtype=np.uint8)
    yellow_upper = np.array([40, 200, 255], dtype=np.uint8)
    yellow_mask = color_mask(hls_img, yellow_lower, yellow_upper)
    if show:
        cv2.imshow('image yellow mask', yellow_mask)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_yellow_mask'), yellow_mask)

    # combine the white and yellow mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    colored_img = cv2.bitwise_and(darkened_img, mask)
    if show:
        cv2.imshow('image color mask', colored_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_color_mask'), colored_img)

    # apply Gaussian blur
    blurred_img = gaussian_blur(colored_img, kernel_size=5)
    if show:
        cv2.imshow('image gaussian blur', blurred_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_blur'), blurred_img)

    # Canny edge detection
    canny_img = canny(blurred_img, low_threshold=60, high_threshold=150)
    if show:
        cv2.imshow('image canny', canny_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_canny'), canny_img)

    # region of interest
    roi_vtx = np.array([[(0, img.shape[0]),
                         (img.shape[1] * 0.4, img.shape[0] * 0.6),
                         (img.shape[1] * 0.6, img.shape[0] * 0.6),
                         (img.shape[1], img.shape[0])]], dtype=np.int32)
    roi_img = roi_mask(canny_img, roi_vtx)
    if show:
        cv2.imshow('image roi', roi_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_roi'), roi_img)

    # draw Hough lines
    hough_lines = get_hough_lines(roi_img, rho=1, theta=np.pi/180, threshold=20,
                                  min_line_len=30, max_line_gap=100)
    hough_img = draw_lines(img, hough_lines, [0, 255, 0], thickness=2)
    if show:
        cv2.imshow('image hough lines', hough_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_hough_lines'), hough_img)

    # draw lanes
    lanes = get_lanes(img, hough_lines, bottom=int(img.shape[0]),
                      top=int(img.shape[0]*0.65))
    lane_img = draw_lines(img, lanes, [0, 255, 0], thickness=10)
    if show:
        cv2.imshow('image lanes', lane_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_lanes'), lane_img)

    # overlay lanes on original image
    overlay_img = cv2.addWeighted(original_img, 1, lane_img, 0.8, 1)
    if show:
        cv2.imshow('image overlay', overlay_img)
        cv2.waitKey(0)
    if save:
        cv2.imwrite(set_save_path(name, '_overlay'), overlay_img)

    return overlay_img


# ########## test ##########
if __name__ == '__main__':
    img_path = get_image_path('lane_curve.jpg')
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    image = cv2.imread(img_path)
    image_res = lane_detection(image, rgb=False, show=True, save=True,
                               name=get_image_name(img_path))
    cv2.imshow('image result', image_res)
    if (cv2.waitKey(0) & 0xFF) == 27:
        cv2.destroyAllWindows()
