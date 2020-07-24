# Lane Detection

## Objective

#### This is the basic method for lane detection based on OpenCV.

---

## Dependencies & Enviroment

* Python 3.6, OpenCV 4.2
* OS: Ubuntu 16.04

---

## Full processes for lane detection

1. Read image.
![alt](./result/detail/lane_straight_original.jpg)

2. Convert original image to grayscale.
![alt](./result/detail/lane_straight_gray.jpg)

3. Darken the grayscale image.
![alt](./result/detail/lane_straight_dark.jpg)

4. Convert original image to HLS color space.
![alt](./result/detail/lane_straight_HLS.jpg)

5. Isolate white from HLS to get white mask.
![alt](./result/detail/lane_straight_white_mask.jpg)

6. Isolate yellow from HLS to get yellow mask.
![alt](./result/detail/lane_straight_yellow_mask.jpg)

7. Combine white mask and yellow mask.
![alt](./result/detail/lane_straight_color_mask.jpg)

8. Apply Gaussian Blur.
![alt](./result/detail/lane_straight_blur.jpg)

9. Get edges with Canny Edge Detector.
![alt](./result/detail/lane_straight_canny.jpg)

10. Segment region of interest.
![alt](./result/detail/lane_straight_roi.jpg)

11. Retrieve Hough lines with cv2.HoughLinesP.
![alt](./result/detail/lane_straight_hough_lines.jpg)

12. Calculate left and right lane from Hough lines and draw them.
![alt](./result/detail/lane_straight_lanes.jpg)

13. Overlay lanes on original image.
![alt](./result/detail/lane_straight_overlay.jpg)

---

## Shortcomings

* Calibration and undistortion are not involved in this project.

* Hough Lines based on straight lines do not work very good for curved lane.
![alt](./result/lane_curve_res.jpg)

---

## Future improvements

* Instead of Hough Lines, it would be beneficial to use Curve Fitting to detect curved lane.