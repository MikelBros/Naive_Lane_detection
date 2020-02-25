import numpy as np
import cv2
from shapely.geometry import LineString


class Line(LineString):

    def line_slope(self, line):
        """
        Calculate slope of line
        """
        x1 = np.float(line.coords[0][0])
        y1 = np.float(line.coords[0][1])
        x2 = np.float(line.coords[1][0])
        y2 = np.float(line.coords[1][1])
        slope = (y2 - y1) / (x2 - x1 + np.finfo(float).eps)
        return slope

    def line_bias(self, line):
        """
        Calculate slope of line
        """
        x1 = np.float(line.coords[0][0])
        y1 = np.float(line.coords[0][1])
        bias = x1 - self.line_slope * y1
        return bias


def compute_lane_from_candidates(line_candidates, img_shape):
    """
    Compute lines that approximate the position of both road lanes.

    :param line_candidates: lines from hough transform
    :param img_shape: shape of image to which hough transform was applied
    :return: lines that approximate left and right lane position
    """

    # separate candidate lines according to their slope
    pos_lines = [l for l in line_candidates if l.line_slope(l) > 0]
    neg_lines = [l for l in line_candidates if l.line_slope(l) < 0]

    # interpolate biases and slopes to compute equbation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.median([np.float(l.coords[0][1]) -
                          ((np.float(l.coords[1][1]) - np.float(l.coords[0][1])) /
                         (np.float(l.coords[1][0]) - np.float(l.coords[0][0]) +
                         np.finfo(float).eps)) * np.float(l.coords[0][0]) for l in neg_lines]).astype(int)
    neg_slope = np.median([((np.float(l.coords[1][1]) - np.float(l.coords[0][1])) /
                         (np.float(l.coords[1][0]) - np.float(l.coords[0][0]) +
                         np.finfo(float).eps)) for l in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = Line([(x1, y1), (x2, y2)])

    # interpolate biases and slopes to compute equation of line that approximates right lane
    # median is employed to filter outliers
    lane_right_bias = np.median([np.float(l.coords[0][1]) -
                          ((np.float(l.coords[1][1]) - np.float(l.coords[0][1])) /
                         (np.float(l.coords[1][0]) - np.float(l.coords[0][0]) +
                         np.finfo(float).eps)) * np.float(l.coords[0][0]) for l in pos_lines]).astype(int)
    lane_right_slope = np.median([((np.float(l.coords[1][1]) - np.float(l.coords[0][1])) /
                         (np.float(l.coords[1][0]) - np.float(l.coords[0][0]) +
                         np.finfo(float).eps)) for l in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
    right_lane = Line([(x1, y1), (x2, y2)])

    return left_lane, right_lane


def filter_candidates(hough_lines):
    """
    If the slope of the lines are within our margins and inte the lower half of the image
    we store it in a list
    """
    line_cadidates = []
    for line in hough_lines:
        slope = line.line_slope(line)
        if 0.5 <= np.abs(slope) <= 1 and line.coords[0][1] > 300:
            line_cadidates.append(line)
    return line_cadidates


def detect_lanes(frames):
    """
    Entry point for lane detection pipeline. Takes as input a list of frames (RGB) and returns an image (RGB)
    with overlaid the inferred road lanes. Eventually, len(frames)==1 in the case of a single image.
    """

    lane_lines = []
    for t in range(0, len(frames)):
        color_image = frames[t]

        # convert to grayscale
        img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # perform gaussian blur to smooth the image in order to remove the noise
        img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)

        # perform edge detection
        img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)

        # perform hough line transform, gives (x1, y1, x2, y2) tuples
        hough_lines = cv2.HoughLinesP(img_edge,
                                      rho=2,
                                      theta=np.pi / 180,
                                      threshold=1,
                                      minLineLength=15,
                                      maxLineGap=5)

        # create line classes out of the lines found in the previous step
        detected_lines = [Line([(l[0][0], l[0][1]), (l[0][2], l[0][3])]) for l in hough_lines]

        line_candidates = filter_candidates(detected_lines)

        for line in line_candidates:
            cv2.line(img_edge,
                     (int(line.coords[0][0]), int(line.coords[0][1])),
                     (int(line.coords[1][0]), int(line.coords[1][1])),
                     [255, 0, 0],
                     10)

        lane_lines = compute_lane_from_candidates(line_candidates, img_gray.shape)
        

    return lane_lines
