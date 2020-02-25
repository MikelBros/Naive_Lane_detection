import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from os.path import join, basename
from collections import deque
from lane_detection import detect_lanes


def plot_straight_lanes(lanes, image):
    intersection_point = lanes[0].intersection(lanes[1])
    cv2.line(in_image,
             (int(lanes[0].coords[0][0]), int(lanes[0].coords[0][1])),
             (int(intersection_point.coords[0][0]), int(intersection_point.coords[0][1])),
             [128, 0, 0],
             10)
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.line(in_image,
             (int(lanes[1].coords[1][0]), int(lanes[1].coords[1][1])),
             (int(intersection_point.coords[0][0]), int(intersection_point.coords[0][1])),
             [128, 0, 0],
             10)
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def plot_segmented_lane(lanes, image):
    intersection_point = lanes[0].intersection(lanes[1])
    contours = np.array([[lanes[0].coords[0][0], lanes[0].coords[0][1]],
                         [lanes[1].coords[1][0], lanes[1].coords[1][1]],
                         [intersection_point.coords[0][0], intersection_point.coords[0][1]]])
    cv2.fillPoly(image, pts=np.int32([contours]), color=(255, 255, 255))


if __name__ == '__main__':

    resize_h, resize_w = 540, 960

    verbose = True
    if verbose:
        plt.ion()
        figManager = plt.get_current_fig_manager()

    # test on images
    test_images_dir = join('data', 'test_images')
    test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]

    for test_img in test_images:

        print('Processing image: {}'.format(test_img))

        out_path = join('out', 'images', basename(test_img))
        in_image = cv2.cvtColor(cv2.imread(test_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        detected_lanes = detect_lanes([in_image])
        plot_segmented_lane(detected_lanes, in_image)
        plot_straight_lanes(detected_lanes, in_image)
        if verbose:
            plt.imshow(in_image)
            plt.waitforbuttonpress()
    plt.close('all')

    # test on videos
    test_videos_dir = join('data', 'test_videos')
    test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]

    for test_video in test_videos:

        print('Processing video: {}'.format(test_video))

        cap = cv2.VideoCapture(test_video)
        out = cv2.VideoWriter(join('out', 'videos', basename(test_video)),
                              fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                              fps=20.0, frameSize=(resize_w, resize_h))

        frame_buffer = deque(maxlen=10)
        while cap.isOpened():
            ret, color_frame = cap.read()
            if ret:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                #color_frame = cv2.resize(color_frame, (resize_w, resize_h))
                #frame_buffer.append(color_frame)
                detected_lanes = detect_lanes([color_frame])
                plot_segmented_lane(detected_lanes, color_frame)
                plot_straight_lanes(detected_lanes, color_frame)
                out.write(cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR))
                cv2.imshow('blend', cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
