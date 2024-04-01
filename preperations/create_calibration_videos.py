### >>> Import ###
import cv2
import numpy as np
import time

from utils.get_points import *
### <<< Import ###

### >>> Parameter ###
framesize = (1920,1080)
fps = 25

train = True
eval = True
calibrate = True

save = True
show = False
### >>> Parameter ###

### Code ###
if show == True:
    cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

if train == True:
    mov_name = f'../data/assets/points_collection_{fps}fps.mp4'
    duration = 120
    points, _, _ = get_points_train(time_corner = duration - 13*6,
                               time_wave = 13,
                               fps = fps,
                               w = 1920,
                               h = 1080,
                               number_of_fixpoints = 12,
                               stay_time_corner = 2,
                               stay_time_wave = 1,
                               distance_w=(1920 - 60) / 2,
                               distance_h=(1080 - 60) / 3,
                               random_seed = 123)

    if save == True:
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        out = cv2.VideoWriter(mov_name, fourcc, fps, framesize)

    index = 0

    while True:
        frame = np.zeros((1080, 1920, 3), np.uint8)

        frequency = 0.97
        circle_size = int(9 + 5 * np.sin(2 * np.pi * frequency * index) + 5)

        frame = cv2.circle(frame,
                           points[index],  # middlepoint
                           circle_size,  # Radius
                           (0, 0, 255),  # BGR color
                           -1)  # Fill figure

        if save == True:
            out.write(frame)

        index += 1
        if index == (duration * fps):
            break

        if show == True:
            cv2.imshow('window', frame)
            if cv2.waitKey(int(1000 / fps)) == ord('q'):
                break

    if save == True:
        out.release()
    cv2.destroyAllWindows()

    time.sleep(3)

    if save == True:
        cap = cv2.VideoCapture(mov_name)
        print(
            f'Saved movie has {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames \n Points vector has {len(points)} points')
        cap.release()
    else:
        print('No video saved | Change parameter save, if the video should be saved')

if eval == True:
    mov_name = f'../data/assets/points_collection_{fps}fps_eval.mp4'
    duration = 20
    points, _, _ = get_points_eval(time=duration, fps=fps, stay_time=2)

    if save == True:
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        out = cv2.VideoWriter(mov_name, fourcc, fps, framesize)

    index = 0

    while True:
        frame = np.zeros((1080, 1920, 3), np.uint8)

        frequency = 0.97
        circle_size = int(9 + 5 * np.sin(2 * np.pi * frequency * index) + 5)

        frame = cv2.circle(frame,
                           points[index],  # middlepoint
                           circle_size,  # Radius
                           (0, 0, 255),  # BGR color
                           -1)  # Fill figure

        if save == True:
            out.write(frame)

        index += 1
        if index == (duration * fps):
            break

        if show == True:
            cv2.imshow('window', frame)
            if cv2.waitKey(int(1000 / fps)) == ord('q'):
                break

    if save == True:
        out.release()
    cv2.destroyAllWindows()

    time.sleep(3)

    if save == True:
        cap = cv2.VideoCapture(mov_name)
        print(
            f'Saved movie has {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames \n Points vector has {len(points)} points')
        cap.release()
    else:
        print('No video saved | Change parameter save, if the video should be saved')

if calibrate == True:
    mov_name = f'../data/assets/points_collection_{fps}fps_calibrate.mp4'
    duration = 60
    points, _, _ = get_points_calibrate(time_corner = duration - 15*2,
                               time_wave = 15,
                               fps = fps,
                               w = 1920,
                               h = 1080,
                               number_of_fixpoints = 12,
                               stay_time_corner = 2,
                               stay_time_wave = 1,
                               distance_w=(1920 - 60),
                               distance_h=(1080 - 60) / 2,
                               random_seed = 456)

    if save == True:
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        out = cv2.VideoWriter(mov_name, fourcc, fps, framesize)

    index = 0

    while True:
        frame = np.zeros((1080, 1920, 3), np.uint8)

        frequency = 0.97
        circle_size = int(9 + 5 * np.sin(2 * np.pi * frequency * index) + 5)

        frame = cv2.circle(frame,
                           points[index],  # middlepoint
                           circle_size,  # Radius
                           (0, 0, 255),  # BGR color
                           -1)  # Fill figure

        if save == True:
            out.write(frame)

        index += 1
        if index == (duration * fps):
            break

        if show == True:
            cv2.imshow('window', frame)
            if cv2.waitKey(int(1000 / fps)) == ord('q'):
                break

    if save == True:
        out.release()
    cv2.destroyAllWindows()

    time.sleep(3)

    if save == True:
        cap = cv2.VideoCapture(mov_name)
        print(
            f'Saved movie has {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames \n Points vector has {len(points)} points')
        cap.release()
    else:
        print('No video saved | Change parameter save, if the video should be saved')




