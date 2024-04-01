### >>> Import ###
import numpy as np
import torch
### <<< Import ###

def eye_pick(x, y, w = 100, h = 50):
    dist_h = (h - abs(y[0]-y[1])) / 2
    dist_w = (w - abs(x[3]-x[2])) / 2
    y_start = int(y[1] - dist_h)
    y_end = int(y[0] + dist_h)
    if x[2] < x[3]:
        x_start = int(x[2] - dist_w)
        x_end = int(x[3] + dist_w)
    if x[3] < x[2]:
        x_start = int(x[3] - dist_w)
        x_end = int(x[2] + dist_w)

    framesize = (y_end - y_start,x_end - x_start)
    return y_start,y_end,x_start,x_end, framesize

def face_pick(x, y, w = 400, h = 80):
    dist_h = h / 2
    dist_w = w / 2
    y_start = int(y[0] - dist_h)
    y_end = int(y[0] + dist_h)
    x_start = int(x[0] - dist_w)
    x_end = int(x[0] + dist_w)

    framesize = (y_end - y_start,x_end - x_start)
    return y_start,y_end,x_start,x_end, framesize


def dict_to_list(data, remove_None = True):
    data = list(data.item().values())
    none_indices = []

    if remove_None == True:
        for index in range(len(data)):
            if data[index] is None:
                none_indices.append(index)

        none_indices.reverse()
        for index in none_indices:
            data.pop(index)

    return data, none_indices

