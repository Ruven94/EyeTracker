### >>> Import ###
import numpy as np
import random
### <<< Import ###

### Code ###
def wave_points(x,y,
                number_of_fixpoints = 12,
                time = 13,
                fps = 25,
                stay_time = 1,
                distance_w= (1920 - 60) / 2,
                distance_h = (1080 - 60) / 3):
    fixpoints = []
    fixpoints.append(np.array([x, y]))

    distance_points_w = distance_w / (number_of_fixpoints - 1)
    for i in range(0, number_of_fixpoints - 1):
        fixpoints.append(np.array(fixpoints[i] + [distance_points_w, (-1) ** i * distance_h]))

    path_length = time * fps - stay_time * fps
    points = []

    # Wait "stay_time" seconds for transition
    points.extend(np.linspace(fixpoints[0], fixpoints[0], fps * stay_time))

    for i in range(0, number_of_fixpoints - 1):
        points.extend(np.linspace(fixpoints[i], fixpoints[i + 1], int(path_length / (number_of_fixpoints - 1))))

    points = [(int(x), int(y)) for x, y in points]
    points = [tuple(point) for point in points]

    while (len(points) < (time * fps)):
        points.append(points[-1])

    return points

def corner_points(time = 42, fps = 25, w = 1920, h = 1080, stay_time = 2):
    # Input: time in seconds
    frames = time * fps  # return total frames
    points = []

    # Define corners
    corners = []
    corners.append(np.array([30, 30]))
    corners.append(np.array([w - 30, 30]))
    corners.append(np.array([w - 30, h - 30]))
    corners.append(np.array([30, h - 30]))

    # Path around corners
    path_length = time * fps - stay_time * fps
    total_distance = 0
    for i in range(len(corners) - 1):
        total_distance += np.linalg.norm(corners[i + 1] - corners[i])
    total_distance += np.linalg.norm(corners[len(corners) - 1] - corners[0])
    # Sum of the distances between all corner points

    # Wait 'stay_time' seconds for transition
    points.extend(np.linspace(corners[0], corners[0], fps * stay_time))

    for i in range(0, len(corners) - 1):
        points.extend(np.linspace(corners[i], corners[i + 1],
                                  int(round(np.linalg.norm(corners[i] - corners[i + 1]) / total_distance * path_length))))
    points.extend(np.linspace(corners[-1], corners[0],
                              int(round(np.linalg.norm(corners[0] - corners[-1]) / total_distance * path_length))))

    points = [(int(x), int(y)) for x, y in points]
    points = [tuple(point) for point in points]

    while (len(points) < (time * fps)):
        points.append(points[-1])

    return points

def get_points_eval(time = 20, fps = 25, w = 1920, h = 1080, stay_time = 2):
    points = []

    # Define corners
    corners = []
    corners.append(np.array([30, 30]))
    corners.append(np.array([w - 30, 30]))
    corners.append(np.array([w - 30, h - 30]))
    corners.append(np.array([30, h - 30]))

    path_length = time * fps - 2 * fps * stay_time

    points.extend(np.linspace(corners[0], corners[0],fps * stay_time)) # Stay for finding the point
    stay_points_len1 = len(points)
    points.extend(np.linspace(corners[0], corners[2],int(path_length/2)))
    stay_points_len2 = len(points)
    points.extend(np.linspace(corners[3], corners[3],fps * stay_time)) # Stay for finding the point
    stay_points_len3 = len(points)
    points.extend(np.linspace(corners[3], corners[1],int(path_length/2)))
    stay_points_len4 = len(points)

    points = [(int(x), int(y)) for x, y in points]
    points = [tuple(point) for point in points]

    while (len(points) < (time * fps)):
        points.append(points[-1])

    stay_indices = list(range(0,stay_points_len1)) + list(range(stay_points_len2,stay_points_len3))
    filtered_points = [point for i, point in enumerate(points) if i not in stay_indices]

    return points, stay_indices, filtered_points


def get_points_train(time_corner=120 - 13 * 6,
                      time_wave=13,
                      fps=25,
                      w=1920,
                      h=1080,
                      number_of_fixpoints=12,
                      stay_time_corner=2,
                      stay_time_wave=1,
                      distance_w=(1920 - 60) / 2,
                      distance_h=(1080 - 60) / 3,
                      random_seed=123):

    stay_points_len = []
    stay_points_len.append(0)

    points = corner_points(time=time_corner, fps=fps, w=w, h=h, stay_time=stay_time_corner)
    stay_points_len.append(len(points))

    if random_seed is not None:
        random.seed(random_seed)

    wave_points_list = [
        wave_points(x=30, y=30, time=time_wave, number_of_fixpoints=number_of_fixpoints,
                    fps=fps, stay_time=stay_time_wave, distance_w=distance_w, distance_h=distance_h),
        wave_points(x=w / 2, y=30, time=time_wave, number_of_fixpoints=number_of_fixpoints,
                    fps=fps, stay_time=stay_time_wave, distance_w=distance_w, distance_h=distance_h),
        wave_points(x=30, y=(h - 60) / 3 + 30, time=time_wave, number_of_fixpoints=number_of_fixpoints,
                    fps=fps, stay_time=stay_time_wave, distance_w=distance_w, distance_h=distance_h),
        wave_points(x=w / 2, y=(h - 60) / 3 + 30, time=time_wave, number_of_fixpoints=number_of_fixpoints,
                    fps=fps, stay_time=stay_time_wave, distance_w=distance_w, distance_h=distance_h),
        wave_points(x=30, y=(h - 60) / 3 * 2 + 30, time=time_wave, number_of_fixpoints=number_of_fixpoints,
                    fps=fps, stay_time=stay_time_wave, distance_w=distance_w, distance_h=distance_h),
        wave_points(x=w / 2, y=(h - 60) / 3 * 2 + 30, time=time_wave, number_of_fixpoints=number_of_fixpoints,
                    fps=fps, stay_time=stay_time_wave, distance_w=distance_w, distance_h=distance_h)
    ]

    # Mische die Wellenpunkte in zufälliger Reihenfolge
    random.shuffle(wave_points_list)

    # Füge die Wellenpunkte in zufälliger Reihenfolge an 'points' an
    for wave_points_data in wave_points_list:
        points.extend(wave_points_data)
        stay_points_len.append(len(points))

    stay_indices = list(range(stay_points_len[0], stay_points_len[0] + stay_time_corner * fps))
    for i in range(1,len(stay_points_len) - 1):
        stay_indices = stay_indices + list(range(stay_points_len[i], stay_points_len[i] + stay_time_wave * fps))

    filtered_points = [point for i, point in enumerate(points) if i not in stay_indices]

    return points, stay_indices, filtered_points

def get_points_calibrate(time_corner=60 - 15*2,
                      time_wave=15,
                      fps=25,
                      w=1920,
                      h=1080,
                      number_of_fixpoints=12,
                      stay_time_corner=2,
                      stay_time_wave=1,
                      distance_w=(1920 - 60),
                      distance_h=(1080 - 60) / 2,
                      random_seed=456):

    stay_points_len = []
    stay_points_len.append(0)

    points = corner_points(time=time_corner, fps=fps, w=w, h=h, stay_time=stay_time_corner)
    stay_points_len.append(len(points))

    if random_seed is not None:
        random.seed(random_seed)

    wave_points_list = [
        wave_points(x= 30, y= 30, time=time_wave, number_of_fixpoints=number_of_fixpoints,
                    fps=fps, stay_time=stay_time_wave, distance_w=distance_w, distance_h=distance_h),
        wave_points(x= 30, y= h/2, time=time_wave, number_of_fixpoints=number_of_fixpoints,
                    fps=fps, stay_time=stay_time_wave, distance_w=distance_w, distance_h=distance_h),
    ]

    # Mische die Wellenpunkte in zufälliger Reihenfolge
    random.shuffle(wave_points_list)

    # Füge die Wellenpunkte in zufälliger Reihenfolge an 'points' an
    for wave_points_data in wave_points_list:
        points.extend(wave_points_data)
        stay_points_len.append(len(points))

    stay_indices = list(range(stay_points_len[0], stay_points_len[0] + stay_time_corner * fps))
    for i in range(1, len(stay_points_len) - 1):
        stay_indices = stay_indices + list(range(stay_points_len[i], stay_points_len[i] + stay_time_wave * fps))

    filtered_points = [point for i, point in enumerate(points) if i not in stay_indices]

    return points, stay_indices, filtered_points
