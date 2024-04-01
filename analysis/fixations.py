### >>> Import ###
import numpy as np
import cv2
import matplotlib.pyplot as plt
### <<< Import ###

### >>> Functions ###
def euclidean_distance(point1, point2):
    euclidean_distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    if np.isnan(euclidean_distance):
        euclidean_distance = np.inf
    return euclidean_distance
def fixation_classifier(gaze_points_sub, distance = 40, saccade_duration = 2):
    fixation_range = [0]
    for i in range(1, len(gaze_points_sub)):
        distance_points = euclidean_distance(gaze_points_sub[i-1], gaze_points_sub[i])
        if distance_points > distance:
            fixation_range.append(i)

    saccade_range = []
    for i in range(1,len(fixation_range)):
        if fixation_range[i] <= fixation_range[i-1] + saccade_duration:
            saccade_range.extend(list(range(fixation_range[i-1], fixation_range[i])))
        else:
            next

    return fixation_range, saccade_range
def fixation_points(gaze_points_sub, distance=40, saccade_duration=2):
    _, saccade_points = fixation_classifier(gaze_points_sub, distance=distance, saccade_duration=saccade_duration)
    for index in reversed(saccade_points):
        del gaze_points_sub[index]
    fixation_boundaries, _ = fixation_classifier(gaze_points_sub, distance=distance, saccade_duration=saccade_duration)

    gaze_point_x = []
    gaze_point_y = []
    for i in range(len(gaze_points_sub)):
        gaze_point_x.append(gaze_points_sub[i][0])
        gaze_point_y.append(gaze_points_sub[i][1])

    i = 0
    fixation_points = []
    fixation_length = []
    for i in range(len(fixation_boundaries) - 1):
        x = int(np.mean(gaze_point_x[fixation_boundaries[i]:fixation_boundaries[i + 1]]))
        y = int(np.mean(gaze_point_y[fixation_boundaries[i]:fixation_boundaries[i + 1]]))
        fixation_points.append((x, y))
        fixation_length.append(len(gaze_point_x[fixation_boundaries[i]:fixation_boundaries[i + 1]]))

    return fixation_points, fixation_length
def fixation_map(frame, fixation_points, fixation_length, save = False, path = None, fixation_count = 1, number_shift = 0):
    frame = frame.copy()
    for i in range(len(fixation_points) - 1):
        cv2.line(frame, fixation_points[i], fixation_points[i + 1], (128, 0, 0), 2)

    overlay = frame.copy()

    for i, point in enumerate(fixation_points):
        x, y = point
        if i == 0:
            color = (0,255,0)
        else:
            color = (255,0,0)
        cv2.circle(overlay, (x, y), 20 + 2 * fixation_length[i], color, -1)
        cv2.circle(overlay, (x, y), 20 + 2 * fixation_length[i], (0, 0, 0), 2)

        text_size = cv2.getTextSize(str(i + 1), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(overlay, str(i + number_shift), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    plt.imshow(frame)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Fixationmap ' + str(fixation_count))
    if save == True:
        plt.savefig(f'{path}Fixationmap_{str(fixation_count)}_{name}.png',
                    transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
### <<< Functions ###

### >>> Parameter ###
name = 'ruven'
lecture_path = '../data/lectures/2023-01-26 E3.1 Matheeinführung Summen_eyetracking_p1.mp4'
lecture_path2 = '../data/lectures/2023-01-26 E3.1 Matheeinführung Summen_eyetracking_p2.mp4'
path = '../data/results/analysis/fixations/'
fps = 25

save = True
### <<< Parameter ###

gaze_points = np.load(f'../data/results/gaze_points/{name}_gazepoints_blinkremoved.npy')
# gaze_points = np.load(f'../data/results/gaze_points/{name}_gazepoints.npy')
shift = 3 * fps + 120 * fps + 20 * fps

# Fixation 1 // Formula 1
fixation_count = 1
gaze_points_sub = gaze_points[(4 * 60 * 25 + 56 * 25):(5 * 60 * 25 + 21 * 25)].tolist()
fixation_point, fixation_length = fixation_points(gaze_points_sub)

vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 5*60*25 + 25 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
fixation_map(frame = frame, fixation_points = fixation_point, fixation_length = fixation_length, fixation_count = fixation_count, save = save, path = path)

# Fixation 2 // Formula 1
fixation_count = 2
gaze_points_sub = gaze_points[(5 * 60 * 25 + 30 * 25):(5 * 60 * 25 + 58 * 25)].tolist()
fixation_point, fixation_length = fixation_points(gaze_points_sub)

vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 6*60*25 + 6 * 25 + 3)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
fixation_map(frame = frame, fixation_points = fixation_point, fixation_length = fixation_length, fixation_count = fixation_count, save = save, path = path)

# Fixation 2.5 // Formula 2 left out

# Fixation 3 // Hint 1
fixation_count = 3
gaze_points_sub = gaze_points[(9 * 60 * 25 + 18 * 25):(9 * 60 * 25 + 32 * 25)]
gaze_points_sub = gaze_points_sub.tolist()
fixation_point, fixation_length = fixation_points(gaze_points_sub)
vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 9*60*25 + 22 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
fixation_map(frame = frame, fixation_points = fixation_point, fixation_length = fixation_length, fixation_count = fixation_count, save = save, path = path)

# Fixation 4 // Formula 1
fixation_count = 4
gaze_points_sub = gaze_points[(11 * 60 * 25 + 0 * 25 + 8):(11 * 60 * 25 + 19 * 25)].tolist()
fixation_point, fixation_length = fixation_points(gaze_points_sub)

vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 11*60*25 + 20 * 25+5)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
fixation_map(frame = frame, fixation_points = fixation_point, fixation_length = fixation_length, fixation_count = fixation_count, save = save, path = path)

# Fixation 5 // Hint 2
fixation_count = 5
gaze_points_sub = gaze_points[(13 * 60 * 25 + 43 * 25):(13 * 60 * 25 + 56 * 25)].tolist()
fixation_point, fixation_length = fixation_points(gaze_points_sub)
vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 13*60*25 + 48 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
fixation_map(frame = frame, fixation_points = fixation_point, fixation_length = fixation_length, fixation_count = fixation_count, save = save, path = path)

### 2. Part ###
cap = cv2.VideoCapture(f'../data/lectures_webcam/{name}_p2.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
gaze_points = gaze_points[-length:]

shift = 3 * fps + 60 * fps + 20 * fps

# Fixation 6 //
fixation_count = 6
gaze_points_sub = gaze_points[(11 * 60 * 25 + 47 * 25):(13 * 60 * 25 + 13 * 25)].tolist()
fixation_point, fixation_length = fixation_points(gaze_points_sub)
vid = cv2.VideoCapture(lecture_path2)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 13*60*25 + 2 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
fixation_map(frame = frame, fixation_points = fixation_point, fixation_length = fixation_length, fixation_count = fixation_count, save = save, path = path)

# Fixation 7 //
fixation_count = 7
gaze_points_sub = gaze_points[(17 * 60 * 25 + 41 * 25):(17 * 60 * 25 + 58 * 25)].tolist()
fixation_point, fixation_length = fixation_points(gaze_points_sub)
vid = cv2.VideoCapture(lecture_path2)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 17*60*25 + 58 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
fixation_map(frame = frame, fixation_points = fixation_point, fixation_length = fixation_length, fixation_count = fixation_count, save = save, path = path)
