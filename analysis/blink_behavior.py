### >>> Import ###
import numpy as np
from torch.optim import Adam
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
### <<< Import ###

### >>> Functions ###
def blink_detection(closed_eyes_points, max_duration = 4):
    blink_list = []
    skip_next = False
    max_duration_count = 0
    stopping_criteria = max_duration + 1

    for value in closed_eyes_points:
        if value == 0 and not skip_next:
            blink_list.append(0)
            skip_next = True
            max_duration_count += 1
        elif value == 0 and skip_next:
            blink_list.append(1)
            max_duration_count += 1
        if value != 0:
            blink_list.append(1)
            skip_next = False
            max_duration_count = 0
        if max_duration_count == stopping_criteria:
            blink_list[-stopping_criteria:] = [1] * stopping_criteria

    blink_list = np.array(blink_list)

    return blink_list
def blinks_per_min_plot(blink_data, fps = 25, save = False, path = None, name = None):
    blinks = blink_data
    minutes = len(blinks) / (fps * 60)

    blinks_per_minute = np.zeros(int(np.ceil(minutes)))

    for blink_number, blink in enumerate(blinks):
        minute_index = int(blink_number / (fps * 60))
        if blink == 0:
            blinks_per_minute[minute_index] += 1

    plt.plot(blinks_per_minute, color='blue', linestyle='-', marker='o', markersize=5)
    plt.xlabel('time (in min)')
    plt.ylabel('number of blinks')
    plt.title('Number of blinks per minute')

    plt.axvline(x=0.47, color='green', linestyle='--', label='Lecture slide', alpha = 0.5)
    plt.axvline(x=0.98, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=2.7, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=4.17, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=9.75, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=11.67, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=11.77, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=16.16, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=17.42, color='#FF00FF', linestyle='--', label= 'R-Code', alpha = 0.5)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    if save == True:
        plt.savefig(f'{path}Blink_detection_figure1_{name}.png',
                    transparent=True, bbox_inches='tight', pad_inches=0)
        print(f'Sucessfully saved')
    plt.show()

    return blinks_per_minute
def blink_duration(blink_data, closed_eyes_data,  blink_indices = None):
    closed_eyes = closed_eyes_data
    if blink_data is not None:
        blinks = blink_data
        blink_indices = np.where(blinks == 0)[0]
    if blink_indices is not None:
        blink_indices = blink_indices
    blink_duration = []
    for i in blink_indices:
        j = i
        closed_eyes_counter = 0
        while True:
            if closed_eyes[j] == 0:
                closed_eyes_counter += 1
                j += 1
            elif closed_eyes[j] == 1:
                blink_duration.append(closed_eyes_counter)
                break

    blink_duration = np.array(blink_duration)

    return blink_duration, blink_indices
def blinks_duration_per_min_plot(blinks, blink_duration, blink_indices, fps=25, show_off_screen = False, save = False, path = None, name = None):
    minutes = len(blinks) / (fps * 60)
    total_blink_duration = np.zeros(int(np.ceil(minutes)))

    for idx, blink_index in enumerate(blink_indices):
        minute_index = int(blink_index / (fps * 60))
        total_blink_duration[minute_index] += blink_duration[idx] * int((1000 / fps))

    if show_off_screen == False:
        plt.plot(total_blink_duration, color='blue', linestyle='-', marker='o', markersize=5)
        plt.xlabel('time (in min)')
        plt.ylabel('blink duration (in ms)')
        plt.title('Blink duration per minute')
    if show_off_screen == True:
        plt.plot(total_blink_duration, color='blue', linestyle='-', marker='o', markersize=5)
        plt.xlabel('lecture time (in min)')
        plt.ylabel('off-screen time (in ms)')
        plt.title('Off-screen time per minute')

    plt.axvline(x=0.47, color='green', linestyle='--', label='Lecture slide', alpha = 0.5)
    plt.axvline(x=0.98, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=2.7, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=4.17, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=9.75, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=11.67, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=11.77, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=16.16, color='green', linestyle='--', alpha = 0.5)
    plt.axvline(x=17.42, color='#FF00FF', linestyle='--', label= 'R-Code', alpha = 0.5)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')

    if save == True and not show_off_screen:
        plt.savefig(f'{path}Blink_detection_figure2_{name}.png',
                    transparent=True, bbox_inches='tight', pad_inches=0)
        print(f'Sucessfully saved')
    if save == True and show_off_screen:
        plt.savefig(f'{path}Blink_detection_figure3_{name}.png',
                    transparent=True, bbox_inches='tight', pad_inches=0)
        print(f'Sucessfully saved')

    plt.show()

    return total_blink_duration
### <<< Functions ###

### >>> Parameter ###
name = 'ruven'
fps = 25
save_boolean = True
save_path = '../data/results/analysis/blinkbehavior/'
save_name = name
### <<< Parameter ###

# Get closed eyes
closed_eyes = np.load(f'../data/closedEyes/{name}_closed_eyes.npy')

cap = cv2.VideoCapture('../data/lectures_webcam/' + str(name) + '_p1.mp4')
length_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
length = int(length_total - length_total % fps)
cap.release()
cap = cv2.VideoCapture('../data/lectures_webcam/' + str(name) + '_p2.mp4')
length_vid2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(cv2.CAP_PROP_FRAME_COUNT) % fps)
cap.release()

calibration_length = 120 * fps
eval_length = 20 * fps
recalibration_length = 60 * fps
follow_point = 3 * fps

closed_eyes = closed_eyes[calibration_length + eval_length : -eval_length -follow_point]
closed_eyes = np.concatenate((closed_eyes[:length], closed_eyes[-length_vid2:]))

blinks = blink_detection(closed_eyes, max_duration=4)

# Double the last blinks, since the frequence is only 29 seconds
# blinks = np.concatenate((blinks, blinks[-29:]), axis = 0)
# closed_eyes = np.concatenate((closed_eyes, closed_eyes[-29:]), axis = 0)

# Blinks per min
blinks_per_min_plot(blink_data=blinks, save = save_boolean, path= save_path, name=save_name)

# Blink-duration
blink_duration_data, blink_indices = blink_duration(blink_data=blinks,closed_eyes_data=closed_eyes)
blinks_duration_per_min_plot(blinks = blinks,blink_duration= blink_duration_data,blink_indices= blink_indices, save = save_boolean, path= save_path, name=save_name)

# Off-screen-time
all_eye_closings = blink_detection(closed_eyes,max_duration=1000)
all_eye_closings = np.where(all_eye_closings == 0)[0]
blinks_indices = np.where(blinks == 0)[0]
off_screen = np.setdiff1d(all_eye_closings, blink_indices)
off_screen_duration_data, off_screen_indices = blink_duration(blink_data=None,closed_eyes_data=closed_eyes, blink_indices=off_screen)
blinks_duration_per_min_plot(blinks = blinks,blink_duration= off_screen_duration_data,blink_indices= off_screen_indices, show_off_screen= True, save = save_boolean, path= save_path, name=save_name)


