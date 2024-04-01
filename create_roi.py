### >>> Import ###
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os

from utils.helper_functions import *
from utils.pre_processing import *
from utils.get_points import *
### <<< Import ###

### >>> Parameter ###
name = 'ruven'
calibration = True
eval = True
recalibration = True
lecture = False
closed_eye_detection = False

omit_wait_points = True
hist_equal = True
augmentation = True
augmentation_count = 5
splits = 9
'''
The split parameter can be utilized to partition the lecture into individual smaller segments during computation. 
This is particularly advantageous if the available memory is a limiting factor. 
The input can vary between 1 and 9 splits.
'''

show = False
save = True
### <<< Parameter ###

### Code ###
# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence= 0.01, min_tracking_confidence= 0.01, max_num_faces=1) # activate if no face is detected

### Define combinations based on parameter ###
parameter_set = []
if calibration == True:
    parameter = {'calibration': True, 'eval': False, 'recalibration': False, 'lecture': False, 'lecture_part': 0, 'ClosedEyes': False, 'augmentation': augmentation}
    parameter_set.append(parameter)
if eval == True:
    for i in range(1,4):
        parameter = {'calibration': False, 'eval': True, 'recalibration': False, 'lecture': False, 'lecture_part': 0, 'part': i, 'ClosedEyes': False, 'augmentation': False}
        parameter_set.append(parameter)
if recalibration == True:
    parameter = {'calibration': False, 'eval': False, 'recalibration': True, 'lecture': False, 'lecture_part': 0, 'ClosedEyes': False, 'augmentation': augmentation}
    parameter_set.append(parameter)
if lecture == True:
    for j in range(1,3):
        for i in range(1,splits + 1):
            parameter = {'calibration': False, 'eval': False, 'recalibration': False, 'lecture': True, 'lecture_part': j, 'part': i, 'ClosedEyes': False, 'augmentation': False}
            parameter_set.append(parameter)
if closed_eye_detection == True:
    for j in range(1,3):
        for i in range(1,splits + 1):
            parameter = {'calibration': False, 'eval': False, 'recalibration': False, 'lecture': False, 'lecture_part': j, 'part': i, 'ClosedEyes': True, 'augmentation': False}
            parameter_set.append(parameter)

for params in parameter_set:
    # # Histogram
    if params['calibration'] == True:
        cap = cv2.VideoCapture('data/calibration/' + str(name) + '.mp4')
        points, stay_indices, _ = get_points_train()
        if omit_wait_points == False:
            stay_indices = []
    if params['eval'] == True:
        cap = cv2.VideoCapture('data/evaluation/' + str(name) + '_' + str(params['part']) +'.mp4')
        points, stay_indices, _ = get_points_eval()
        if omit_wait_points == False:
            stay_indices = []
    if params['recalibration'] == True:
        cap = cv2.VideoCapture('data/recalibration/' + str(name) + '.mp4')
        points, stay_indices, omit_calib_points = get_points_calibrate()
        if omit_wait_points == False:
            stay_indices = []
    if params['lecture'] == True:
        cap = cv2.VideoCapture('data/lectures_webcam/' + str(name) + '_p' + str(params['lecture_part']) + '.mp4')
        stay_indices = []
    if params['ClosedEyes'] == True:
        # if '_record' not in name:
        #     name = name.replace(str(name), str(name) + '_record')
        cap = cv2.VideoCapture('data/closedEyes/' + str(name) + '_p' + str(params['lecture_part']) + '.mp4')
        stay_indices = []

    if params['calibration'] == True:
        print(f'>>> ROI process started for: calibration <<<')
    if params['eval'] == True and params['part'] == 1:
        print(f'>>> ROI process started for: evaluation <<<')
    if params['recalibration'] == True:
        print(f'>>> ROI process started for: recalibration <<<')
    if params['lecture'] == True and params['part'] == 1:
        print('>>> ROI process started for: lecture part ' + str(params['lecture_part']) + ' <<<')
    if params['ClosedEyes'] == True and params['part'] == 1:
        print('>>> ROI process started for: closed_eyes | lecture part ' + str(params['lecture_part']) + ' <<<')

    ### Setup ###
    ret = True
    image_list = []
    data_right = []
    data_left = []
    data_region = []
    augmented_left = []
    augmented_right = []
    augmented_region = []
    breaks = 0

    ### Reading Images ###
    if params['lecture'] == True or params['ClosedEyes'] == True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / splits * (params['part'] - 1)))
        stop_criteria = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / splits * (params['part']) - int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / splits * (params['part'] - 1)))
        for _ in tqdm(range(stop_criteria), desc= 'Reading Images'):
            ret, image = cap.read()
            if ret == False:
                break
            image_list.append(image)
    else:
        progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Reading Images")
        while ret == True:
            ret, image = cap.read()
            if ret == False:
                break
            image_list.append(image)
            progress_bar.update(1)
        progress_bar.close()

    if params['calibration'] == True or params['eval'] == True or params['recalibration'] == True:
        image_list = image_list[:len(points)]
    height, width = (1080,1920)

    if params['ClosedEyes'] != True:
        ### Face recognition ###
        for image in image_list:
            # Face landmarks
            result = face_mesh.process(image)

            ### No Face ###
            if result.multi_face_landmarks == None:
                breaks += 1
                data_right.append(data_right[-1])
                data_left.append(data_left[-1])
                data_region.append(data_region[-1])
                continue
                # Continue if no face detected and also adding 1 to breaks

            ### Left eye ###
            left_eye = [253, 257, 359, 463]
            for facial_landmarks in result.multi_face_landmarks:
                x_left = []
                y_left = []
                for i in left_eye:
                    pt = facial_landmarks.landmark[i]
                    x_left.append(int(pt.x * width))
                    y_left.append(int(pt.y * height))
                if params['augmentation'] == True:
                    y_start, y_end, x_start, x_end, framesize = eye_pick(x_left, y_left, w=130, h=70)
                else:
                    y_start, y_end, x_start, x_end, framesize = eye_pick(x_left, y_left, w=100, h=50)
                image_left = image[y_start:y_end, x_start:x_end]
                data_left.append(image_left)

            ### Right eye ###
            right_eye = [23, 27, 130, 243]
            for facial_landmarks in result.multi_face_landmarks:
                x_right = []
                y_right = []
                for i in right_eye:
                    pt = facial_landmarks.landmark[i]
                    x_right.append(int(pt.x * width))
                    y_right.append(int(pt.y * height))
                if params['augmentation'] == True:
                    y_start, y_end, x_start, x_end, framesize = eye_pick(x_right, y_right, w=130, h=70)
                else:
                    y_start, y_end, x_start, x_end, framesize = eye_pick(x_right, y_right, w=100, h=50)
                image_right = image[y_start:y_end, x_start:x_end]
                data_right.append(image_right)

            ### Eye Region ###
            face = [168]
            for facial_landmarks in result.multi_face_landmarks:
                x_eyeregion = []
                y_eyeregion = []
                for i in face:
                    pt = facial_landmarks.landmark[i]
                    x_eyeregion.append(int(pt.x * width))
                    y_eyeregion.append(int(pt.y * height))
                if params['augmentation'] == True:
                    y_start, y_end, x_start, x_end, framesize = face_pick(x_eyeregion, y_eyeregion, w=500, h=120)
                else:
                    y_start, y_end, x_start, x_end, framesize = face_pick(x_eyeregion, y_eyeregion, w=400, h=80)
                image_region = image[y_start:y_end, x_start:x_end]
                data_region.append(image_region)

            if show == True:
                cv2.imshow('Right eye', image_right)
                cv2.imshow('Left eye', image_left)
                cv2.imshow('Eye region', image_region)
                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    if params['ClosedEyes'] == True:
        ### Face recognition ###
        for image in image_list:
            # Face landmarks
            result = face_mesh.process(image)

            ### No Face ###
            if result.multi_face_landmarks == None:
                breaks += 1
                data_right.append(data_right[-1])
                data_left.append(data_left[-1])
                continue
                # Continue if no face detected and also adding 1 to breaks

            ### Left eye ###
            left_eye = [253, 257, 359, 463]
            for facial_landmarks in result.multi_face_landmarks:
                x_left = []
                y_left = []
                for i in left_eye:
                    pt = facial_landmarks.landmark[i]
                    x_left.append(int(pt.x * width))
                    y_left.append(int(pt.y * height))
                y_start, y_end, x_start, x_end, framesize = eye_pick(x_left, y_left, w=100, h=100)
                image_left = image[y_start:y_end, x_start:x_end]
                data_left.append(image_left)

            ### Right eye ###
            right_eye = [23, 27, 130, 243]
            for facial_landmarks in result.multi_face_landmarks:
                x_right = []
                y_right = []
                for i in right_eye:
                    pt = facial_landmarks.landmark[i]
                    x_right.append(int(pt.x * width))
                    y_right.append(int(pt.y * height))
                y_start, y_end, x_start, x_end, framesize = eye_pick(x_right, y_right, w=100, h=100)
                image_right = image[y_start:y_end, x_start:x_end]
                data_right.append(image_right)

            if show == True:
                cv2.imshow('Right eye', image_right)
                cv2.imshow('Left eye', image_left)
                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    print(f'Face not recognized: {breaks} times')

    ### Omit stay points ###
    if omit_wait_points == True:
        data_left = [value for index, value in enumerate(data_left) if index not in stay_indices]
        data_right = [value for index, value in enumerate(data_right) if index not in stay_indices]
        if params['ClosedEyes'] != True:
            data_region = [value for index, value in enumerate(data_region) if index not in stay_indices]

    ### Histogram equalization ###
    if hist_equal == True:
        data_left_copy = data_left.copy()
        data_left = [histogram_equalization(data_left_copy[i]) for i in range(len(data_left_copy))]
        data_right_copy = data_right.copy()
        data_right = [histogram_equalization(data_right_copy[i]) for i in range(len(data_right_copy))]
        if params['ClosedEyes'] != True:
            data_region_copy = data_region.copy()
            data_region = [histogram_equalization(data_region_copy[i]) for i in range(len(data_region_copy))]

    ### Augmenation ###
    if params['augmentation'] == True:
        ### Left eye ###
        augmented_list = []
        for i in range(len(data_left)):
            augmented_list.append(data_augmentation(data_left[i], rep = augmentation_count,
                                                    rotation=3, scale=0.15,
                                                    height = 50, width = 100, px_h = 20, px_w = 30, seed= 123 + i)[1])
        augmented_list = [list(group) for group in zip(*augmented_list)]
        for i in range(len(augmented_list)):
            augmented_left.extend(augmented_list[i])
        ### Right eye ###
        augmented_list = []
        for i in range(len(data_right)):
            augmented_list.append(data_augmentation(data_right[i], rep = augmentation_count,
                                                    rotation=3, scale=0.15,
                                                    height = 50, width = 100, px_h = 20, px_w = 30, seed= 123 + i)[1])
        augmented_list = [list(group) for group in zip(*augmented_list)]
        for i in range(len(augmented_list)):
            augmented_right.extend(augmented_list[i])
        ### Eye region ###
        augmented_list = []
        for i in range(len(data_region)):
            augmented_list.append(data_augmentation(data_region[i], rep = augmentation_count,
                                                    rotation=5, scale= 0.1,
                                                    height = 80, width = 400, px_h = 40, px_w = 100, seed= 123 + i)[1])
        augmented_list = [list(group) for group in zip(*augmented_list)]
        for i in range(len(augmented_list)):
            augmented_region.extend(augmented_list[i])
        print('Augmentation done')
        augmented_list = []

        data_left = [data_left[i][10:60, 15:115, :] for i in range(len(data_left))]
        data_right = [data_right[i][10:60, 15:115, :] for i in range(len(data_right))]
        data_region = [data_region[i][20:100, 50:450, :] for i in range(len(data_region))]

        data_left.extend(augmented_left)
        data_right.extend(augmented_right)
        data_region.extend(augmented_region)

    if params['ClosedEyes'] == True:
        data_left = [cv2.cvtColor(data_left[i], cv2.COLOR_BGR2GRAY) for i in range(len(data_left))]
        data_right = [cv2.cvtColor(data_right[i], cv2.COLOR_BGR2GRAY) for i in range(len(data_right))]

    if params['augmentation'] == True:
        augmentation_name = '_augmented'
    else:
        augmentation_name = ''

    if save == True:
        if params['calibration'] == True:
            np.save('data/calibration/numpy/' + name + '_lefteye' + augmentation_name, data_left)
            np.save('data/calibration/numpy/' + name + '_righteye' + augmentation_name, data_right)
            np.save('data/calibration/numpy/' + name  + '_eye_region' + augmentation_name, data_region)
            print(f'Calibration frames successfully saved with {len(data_right)} frames for the right eye, {len(data_left)} frames for the left eye and {len(data_region)} frames for the eye region')
        if params['eval'] == True:
            np.save('data/evaluation/numpy/' + name + '_' + str(params['part']) + '_lefteye', data_left)
            np.save('data/evaluation/numpy/' + name + '_' + str(params['part']) + '_righteye', data_right)
            np.save('data/evaluation/numpy/' + name + '_' + str(params['part']) + '_eye_region', data_region)
            print(f'Eval frames successfully saved with {len(data_right)} frames for the right eye, {len(data_left)} frames for the left eye and {len(data_region)} frames for the eye region')
            if params['part'] == 3:
                folder_path = 'data/evaluation/numpy'
                # Left eye
                file_names = os.listdir(folder_path)
                file_names = sorted(file_names)
                loaded_data = []
                if 'glasses' in str(name):
                    for file in file_names:
                        if file.startswith(str(name)) and file.endswith('lefteye.npy'):
                            file_path = os.path.join(folder_path, file)
                            loaded_data.append(np.load(str(file_path)))
                            os.remove(file_path)
                if 'glasses' not in str(name):
                    for file in file_names:
                        if file.startswith(str(name)) and file.endswith('lefteye.npy') and "glasses" not in file:
                            file_path = os.path.join(folder_path, file)
                            loaded_data.append(np.load(str(file_path)))
                            os.remove(file_path)

                data = np.concatenate(loaded_data, axis=0)
                np.save(folder_path + '/' + str(name) + '_lefteye', data)

                # Right eye
                file_names = os.listdir(folder_path)
                file_names = sorted(file_names)
                loaded_data = []
                if 'glasses' in name:
                    for file in file_names:
                        if file.startswith(str(name)) and file.endswith('righteye.npy'):
                            file_path = os.path.join(folder_path, file)
                            loaded_data.append(np.load(str(file_path)))
                            os.remove(file_path)

                if 'glasses' not in name:
                    for file in file_names:
                        if file.startswith(str(name)) and file.endswith('righteye.npy') and "glasses" not in file:
                            file_path = os.path.join(folder_path, file)
                            loaded_data.append(np.load(str(file_path)))
                            os.remove(file_path)

                data = np.concatenate(loaded_data, axis=0)
                np.save(folder_path + '/' + str(name) + '_righteye', data)

                # Eye region
                file_names = os.listdir(folder_path)
                file_names = sorted(file_names)
                loaded_data = []
                if 'glasses' in name:
                    for file in file_names:
                        if file.startswith(str(name)) and file.endswith('eye_region.npy'):
                            file_path = os.path.join(folder_path, file)
                            loaded_data.append(np.load(str(file_path)))
                            os.remove(file_path)
                if 'glasses' not in name:
                    for file in file_names:
                        if file.startswith(str(name)) and file.endswith('eye_region.npy') and "glasses" not in file:
                            file_path = os.path.join(folder_path, file)
                            loaded_data.append(np.load(str(file_path)))
                            os.remove(file_path)

                data = np.concatenate(loaded_data, axis=0)
                np.save(folder_path + '/' + str(name) + '_eye_region', data)
        if params['recalibration'] == True:
            # Augmentation
            np.save('data/recalibration/numpy/' + name + '_lefteye' + augmentation_name, data_left)
            np.save('data/recalibration/numpy/' + name + '_righteye' + augmentation_name, data_right)
            np.save('data/recalibration/numpy/' + name + '_eye_region' + augmentation_name, data_region)
            print(f'Recalibration frames successfully saved with {len(data_right)} frames for the right eye, {len(data_left)} frames for the left eye and {len(data_region)} frames for the eye region')
        if params['lecture'] == True:
            np.save('data/lectures_webcam/numpy/' + name + '_p' + str(params['lecture_part']) + str(params['part']) + '_lefteye', data_left)
            np.save('data/lectures_webcam/numpy/' + name + '_p' + str(params['lecture_part']) + str(params['part']) + '_righteye', data_right)
            np.save('data/lectures_webcam/numpy/' + name + '_p' + str(params['lecture_part']) + str(params['part']) + '_eye_region', data_region)
            print(f'Lecture frames successfully saved with {len(data_right)} frames for the right eye, {len(data_left)} frames for the left eye and {len(data_region)} frames for the eye region')
            if params['part'] == splits and params['lecture_part'] == 2:
                folder_path = 'data/lectures_webcam/numpy'
                # Left eye
                file_names = os.listdir(folder_path)
                file_names = sorted(file_names)
                loaded_data = []
                for file in file_names:
                    if file.startswith(str(name)) and file.endswith('lefteye.npy'):
                        file_path = os.path.join(folder_path, file)
                        loaded_data.append(np.load(str(file_path)))
                        os.remove(file_path)

                data = np.concatenate(loaded_data, axis=0)
                np.save(folder_path + '/' + str(name) + '_lefteye', data)

                # Right eye
                file_names = os.listdir(folder_path)
                file_names = sorted(file_names)
                loaded_data = []
                for file in file_names:
                    if file.startswith(str(name)) and file.endswith('righteye.npy'):
                        file_path = os.path.join(folder_path, file)
                        loaded_data.append(np.load(str(file_path)))
                        os.remove(file_path)

                data = np.concatenate(loaded_data, axis=0)
                np.save(folder_path + '/' + str(name) + '_righteye', data)

                # Eye region
                file_names = os.listdir(folder_path)
                file_names = sorted(file_names)
                loaded_data = []
                for file in file_names:
                    if file.startswith(str(name)) and file.endswith('eye_region.npy'):
                        file_path = os.path.join(folder_path, file)
                        loaded_data.append(np.load(str(file_path)))
                        os.remove(file_path)

                data = np.concatenate(loaded_data, axis=0)
                np.save(folder_path + '/' + str(name) + '_eye_region', data)

        if params['ClosedEyes'] == True:
            np.save('data/closedEyes/numpy/' + name + '_p' + str(params['lecture_part']) + str(params['part']) + '_lefteye', data_left)
            np.save('data/closedEyes/numpy/' + name + '_p' + str(params['lecture_part']) + str(params['part']) + '_righteye', data_right)
            print(f'Closed_eye frames successfully saved with {len(data_right)} frames for the right eye and {len(data_left)} frames for the left eye')
            if params['part'] == splits and params['lecture_part'] == 2:
                folder_path = 'data/closedEyes/numpy'
                # Left eye
                file_names = os.listdir(folder_path)
                file_names = sorted(file_names)
                loaded_data = []
                for file in file_names:
                    if file.startswith(str(name)) and file.endswith('lefteye.npy'):
                        file_path = os.path.join(folder_path, file)
                        loaded_data.append(np.load(str(file_path)))
                        os.remove(file_path)

                data = np.concatenate(loaded_data, axis=0)
                # name = name.replace('_record', '')
                np.save(folder_path + '/' + str(name) + '_lefteye', data)
                # name = name.replace(str(name), str(name) + '_record')

                # Right eye
                file_names = os.listdir(folder_path)
                file_names = sorted(file_names)
                loaded_data = []
                for file in file_names:
                    if file.startswith(str(name)) and file.endswith('righteye.npy'):
                        file_path = os.path.join(folder_path, file)
                        loaded_data.append(np.load(str(file_path)))
                        os.remove(file_path)

                data = np.concatenate(loaded_data, axis=0)
                # name = name.replace('_record', '')
                np.save(folder_path + '/' + str(name) + '_righteye', data)
                # name = name.replace(str(name), str(name) + '_record')