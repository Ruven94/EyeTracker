### >>> Import ###
import numpy as np
from torch.optim import Adam
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

from ClosedEyes.nn_closed_eyes_detection import *
### <<< Import ###

### >>> Parameter ###
name = 'ruven'
splits = 4
### <<< Parameter ###
def closed_eyes_detection_function(data_left_eye, data_right_eye, device = "mps"):
    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    model = ClosedEyeDetection().to(device)
    lr = 0.001
    optimizer = Adam(model.parameters(), lr=lr)

    par = torch.load('ClosedEyes/model_closed_eye_detection.pt')
    model.load_state_dict(par['model_state'])
    optimizer.load_state_dict(par['optimizer_state'])
    if data_left_eye.shape[1] == 3:
        data_le_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_left_eye]
    else:
        data_le_gray = data_left_eye
    if data_right_eye.shape[1] == 3:
        data_re_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_right_eye]
    else:
        data_re_gray = data_right_eye
    transform = transforms.Compose([transforms.ToTensor()])
    images_left_tensor = torch.stack([transform(img) for img in data_le_gray]).to(device)
    images_right_tensor = torch.stack([transform(img) for img in data_re_gray]).to(device)

    model.eval()

    with torch.no_grad():
        outputs_left = model(images_left_tensor)
    with torch.no_grad():
        outputs_right = model(images_right_tensor)

    probabilities_left = nn.functional.softmax(outputs_left, dim=1)
    probabilities_right = nn.functional.softmax(outputs_right, dim=1)

    probabilities_combined = (probabilities_left + probabilities_right) / 2

    predicted_classes_combined = torch.argmax(probabilities_combined, dim=1)
    predicted_classes_combined_np = predicted_classes_combined.cpu().numpy()

    return predicted_classes_combined_np
def eyes_to_remove(closed_eyes_points):
    remove_list = []
    skip_next = False
    tracker = 0

    for value in closed_eyes_points:
        if value == 0:
            remove_list.append(0)
            tracker = 3
        if value == 1 and tracker != 0:
            remove_list.append(0)
            tracker = tracker - 1
        elif value == 1 and tracker == 0:
            remove_list.append(1)

    remove_list = np.array(remove_list)

    return remove_list

data_le = np.load('data/closedEyes/numpy/' + str(name) + '_lefteye.npy', allow_pickle= True)
data_re = np.load('data/closedEyes/numpy/' + str(name) + '_righteye.npy', allow_pickle= True)

length = int(len(data_le)/splits)
closed_eyes_list = []
for i in range(splits):
    if i < (splits - 1):
        closed_eyes_list.extend(closed_eyes_detection_function(data_le[length * i: length * (i+1)], data_re[length * i: length * (i+1)], device = "cpu"))
    if i == (splits - 1):
        closed_eyes_list.extend(
        closed_eyes_detection_function(data_le[length * i:], data_re[length * i:], device="cpu"))

np.save('data/closedEyes/' + str(name) + '_closed_eyes', closed_eyes_list)

remove_list = eyes_to_remove(closed_eyes_list)
np.save('data/closedEyes/' + str(name) + '_remove_eyes', remove_list)