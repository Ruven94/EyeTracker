'''
This file can be used to label images from records with 'closed eye' or 'open eye' quickly
'''

### <<< Import ###
import numpy as np
import cv2
import torch
### >>> Import ###

### >>> Parameter ###
name = 'ruven'
### <<< Parameter ###

data_le = np.load(f'../data/calibration/numpy/{name}_lefteye_augmented.npy', allow_pickle= True)[:2800] # without augmentation
data_re = np.load(f'../data/calibration/numpy/{name}_righteye_augmented.npy', allow_pickle= True)[:2800] # without augmentation

### >>> Model ###
data_le_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_le]
data_re_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_re]

i = 0
labels = []
while True:
    cv2.putText(data_le[i], f'({i})', (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(data_re[i], f'({i})', (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.imshow('Left eye', data_le[i])
    cv2.imshow('Right eye', data_re[i])

    key = cv2.waitKey()

    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('e'):
        labels.append(1)
        i += 1
        continue
    if key & 0xFF == ord('d'):
        labels.append(0)
        i += 1
        continue
    if key & 0xFF == ord('w'):
        del labels[-1]
        i -= 1
        continue
    if key & 0xFF == ord('s'):
        del labels[-1]
        i -= 1
        continue

cv2.destroyAllWindows()

np.save(f'data_labeled/{name}_closed_eyes_labels.npy', labels)