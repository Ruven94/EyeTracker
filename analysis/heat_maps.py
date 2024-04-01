### >>> Import ###
import seaborn as sns
from scipy.stats import gaussian_kde
from utils.get_points import *
import cv2
import matplotlib.pyplot as plt
### <<< Import ###

### >>> Functions ###
def heatmap_eye(gaze_points, frame, colorbar = False, title = '', save=False, path=None):
    x, y = zip(*gaze_points)

    kde = gaussian_kde(np.vstack([x, y]))

    frame_height, frame_width, _ = frame.shape
    X, Y = np.meshgrid(np.linspace(0, frame_width, 500), np.linspace(0, frame_height, 500))
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)

    plt.figure(figsize=(10, 6))
    sns.set(style="white")
    sns.color_palette("Spectral", as_cmap=True)
    plt.imshow(frame, extent=[0, frame_width, 0, frame_height])  # Invertierung der Y-Achse entfernt
    plt.imshow(Z, cmap="Spectral_r", alpha=0.6, extent=[0, frame_width, 0, frame_height])  # Invertierung der Y-Achse entfernt
    if colorbar == True:
        plt.colorbar(label='Density')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(str(title))
    if save == True:
        plt.savefig(f'{path}Heatmap_slide{str(slide)} {name}.png',
                    transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
### <<< Functions ###

### >>> Parameter ###
name = 'ruven'
lecture_path = '../data/lectures/2023-01-26 E3.1 Matheeinführung Summen_eyetracking_p1.mp4'
lecture_path2 = '../data/lectures/2023-01-26 E3.1 Matheeinführung Summen_eyetracking_p2.mp4'
heatmap_path = '../data/results/analysis/heat_maps/'
fps = 25

save = True
### <<< Parameter ###

gaze_points = np.load(f'../data/results/gaze_points/{name}_gazepoints_blinkremoved.npy')
# gaze_points = np.load(f'../data/results/gaze_points/{name}_gazepoints.npy')
shift = 3 * fps + 120 * fps + 20 * fps

### Slide 4 ###
slide = 4
gaze_points_sub = gaze_points[8 * 25:28 * 25]
nan_mask = np.isnan(gaze_points_sub)
gaze_points_sub = gaze_points_sub[~nan_mask.any(axis=1)]
vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 11*25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

heatmap_eye(gaze_points_sub,frame, colorbar= True, title = 'Slide ' + str(slide), save = save, path = heatmap_path)

### Slide 5 ###
slide = 5
gaze_points_sub = gaze_points[28*25:59*25]
nan_mask = np.isnan(gaze_points_sub)
gaze_points_sub = gaze_points_sub[~nan_mask.any(axis=1)]
vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 40*25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

heatmap_eye(gaze_points_sub,frame,title = 'Slide ' + str(slide), save = save, path = heatmap_path)

### Slide 6 ###
slide = 6
gaze_points_sub = gaze_points[59*25 : (2 * 60 * 25 + 42 * 25)]
nan_mask = np.isnan(gaze_points_sub)
gaze_points_sub = gaze_points_sub[~nan_mask.any(axis=1)]
vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 66*25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

heatmap_eye(gaze_points_sub,frame,title = 'Slide ' + str(slide), save = save, path = heatmap_path)

### Slide 7 ###
slide = 7
gaze_points_sub = gaze_points[(2 * 60 * 25 + 42 * 25):(4 * 60 * 25 + 10 * 25)]
nan_mask = np.isnan(gaze_points_sub)
gaze_points_sub = gaze_points_sub[~nan_mask.any(axis=1)]
vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 4500)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

heatmap_eye(gaze_points_sub,frame,title = 'Slide ' + str(slide), save = save, path = heatmap_path)

### Slide 8 ###
slide = 8
gaze_points_sub = gaze_points[(4 * 60 * 25 + 10 * 25):(9 * 60 * 25 + 45 * 25)]
nan_mask = np.isnan(gaze_points_sub)
gaze_points_sub = gaze_points_sub[~nan_mask.any(axis=1)]
vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 9*60*25 + 22 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

heatmap_eye(gaze_points_sub,frame,title = 'Slide ' + str(slide), save = save, path = heatmap_path)

### Slide 9 ###
slide = 9
gaze_points_sub = gaze_points[(9 * 60 * 25 + 45 * 25):(11 * 60 * 25 + 46 * 25)]
nan_mask = np.isnan(gaze_points_sub)
gaze_points_sub = gaze_points_sub[~nan_mask.any(axis=1)]
vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 11*60*25 + 24 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

heatmap_eye(gaze_points_sub,frame,title = 'Slide ' + str(slide), save = save, path = heatmap_path)

### Slide 10 ###
slide = 10
gaze_points_sub = gaze_points[(11 * 60 * 25 + 46 * 25):(16 * 60 * 25 + 10 * 25)]
nan_mask = np.isnan(gaze_points_sub)
gaze_points_sub = gaze_points_sub[~nan_mask.any(axis=1)]
vid = cv2.VideoCapture(lecture_path)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 13*60*25 + 48 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

heatmap_eye(gaze_points_sub,frame,title = 'Slide ' + str(slide), save = save, path = heatmap_path)

### 2. Part ###
cap = cv2.VideoCapture(f'../data/lectures_webcam/{name}_p2.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

gaze_points = gaze_points[-length:]

shift = 3 * fps + 60 * fps + 20 * fps


### Slide 11 ###
slide = 11
gaze_points_sub = gaze_points[(0 * 60 * 25 + 4 * 25):(1 * 60 * 25 + 15 * 25)]
nan_mask = np.isnan(gaze_points_sub)
gaze_points_sub = gaze_points_sub[~nan_mask.any(axis=1)]
vid = cv2.VideoCapture(lecture_path2)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 0*60*25 + 15 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

heatmap_eye(gaze_points_sub,frame,title = 'Slide ' + str(slide), save = save, path = heatmap_path)

### R Slide 1 ###
slide = 12
gaze_points_sub = gaze_points[(14 * 60 * 25 + 30 * 25):(17 * 60 * 25 + 39 * 25)]
nan_mask = np.isnan(gaze_points_sub)
gaze_points_sub = gaze_points_sub[~nan_mask.any(axis=1)]
vid = cv2.VideoCapture(lecture_path2)
vid.set(cv2.CAP_PROP_POS_FRAMES, shift + 17*60*25 + 30 * 25)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

heatmap_eye(gaze_points_sub,frame,title = 'Example R-Slide', save = save, path = heatmap_path)
