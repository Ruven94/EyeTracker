### >>> Import ###
import cv2
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from moviepy.editor import VideoFileClip, CompositeAudioClip
from tqdm import tqdm  # tqdm für die Fortschrittsanzeige

from utils.nn_eyetracker import *
from utils.get_points import *
### <<< Import ###

### >>> Parameter  ###
name = 'ruven'
epoch = 200
epoch_recalibration = 50
fps = 25
# model_name = str(name) + '_model_' + str(epoch)
# model_name_recalibrated = str(name) + '_recalibrated_model_' + str(epoch_recalibration)
model_name = str(name) + '_model_best'
model_name_recalibrated = str(name) + '_recalibrated_model_best'
### <<< Parameter  ###

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(f'Using device: {device}')

cap = cv2.VideoCapture('../data/lectures_webcam/' + str(name) + '_p1.mp4')
length_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
length = int(length_total - length_total % fps)
cap.release()
cap = cv2.VideoCapture('../data/lectures_webcam/' + str(name) + '_p2.mp4')
length_vid2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(cv2.CAP_PROP_FRAME_COUNT) % fps)
cap.release()

video_seq = [range(0,length), range(length_total, length_total + length_vid2)]
model_name_list = [str(model_name),str(model_name_recalibrated)]

y_pred_np = []
for i, sec in enumerate(video_seq):

    model = EyeTracker().to(device)
    par = torch.load('../data/results/models/' + model_name_list[i] + '.pt')
    model.load_state_dict(par['model_state'])
    loss_fn_sum = nn.MSELoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=0.01)
    optimizer.load_state_dict(par['optimizer_state'])

    # eye region
    data_er = np.load('../data/lectures_webcam/numpy/' + name + '_eye_region.npy', allow_pickle= True)[sec]
    images_er = torch.Tensor(data_er)
    images_er = images_er.permute(0,3,1,2).to(device)
    # left eye
    data_le = np.load('../data/lectures_webcam/numpy/' + name + '_lefteye.npy', allow_pickle= True)[sec]
    images_le = torch.Tensor(data_le)
    images_le = images_le.permute(0,3,1,2).to(device)
    # right eye
    data_re = np.load('../data/lectures_webcam/numpy/' + name + '_righteye.npy', allow_pickle= True)[sec]
    images_re = torch.Tensor(data_re)
    images_re = images_re.permute(0,3,1,2).to(device)

    images = [images_er,images_le,images_re]

    dataset = TensorDataset(*images)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle = False)

    print('Data loaded!')

    model.eval()  # froze weights

    y_pred = []

    progress_bar = tqdm(total=len(dataloader), desc='Frames processed')
    with torch.no_grad():
        for img_er, img_le, img_re in dataloader:
        # for _,img_le, img_re in dataloader:
        #     y_pred.extend(model(img_le,img_re))
            y_pred.extend(model(img_er,img_le,img_re))
            progress_bar.update(1)

    y_pred_np.append([elem.numpy() for elem in y_pred])
    progress_bar.close()

gaze_points = []
gaze_points.extend(y_pred_np[0])
gaze_points.extend(y_pred_np[1])
gaze_points = np.array(gaze_points)
np.save(f'../data/results/gaze_points/{name}_gazepoints.npy', gaze_points)

removed_eyes = np.load(f'../data/closedEyes/{name}_remove_eyes.npy')

calibration_length = 120 * fps
eval_length = 20 * fps
recalibration_length = 60 * fps
follow_point = 3 * fps

removed_eyes = removed_eyes[calibration_length + eval_length : -eval_length -follow_point]
removed_eyes_lecture = np.concatenate((removed_eyes[:length], removed_eyes[-length_vid2:]))

gaze_points[removed_eyes_lecture == 0] = [None, None]

np.save(f'../data/results/gaze_points/{name}_gazepoints_blinkremoved.npy', gaze_points)
