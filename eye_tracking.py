### >>> Import ###
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from torch.optim.lr_scheduler import StepLR

import torch
import os

from utils.get_points import *
from utils.pre_processing import *
from utils.helper_functions import *
# from utils.nn_eyetracker import *
from utils.nn_eyetracker_twoeyes import *
'''
Use utils.nn_eyetracker_twoeyes to compute faster but only with the left and right eye without the eye region
'''
### <<< Import ###

### >>> Parameter ###
name = 'ruven'
out_name = name
  # Change, if the model should have another name
server_run = False
augmentation = True
  # Was augmentation used?
calibration = True
recalibration = True
  # Should the model recalibrated?
start_epoch = None
  # Set to None if the process is not trained so far
num_epochs = 200
### <<< Parameter ###

parameter_set = []
if calibration == True:
  parameter = {'recalibration': False, 'start_epoch': start_epoch, 'augmentation': augmentation}
  parameter_set.append(parameter)
if recalibration == True:
  parameter = {'recalibration': True, 'start_epoch': num_epochs, 'augmentation': augmentation}
  parameter_set.append(parameter)
start_time = time.time()

closed_eyes = np.load('data/closedEyes/' + name + '_remove_eyes.npy', allow_pickle= True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = "cpu"
print(f'Using device: {device}')
# torch.mps.empty_cache()


if augmentation == True:
  augmentation_name = '_augmented'
else:
  augmentation_name = ''

### CALIBRATION DATA ###
_, stay_indices, points = get_points_train(random_seed= 123) # on-screen coordinates from data collection video

closed_eyes_calibration = closed_eyes[:len(get_points_train(random_seed=123)[0])]
closed_eyes_calibration = [value for index, value in enumerate(closed_eyes_calibration) if index not in stay_indices]
# eye region
data_er = np.load('data/calibration/numpy/' + str(name) + '_eye_region' + str(augmentation_name) + '.npy', allow_pickle= True)
rep = int(len(data_er) / len(points))
closed_eyes_calibration = closed_eyes_calibration * rep

data_er = np.array([data for i, data in enumerate(data_er) if closed_eyes_calibration[i] != 0])
images_er = torch.Tensor(data_er)
images_er = images_er.permute(0,3,1,2).to(device)

# left eye
data_le = np.load('data/calibration/numpy/' + str(name) + '_lefteye' + str(augmentation_name) + '.npy', allow_pickle= True)
data_le = np.array([data for i, data in enumerate(data_le) if closed_eyes_calibration[i] != 0])
images_le = torch.Tensor(data_le)3
images_le = images_le.permute(0,3,1,2).to(device)
# right eye
data_re = np.load('data/calibration/numpy/' + str(name) + '_righteye' + str(augmentation_name) + '.npy', allow_pickle= True)
data_re = np.array([data for i, data in enumerate(data_re) if closed_eyes_calibration[i] != 0])
images_re = torch.Tensor(data_re)
images_re = images_re.permute(0,3,1,2).to(device)

images = [images_er,images_le,images_re]

points = points * rep
filtered_points = [point for i, point in enumerate(points) if closed_eyes_calibration[i] != 0]
target_gaze = torch.Tensor(filtered_points).to(device)

calibrationset = TensorDataset(*images, target_gaze)

### EVAL DATA ###
cap = cv2.VideoCapture(f'data/closedEyes/{name}_p1.mp4')
p1_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
cap = cv2.VideoCapture(f'data/closedEyes/{name}_p2.mp4')
p2_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
eval_select_list = [range(0,401),range(400,801),range(800,1201)]
closed_eyes_select_list = [range(len(get_points_train(random_seed=123)[0]), (len(get_points_train(random_seed=123)[0]) + len(get_points_eval()[0])) + 1),
                           range(len(get_points_calibrate(random_seed=456)[0]) + p1_length, (len(get_points_calibrate(random_seed=456)[0]) + len(get_points_eval()[0])) + p1_length + 1),
                           range(p1_length + p2_length - len(get_points_eval()[0]), p1_length + p2_length + 1)]

evalset = []
for i in range(0,3):
  closed_eyes_eval = closed_eyes[closed_eyes_select_list[i][0]:closed_eyes_select_list[i][-1]]
  closed_eyes_eval = [value for index, value in enumerate(closed_eyes_eval) if index not in get_points_eval()[1]]

  # eye region
  data_eval_er = np.load('data/evaluation/numpy/' + name + '_eye_region.npy', allow_pickle=True)[eval_select_list[i][0]:eval_select_list[i][-1]]
  data_eval_er = np.array([data for i, data in enumerate(data_eval_er) if closed_eyes_eval[i] != 0])
  images_eval_er = torch.Tensor(data_eval_er)
  images_eval_er = images_eval_er.permute(0, 3, 1, 2).to(device)
  # left eye
  data_eval_le = np.load('data/evaluation/numpy/' + name + '_lefteye.npy', allow_pickle=True)[eval_select_list[i][0]:eval_select_list[i][-1]]
  data_eval_le = np.array([data for i, data in enumerate(data_eval_le) if closed_eyes_eval[i] != 0])
  images_eval_le = torch.Tensor(data_eval_le)
  images_eval_le = images_eval_le.permute(0, 3, 1, 2).to(device)
  # right eye
  data_eval_re = np.load('data/evaluation/numpy/' + name + '_righteye.npy', allow_pickle=True)[eval_select_list[i][0]:eval_select_list[i][-1]]
  data_eval_re = np.array([data for i, data in enumerate(data_eval_re) if closed_eyes_eval[i] != 0])
  images_eval_re = torch.Tensor(data_eval_re)
  images_eval_re = images_eval_re.permute(0, 3, 1, 2).to(device)

  images_eval = [images_eval_er,images_eval_le,images_eval_re]
  _, _, points_eval = get_points_eval()
  filtered_points_eval = [point for i, point in enumerate(points_eval) if closed_eyes_eval[i] != 0]
  target_gaze_eval = torch.Tensor(filtered_points_eval).to(device)

  evalset.append(TensorDataset(*images_eval, target_gaze_eval))
  print(f'Eval reading {i} done')

### RECALIBRATION DATA ###
_, stay_indices, points_recalibrate = get_points_calibrate(random_seed= 456) # on-screen coordinates from data collection video

closed_eyes_recalibration = closed_eyes[p1_length:(p1_length + len(get_points_calibrate(random_seed=456)[0]))]
closed_eyes_recalibration = [value for index, value in enumerate(closed_eyes_recalibration) if index not in stay_indices]
# eye region
data_recalibrate_er = np.load('data/recalibration/numpy/' + name + '_eye_region' + augmentation_name + '.npy', allow_pickle= True)
rep = int(len(data_recalibrate_er) / len(points_recalibrate))
closed_eyes_recalibration = closed_eyes_recalibration * rep

data_recalibrate_er = np.array([data for i, data in enumerate(data_recalibrate_er) if closed_eyes_recalibration[i] != 0])
images_recalibrate_er = torch.Tensor(data_recalibrate_er)
images_recalibrate_er = images_recalibrate_er.permute(0,3,1,2).to(device)

# left eye
data_recalibrate_le = np.load('data/recalibration/numpy/' + name + '_lefteye' + augmentation_name + '.npy', allow_pickle= True)
data_recalibrate_le = np.array([data for i, data in enumerate(data_recalibrate_le) if closed_eyes_recalibration[i] != 0])
images_recalibrate_le = torch.Tensor(data_recalibrate_le)
images_recalibrate_le = images_recalibrate_le.permute(0,3,1,2).to(device)
# right eye
data_recalibrate_re = np.load('data/recalibration/numpy/' + name + '_righteye' + augmentation_name + '.npy', allow_pickle= True)
data_recalibrate_re = np.array([data for i, data in enumerate(data_recalibrate_re) if closed_eyes_recalibration[i] != 0])
images_recalibrate_re = torch.Tensor(data_recalibrate_re)
images_recalibrate_re = images_recalibrate_re.permute(0,3,1,2).to(device)

images_recalibrate = [images_recalibrate_er,images_recalibrate_le,images_recalibrate_re]

points_recalibrate = points_recalibrate * rep
filtered_points_recalibrate = [point for i, point in enumerate(points_recalibrate) if closed_eyes_recalibration[i] != 0]
# filtered_points_recalibrate = points_recalibrate

target_gaze_recalibrate = torch.Tensor(filtered_points_recalibrate).to(device)

recalibrationset = TensorDataset(*images_recalibrate, target_gaze_recalibrate)

for params in parameter_set:
  ### DATALOADER ###
  batch_size = 128
  evalloader = []
  if params['recalibration'] == True: # Code for Recalibration
    trainloader = torch.utils.data.DataLoader(recalibrationset, batch_size=batch_size, shuffle = True)
    for i in range(1, 3):
      evalloader.append(torch.utils.data.DataLoader(evalset[i], batch_size=batch_size, shuffle=False))
  else: # Code without Recalibration
    trainloader = torch.utils.data.DataLoader(calibrationset, batch_size=batch_size, shuffle=True)
    for i in range(0, 3):
      evalloader.append(torch.utils.data.DataLoader(evalset[i], batch_size=batch_size, shuffle=False))

  print("Data loading complete")

  if server_run == True:
    path_model = ''
    path_loss = ''
    save_step = 20
  else:
    path_model = 'data/results/models/'
    path_loss = 'data/results/loss/'
    save_step = 30

  # EyeTracker
  model = EyeTracker().to(device)

  if params['start_epoch'] != None and params['recalibration'] == False:
    par = torch.load(path_model + out_name + '_model_' + str(params['start_epoch']) + '.pt')
    model.load_state_dict(par['model_state'])
  if params['recalibration'] == True:
    par = torch.load(path_model + out_name + '_model_best.pt')
    model.load_state_dict(par['model_state'])

  if params['recalibration'] == True:
    out_name = out_name + '_recalibrated'
    num_epochs = int(num_epochs / 2)

  # loss function
  loss_fn = nn.MSELoss()
  loss_fn_sum = nn.MSELoss(reduction='sum')

  # optimizer
  if params['start_epoch'] != None and params['recalibration'] == False:
    lr_history = np.load(f'{path_model}{out_name}_lr_history_' + str(params['start_epoch']) + '.npy').tolist()
    lr = lr_history[-1]
  elif params['start_epoch'] != None and params['recalibration'] == True:
    lr = 0.0001
    lr_history = [lr]
  else:
    lr = 0.01
    lr_history = [lr]

  optimizer = Adam(model.parameters(), lr = lr)

  if params['recalibration'] == True:
    optimizer = Adam(model.parameters(), lr = lr, weight_decay= 0.01)
  if params['start_epoch'] != None and params['recalibration'] == False:
    optimizer.load_state_dict(par['optimizer_state'])
    optimizer.param_groups[0]['lr'] = lr
  scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

  print(f'Model read in')

  def train_et():
    total_loss = 0
    total_distance = 0
    model.train()  # learning weights
    # for img_er, img_le, img_re, labels in trainloader:
    for _, img_le, img_re, labels in trainloader:
      # set gradients to zero
      optimizer.zero_grad()
      # forward pass
      # y_pred = model(img_er,img_le,img_re)
      y_pred = model(img_le,img_re)
      # compute loss
      loss = loss_fn(y_pred.squeeze(), labels)
      # loss for training loss
      loss_train = loss_fn_sum(y_pred.squeeze(), labels)
      distance = torch.norm(y_pred - labels, dim=1).sum()
      # backward pass
      loss.backward()
      # adjust weights
      optimizer.step()
      # Track loss
      total_loss += loss_train.item()
      total_distance += distance

    total_distance = total_distance.item()
    return total_loss, total_distance

  def eval_et(evalloader):
    total_loss = 0
    total_distance = 0
    model.eval()  # froze weights
    with torch.no_grad():
      # for img_er, img_le, img_re, labels in evalloader:
      for _, img_le, img_re, labels in evalloader:
        # y_pred = model(img_er, img_le, img_re)
        y_pred = model(img_le, img_re)
        loss = loss_fn_sum(y_pred.squeeze(), labels)
        distance = torch.norm(y_pred - labels, dim=1).sum()
        total_loss += loss.item()
        total_distance += distance

    total_distance = total_distance.item()
    return total_loss, total_distance

  if params['start_epoch'] != None and params['recalibration'] == False:
    loss_matrix = np.load(path_loss + str(out_name) + '_loss_' + str(params['start_epoch']) +'.npy').tolist()
    distance_matrix = np.load(path_loss + str(out_name) + '_distance_' + str(params['start_epoch']) +'.npy').tolist()
    start_epoch == params['start_epoch']
  else:
    loss_matrix = []
    distance_matrix = []
    start_epoch = 0

  for epoch in range(start_epoch + 1,num_epochs + 1):
    epoch_time = time.time()
    train_loss, train_distance = train_et()
    train_loss = train_loss / len(calibrationset) / 2
    train_distance = train_distance / len(calibrationset)
    print(f'Epoch {epoch} | Time: {round(time.time() - epoch_time)} sec | Train loss = {round(train_loss)} | Train distance = {round(train_distance)} | lr = {lr_history[-1]}')

    if epoch%1==0 or epoch < 10:
      eval_loss = []
      eval_distance = []
      for i in range(0,len(evalloader)):
        loss, distance = eval_et(evalloader[i])
        eval_loss.append(loss / len(evalset[i]) / 2)
        eval_distance.append(distance / len(evalset[i]))

      if params['recalibration'] == False:
        loss_matrix.append([epoch, train_loss, eval_loss[0], eval_loss[1], eval_loss[2]])
        distance_matrix.append([epoch, train_distance, eval_distance[0], eval_distance[1], eval_distance[2]])

        print(f'Epoch {epoch} | Time: {round(time.time() - epoch_time)} sec | Train loss = {round(train_loss)} | Eval1 loss = {round(eval_loss[0])} | Eval2 loss = {round(eval_loss[1])} | Eval3 loss = {round(eval_loss[2])} ')
        print(f'Epoch {epoch} | Time: {round(time.time() - epoch_time)} sec | Eval1 distance = {round(eval_distance[0])} | Eval2 distance = {round(eval_distance[1])} | Eval3 distance = {round(eval_distance[2])} ')

      else:
        loss_matrix.append([epoch, train_loss, eval_loss[0], eval_loss[1]])
        distance_matrix.append([epoch, train_distance, eval_distance[0], eval_distance[1]])
        print(f'Epoch {epoch} | Time: {round(time.time() - epoch_time)} sec | Train loss = {round(train_loss)} | Eval1 loss = {round(eval_loss[0])} | Eval2 loss = {round(eval_loss[1])}')
        print(f'Epoch {epoch} | Time: {round(time.time() - epoch_time)} sec | Eval1 distance = {round(eval_distance[0])} | Eval2 distance = {round(eval_distance[1])}')

    if epoch%save_step==0:
        torch.save({'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                    }, f'{path_model}{out_name}_model_{epoch}.pt')
        np.save(f'{path_loss}{out_name}_loss_{epoch}', loss_matrix)
        np.save(f'{path_loss}{out_name}_distance_{epoch}', distance_matrix)
        np.save(f'{path_model}{out_name}_lr_history_{epoch}', np.array(lr_history))
        print("Parameter and loss of Iteration " + str(epoch) + " saved!")

    if 'best_model_score' not in locals() and 'best_model_score' not in globals():
      best_model_score = np.inf
      print(f'Best model score: {best_model_score}')

    if params['recalibration'] == False:
      new_score = distance_matrix[-1][2] + distance_matrix[-1][3] + distance_matrix[-1][4]
      check_epoch = 150
    else:
      new_score = distance_matrix[-1][2] + distance_matrix[-1][3]
      check_epoch = 30

    if epoch >= check_epoch and new_score < best_model_score:
      torch.save({'model_state': model.state_dict(),
                  'optimizer_state': optimizer.state_dict()
                  }, f'{path_model}{out_name}_model_best.pt')
      best_model_score = new_score
      print(f'Best model saved: {best_model_score}')

    scheduler.step()
    lr_history.append(optimizer.param_groups[0]['lr'])

    if epoch == num_epochs and server_run == False:
      folder_path = 'data/results/models'
      file_names = os.listdir(folder_path)
      file_names = sorted(file_names)
      for file in file_names:
        if file.startswith(str(out_name) + '_lr_history'):
          file_path = os.path.join(folder_path, file)
          os.remove(file_path)
        if file.startswith(str(out_name) + '_model') and not file.endswith('_200.pt'):
          file_path = os.path.join(folder_path, file)
          os.remove(file_path)
      folder_path = 'data/results/loss'
      file_names = os.listdir(folder_path)
      file_names = sorted(file_names)
      for file in file_names:
        if file.startswith(str(out_name) + '_loss') and not file.endswith('_200.npy'):
          file_path = os.path.join(folder_path, file)
          os.remove(file_path)
        if file.startswith(str(out_name) + '_distance') and not file.endswith('_200.npy'):
          file_path = os.path.join(folder_path, file)
          os.remove(file_path)


  print(f'Process finished | Total time: {round((time.time() - start_time)/60)} min!')
