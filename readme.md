# How to prepare the eye tracking:
### (1) Create calibration / recalibration and evaluation videos with the help of "preperations/create_calibration_videos.py"
### (2) Concatenate calibration videos with lecture videos with "preperations/create_lecture_eyetracking.py"
### (3) Place the recording in the following folder path: data/records

# How to use the eye tracker:
### (1) Use "create create_record_splits.py" to create splits of the used recording
### (2) Use "create_roi.py" to create region of interest for every part of the recording
### (3) Use "closed_eye_detection" to identify closed eyes and save a list
### (4) Use "eye_tracking.py" to train the neural network to estimate the gaze point

# How to analyse the trained model and the results:
### (1) Create gaze points with "analysis/gaze_points.py"
### (2) Create loss curves with "analysis/loss_curves.py"
### (3) Create heat maps with "analysis/heat_maps.py"
### (4) Create fixation maps with "analysis/fixations.py"
### (5) Create plots for the blinking behavior with "analysis/blink_behavior.py"
### Note: Alle the results are saved in the following folder path: "data/results"
