### >>> Import ###
from moviepy.editor import VideoFileClip
import cv2
### <<< Import ###

### >>> Parameter ###
name = 'ruven_glasses'
computer_ressource_shift = 4 # A shift regarding the delay from screen to webcam
### <<< Parameter ###

### Code ###
def split_video(input, output):
    video_path, start_frame, end_frame = input

    video = VideoFileClip(video_path)

    fps = video.fps
    # fps = 25

    start_time = start_frame / fps
    end_time = end_frame / fps

    video = video.subclip(start_time, end_time)
    video.write_videofile(output, codec="libx264", audio_codec="aac")

input_path = 'data/records/' + str(name) + '_record_p1.mp4'
output_path_record1 = 'data/lectures_webcam/'+str(name)+'_p1.mp4'
output_path_record2 = 'data/lectures_webcam/'+str(name)+'_p2.mp4'
output_path_calib = 'data/calibration/'+str(name)+'.mp4'
output_path_eval1 = 'data/evaluation/'+str(name)+'_1.mp4'
output_path_eval2 = 'data/evaluation/'+str(name)+'_2.mp4'
output_path_eval3 = 'data/evaluation/'+str(name)+'_3.mp4'
output_path_recalib = 'data/recalibration/'+str(name)+'.mp4'
output_closed_eyes1 = 'data/closedEyes/'+str(name)+'_p1.mp4'
output_closed_eyes2 = 'data/closedEyes/'+str(name)+'_p2.mp4'

# Synchronize videos
vid = cv2.VideoCapture(input_path)
fps = int(vid.get(cv2.CAP_PROP_FPS))
current_frame = 0
while True:
    ret, frame = vid.read()
    if frame[2,2][2] > 50 and current_frame > int(vid.get(cv2.CAP_PROP_FPS)) * 3 - 5:
        shift = current_frame - fps * 3
        break
        print(frame[2,2][2])
    current_frame += 1
# shift = 0
shift += computer_ressource_shift
print(shift)

input_train = (input_path, 3 * fps + shift, 123 * fps + shift)
split_video(input_train, output_path_calib)
input_eval1 = (input_path, 123 * fps + shift, 143 * fps + shift)
split_video(input_eval1, output_path_eval1)
input_lecture1 = (input_path, 143 * fps + shift, 1113 * fps + shift)
split_video(input_lecture1, output_path_record1)
input_closedeyes = (input_path, 3 * fps + shift, 1113 * fps + shift)
split_video(input_closedeyes, output_closed_eyes1)

# Synchronize videos
input_path = input_path.replace('_p1.mp4', '_p2.mp4')
vid = cv2.VideoCapture(input_path)
current_frame = 0
while True:
    ret, frame = vid.read()
    if frame[2,2][2] > 50 and current_frame > int(vid.get(cv2.CAP_PROP_FPS)) * 3 - 5:
        shift = current_frame - int(vid.get(cv2.CAP_PROP_FPS)) * 3
        break
        print(frame[2, 2][2])

    current_frame += 1
# shift = 0
shift += computer_ressource_shift
print(shift)

input_calib = (input_path, 3 * fps + shift, 63 * fps + shift)
split_video(input_calib, output_path_recalib)
input_eval2 = (input_path, 63 * fps + shift, 83 * fps + shift)
split_video(input_eval2, output_path_eval2)
input_lecture2 = (input_path, 83 * fps + shift, 1242 * fps + shift)
split_video(input_lecture2, output_path_record2)
input_eval3 = (input_path, 1245 * fps + shift, 1265 * fps + shift)
split_video(input_eval3, output_path_eval3)
input_closedeyes = (input_path, 3 * fps + shift, 1265 * fps + shift)
split_video(input_closedeyes, output_closed_eyes2)