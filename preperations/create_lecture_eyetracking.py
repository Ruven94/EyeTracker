### >>> Import ###
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
### <<< Import ###

def concatenate_videos(video_list, output_path):
    clips = []

    for video_path, start_time, end_time in video_list:
        video = VideoFileClip(video_path).subclip(start_time, end_time)
        clips.append(video)

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

### >>> Parameter ###
lecture_name = '2023-01-26 E3.1 Matheeinführung Summen'
lecture_break = 16 * 60 + 10
fps = 25
### <<< Parameter ###

### Code ###
lecture = cv2.VideoCapture('data/lectures/' + lecture_name + '.mp4')
train = ('data/assets/points_collection_' + str(fps) + 'fps.mp4', 0, 120)
eval = ('data/assets/points_collection_' + str(fps) + 'fps_eval.mp4',0,20)
calibrate = ('data/assets/points_collection_' + str(fps) + 'fps_calibrate.mp4',0 , 60)
lecture_1 = ('data/lectures/' + lecture_name + '.mp4', 0 , lecture_break)
lecture_2 = ('data/lectures/' + lecture_name + '.mp4', lecture_break, int(int(lecture.get(cv2.CAP_PROP_FRAME_COUNT)) / int(lecture.get(cv2.CAP_PROP_FPS))))
instruction = ('data/assets/instruction_vid.mp4', 0, 3)

# # Time line # #
# PART 1 #
# instruction 0 - 3
# train 3 - 123
# eval 123 - 143
# 143 + 16 * 60 + 10
# lecture_1 143 - 1113

# PART 2 #
# instruction 0 - 3
# calib 3 - 63
# eval 63 - 83
# 83 + 35 * 60 + 29 - 16 * 60 - 10
# lecture_2 83 - 1242
# instruction 1242 - 1245
# eval 1245 - 1265

video_list_1 = [instruction, train, eval, lecture_1]
video_list_2 = [instruction, calibrate, eval, lecture_2, instruction, eval]

# Videos for eye tracking
output_path_1 = 'data/lectures/' + lecture_name + '_eyetracking_p1.mp4'
output_path_2 = 'data/lectures/' + lecture_name + '_eyetracking_p2.mp4'

concatenate_videos(video_list_1, output_path_1)
concatenate_videos(video_list_2, output_path_2)
