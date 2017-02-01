from collections import deque

from moviepy.editor import VideoFileClip

from find_cars_in_image import *


# Converts a clip from raw file to a file with lane overlay
def write_clip(input_file, output_file, function):
    clip = VideoFileClip(input_file)

    global prev_hot_windows
    prev_hot_windows = deque([])

    white_clip = clip.fl_image(function)  # NOTE: this function expects color images!
    white_clip.write_videofile(output_file, audio=False)


def process_frame(image):
    global prev_hot_windows
    result_image, prev_hot_windows = find_cars_in_image(image, prev_hot_windows)
    return result_image


video_input1 = 'project_video.mp4'
video_output1 = 'project_video_solution.mp4'
write_clip(video_input1, video_output1, process_frame)
