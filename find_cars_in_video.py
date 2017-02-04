from collections import deque

from moviepy.editor import VideoFileClip

from find_cars_in_image import *


# Converts a clip from raw file to a file with lane overlay
def write_clip(input_file, output_file, function):
    clip = VideoFileClip(input_file)

    # Create global previous hot frames to be used by frame processing function
    global prev_hot_windows
    prev_hot_windows = deque([])

    white_clip = clip.fl_image(function)  # NOTE: this function expects color images!
    white_clip.write_videofile(output_file, audio=False)


# Processes one frame of video
def process_frame(image):
    # Obtain previous hot frames
    global prev_hot_windows

    # Find cars in given frame and draw bounding boxes around them
    result_image, prev_hot_windows = find_cars_in_image(image, prev_hot_windows)

    # Return resulting image
    return result_image


# Define input and output files for video
video_input1 = 'project_video.mp4'
video_output1 = 'project_video_solution.mp4'

# Process the video
write_clip(video_input1, video_output1, process_frame)
