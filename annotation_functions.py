import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from visualization1 import add_rectangles, add_circles, draw_ruler_horizontal, draw_ruler_vertical
from main import load_model, predict

# bucket_name = "dataset-eyedata"
model, device = load_model()

def read_video(video_path):
    """
    Reads video from a given path and returns it as a numpy array.
    Input: path to video
    Output: numpy array
    """
    assert isinstance(video_path, str), "Video path must be a string type"

    recording = cv2.VideoCapture(video_path)
    recording_array = []

    if recording.isOpened() == False:
        print("Error while opening the video")

    while recording.isOpened():
        ret, frame = recording.read()
        # ret is a boolean variable telling if the video is read correctly or not
        if ret == True:
            recording_array.append(frame)
        else:
            break

    recording.release()
    recording_array = np.array(recording_array)
    return recording_array


def annotate_video_yolo(video):
    """
    Takes video as a numpy array and iterates over its frames to draw rectangles with YOLO predictions.
    Input: numpy array with video in shape [frames, x_dim, y_dim, channels]
    Output: numpy array with annotated video in shape [frames, x_dim, y_dim, channels],
    numpy array with a pupil sizes, numpy array with means of frames intensities
    """
    annotated_recording = []
    frame_means = []
    plot = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    upperLeftCornerOfText = (20, 100)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    print("Annotation started")
    for i in range(video.shape[0]):

        time = round(i/30, 1)
        frame = video[i]
        # yolo requires input in shape 640x640, the frame is later resized into
        # its original size
        frame_means.append(np.mean(frame))
        frame_resized = cv2.resize(frame, (640, 640))
        prediction = predict(model, device, frame_resized)
        try:
            frame_circle = add_circles(frame_resized, prediction)
            plot.append(prediction['pupil_mm'])
            frame_rectangle = add_rectangles(frame_resized, prediction)
            plot.append(prediction['pupil_mm'])

            cv2.putText(frame_resized, 'Time: ' + str(time) + ' s',
                upperLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
            draw_ruler_horizontal(frame_resized, 150, 170, 411, 170, color=(210,210,210), thickness=1)
            draw_ruler_vertical(frame_resized, 125, 210, 125, 426, color=(210,210,210), thickness=1)

        except:
            frame_circle = frame_resized
            plot.append(0)
            frame_rectangle = frame_resized
            plot.append(0)
            print('Missing prediction')
        frame_original_size = cv2.resize(
            frame_rectangle, (video.shape[2], video.shape[1])
        )
        frame_original_size = cv2.resize(
            frame_circle, (video.shape[2], video.shape[1])
        )
        # this resize currently results in a warning from imageio.mimwrite since input shape is not 512x512 but 510x512,
        # but maintaining input shape seems to be a correct approach
        annotated_recording.append(cv2.cvtColor(frame_original_size, cv2.COLOR_BGR2RGB))
        if i % 20 == 0 and i != 0:
            print("Finished processing {} frame".format(i))
    print("Annotation finished")
    return np.array(annotated_recording), np.array(plot), np.array(frame_means)


def phase_one_end_std(plr, beginning=15):
    """
    Method to calculate end of phase one of PLR recording. It is the phase before the flash.
    It works by calculating the standard deviation and mean of the beginning of the recording and
    finding the moment when intensity exceeds 3 standard deviations from the mean.
    Input: PLR as a series of mean of frames, number of last frame which for sure is before the flash
    Output: tuple in the form (index of end of first phase, standard deviation of first phase)
    """
    std_beginning, mean_beginning = np.std(plr[:beginning]), np.mean(
        plr[:beginning]
    )

    for i, p in enumerate(plr[beginning:]):
        if (
            p > mean_beginning + 3 * std_beginning
            or p < mean_beginning - 3 * std_beginning
        ):
            return i + beginning, std_beginning


def save_annotate_video_file(video_path, save_path, fps=30):
    """
    Takes path to the video and saves under the given path annotated video.
    Input: str with path to video, str with path to output, int with number of frames per second of the output video,
    with default fps=10 output video is 3x slower than the input (30fps)
    Output: None
    """
    assert isinstance(video_path, str), "Video path must be a string type"
    assert isinstance(save_path, str), "Save path must be a string type"

    video = read_video(video_path)
    annotated_video, plot, frame_means = annotate_video_yolo(video)
    #flash_start, _ = phase_one_end_std(frame_means)
    #plt.plot(plot)
    #plt.vlines(flash_start, np.min(plot), np.max(plot), color='red')
    #print("Saving")
    #plt.savefig(save_path[:-4] + 'plot.jpg')
    imageio.mimwrite(save_path, annotated_video, fps=fps)
    print("Annotated and saved")