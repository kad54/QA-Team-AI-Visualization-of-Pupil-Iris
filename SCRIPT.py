from annotation_functions import save_annotate_video_file
from datetime import datetime

folder_path = ""


def main():
    video_path = input("Enter path to file: ")
    video_path = str(video_path).replace('""', "\\")

    if video_path[0] == '"':
        video_path = video_path[1:-1]

    output_path = folder_path + str(datetime.today()).replace(":", "-") + ".mp4"
    save_annotate_video_file(video_path, output_path, fps=30)


if __name__ == "__main__":
    main()
