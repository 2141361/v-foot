import subprocess
import sys
import cv2


def install_requirements(file_path):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", file_path])

def read_video_in_batches(video_reader, start_frame, batch_size):
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(batch_size):
        ret, frame = video_reader.read()
        if not ret:
            break
        frames.append(frame)

    return frames