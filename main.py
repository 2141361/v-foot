from tracker import Tracker
from utils import read_video_in_batches
from utils.utils import install_requirements
import cv2


def main():
    need_install_requirements = False
    if need_install_requirements:
        install_requirements('requirements.txt')

    video_path = 'data/videos/v30s.mp4'
    output_path = 'data/videos/v30s_annotated.mp4'

    batch_size = 150
    video_reader = cv2.VideoCapture(video_path)
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = Tracker('models/old_data.pt')

    # Video setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Color scheme for different classes
    class_colors = {
        'players': (0, 255, 0),      # Green
        'goalkeepers': (255, 0, 0),  # Red
        'referees': (0, 255, 255),   # Yellow
        'ball': (255, 0, 255)        # Magenta
    }

    # Class name abbreviations
    class_abbr = {
        'players': 'PL',
        'goalkeepers': 'GK',
        'referees': 'RF',
        'ball': 'BL'
    }

    for start_frame in range(0, total_frames, batch_size):
        video_frames = read_video_in_batches(video_reader, start_frame, batch_size)

        if not video_frames:
            break

        tracks = tracker.get_object_tracks(video_frames)
        tracker.add_position_to_tracks(tracks)


        for i, video_frame in enumerate(video_frames):
            # Annotate all classes
            for class_name in ['players', 'goalkeepers', 'referees', 'ball']:
                if class_name in tracks and i < len(tracks[class_name]):
                    for track_id, track_info in tracks[class_name][i].items():
                        bbox = track_info['bbox']
                        position = track_info['position']

                        # Draw bounding box
                        cv2.rectangle(video_frame,
                                     (int(bbox[0]), int(bbox[1])),
                                     (int(bbox[2]), int(bbox[3])),
                                     class_colors[class_name], 2)

                        # Draw class abbreviation and track ID
                        label = f"{class_abbr[class_name]}:{track_id}"
                        cv2.putText(video_frame,
                                    label,
                                    (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    class_colors[class_name],
                                    2)

                        # Draw position marker (small circle)
                        cv2.circle(video_frame,
                                  (int(position[0]), int(position[1])),
                                  3,
                                  class_colors[class_name],
                                  -1)

            # Write to output video
            out.write(video_frame)

    cap.release()
    out.release()


if __name__ == '__main__':
    main()