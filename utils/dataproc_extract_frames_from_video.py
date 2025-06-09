# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import cv2
import os
import glob


def extract_frames_from_folder(input_folder, output_base_dir, frame_rate=10):
    """
    Extract frames from all MP4 videos in a folder and save them as images.

    :param input_folder: Path to the folder containing MP4 video files.
    :param output_base_dir: Directory where the frames will be saved.
    :param frame_rate: Number of frames to extract per second.
    """
    # Find all MP4 files in the input folder
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))

    if not video_files:
        print(f"No MP4 files found in {input_folder}")
        return

    for video_file in video_files:
        # Get the video filename without extension
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_dir = os.path.join(output_base_dir, video_name)

        # Extract frames for the current video
        extract_frames(video_file, output_dir, frame_rate)


def extract_frames(video_path, output_dir, frame_rate=10):
    """
    Extract frames from a video and save them as images.

    :param video_path: Path to the input video file.
    :param output_dir: Directory where the frames will be saved.
    :param frame_rate: Number of frames to extract per second.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps / frame_rate)
    if frame_interval == 0:
        frame_interval = 1

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame every 'frame_interval' frames
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    print(f"Frames extracted from {video_path}. Total frames saved: {saved_frame_count}")

    # Release the video capture object
    cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing MP4 video files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Directory where the frames will be saved.",
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=10,
        help="Number of frames to extract per second (default: 10).",
    )

    args = parser.parse_args()

    # Call the main function with parsed arguments
    extract_frames_from_folder(args.input_folder, args.output_folder, args.frame_rate)

