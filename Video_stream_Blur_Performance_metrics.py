##

# THIS CODE RECEIVES THE VIDEO STREAM THROUGH A VIDEO CAPTURE DEVICE
# THIS CODE CAN BE RUN ON A COMPUTER THAT IS CONNECTED TO AN INTRAOPERATIVE CAMERA THROUGH A VIDEO CAPTURE DEVICE
# THIS CODE INTEGRATES THE FOLLOWING
# AN ALGORITHM TO DIFFERENTIATE SURGICAL SCENES FROM NON-SURGICAL SCENES
# OPTICAL FLOW TO DETECT CAMERA MOVEMENT AND SPEED
# INSTRUMENT DETECTION ALGORITHMS TO DETECT INSTRUMENTS AND CALCULATE MOVEMENT METRICS

# Performance metrics related to instrument movement, safety, dexterity, and efficiency are to be calculated
# based on instrument location, speed, acceleration, time, length of the procedure, time without any instrument,
# number of instruments being used, how close the two instruments are, amount of time the presence of more than one instrument,
# percentage of time using each instrument, and more to be defined specific to different procedures.

import numpy as np
import cv2
from ultralytics import YOLO
import datetime
import os
import csv


# VIDEO CAPTURE DEVICE
device_index = 0    # 0 for webcam / 1 for video capture device
cap = cv2.VideoCapture(device_index)
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# To set the video capture to 4K resolution: 3840 x 2160
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Recording at resolution: {frame_width}x{frame_height}")

# LOAD THE COMPUTER VISION MODELS
model_path_deidentification = 'modelfordeidentification'
model_path_instrumentdetection = 'modelforinstrumentdetection'
# model_path_phasesandsteps =

model_instrumentdetection = YOLO(model_path_instrumentdetection)
model_deidentification = YOLO(model_path_deidentification)
class_names_dict = model_instrumentdetection.names  # Adjust if the way to access class names differs


## RECORD METRICS

# Define the CSV file path
current_dir = os.getcwd()     # current directory
current_time = datetime.datetime.now()
csv_file_path = os.path.join(current_dir, "Performance_data_" + current_time.strftime("%Y%m%d_%H%M%S") + ".csv")


## INSTRUMENT TRACKING

class InstrumentTracker:
    def __init__(self):
        self.previous_positions = {}  # Stores previous positions of instruments {id: (x_center, y_center)}
        self.speeds = {}  # Stores speeds of instruments {id: speed}
        self.speeds_x = {}
        self.speeds_y = {}

    def update(self, detection, frame_rate):
        data = detection.data.cpu().numpy()
        x_center, y_center, x2, y2 = data[0][:4].astype(int)
        id = int(data[0][5])
        # id = detection.boxes.id.int().item()
        # x_center, y_center = detection.boxes.xywh[0], detection.boxes.xywh[1]
        current_position = (x_center, y_center)

        # Calculate speed if previous position exists
        if id in self.previous_positions:
            prev_position = self.previous_positions[id]
            speed = (np.linalg.norm(np.array(current_position) - np.array(prev_position))) * frame_rate
            # Calculate the distance in x and y separately
            speed_x = (current_position[0] - prev_position[0]) * frame_rate
            speed_y = (current_position[1] - prev_position[1]) * frame_rate

            self.speeds[id] = [speed, speed_x, speed_y]
        else:
            self.speeds[id] = [0, 0, 0]  # Initial speed is 0

        # Update the position
        self.previous_positions[id] = current_position
        # self.speeds[id] = (speed, speed_x, speed_y)

    def get_speed(self, id):
        return self.speeds.get(id, (0, 0, 0))


frame_rate = cap.get(cv2.CAP_PROP_FPS)
tracker = InstrumentTracker()
prev_positions = {}
current_positions = {}


## OPTICAL FLOW

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)     # Number of points to track is maxCorners
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
frame_height, frame_width = old_frame.shape[:2]
mask = np.zeros_like(old_frame[:, :, 0])
mask[:, frame_width // 4:frame_width * 3 // 4] = 255

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
original_count = len(p0)

# Adjustable parameter for deciding when to display "unknown"
unknown_threshold = 0.75  # if 75% of points not red then the speed is unknown


## BLURRING

# Function to apply blurring
def apply_blur(frame):
    return cv2.GaussianBlur(frame, (111, 111), 0)


## OPTION TO SAVE VIDEO

save_video = True
if save_video:
    start_time = datetime.datetime.now()
    video_file_path = os.path.join(current_dir, "Video_output_" + start_time.strftime("%Y%m%d_%H%M%S") + ".mp4")
    out = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (frame_width, frame_height))

# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = ['Frame_number', 'Time', 'Camera Speed', 'Camera Speed X Axis', 'Camera Speed Y Axis']
    for i in range(1, 13):
        header += [f'Instrument {i} Name', f'Instrument {i} Speed', f'Instrument {i} Speed X Axis', f'Instrument {i} Speed Y Axis']
    csv_writer.writerow(header)  # Write header

    ## VISUALIZE VIDEO
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        ## SURGICAL VS NON-SURGICAL
        # Classify the frame
        results = model_deidentification(frame, verbose=False, conf=0.9)[0]
        probs = results.probs  # Classification probabilities for each detected class
        predicted_class = model_deidentification.names[int(probs.top1)]  # Get the class with the highest probability

        if predicted_class == 'Ex Situ':    # Nothing applies to the frame expect blurring if it is non-surgical
            frame = apply_blur(frame)
            if save_video:
                out.write(frame)  # Save frame to video without modifications except blurring

        else:
            if save_video:
                out.write(frame)  # Save frame to video without modifications

            # Perform instrument detection on the frame
            results = model_instrumentdetection(frame, conf = 0.7)

            result = results[0]

            # Iterate over detections stored in the boxes attribute
            for detection in result.boxes:
                # Ensure detection data tensor is on CPU and converted to numpy array
                data = detection.data.cpu().numpy()
                tracker.update(detection, frame_rate)

                # Extract bounding box coordinates as integers
                x1, y1, x2, y2 = data[0][:4].astype(int)
                # Extract confidence and keep it as float, format it to two decimal places later
                conf = data[0][4]
                # Extract class ID as integer
                cls_id = int(data[0][5])
                class_name = class_names_dict[cls_id]

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            line_height = 30  # Adjust based on font size

            # Display instrument names and speeds
            row = []
            for i, (id, speeds) in enumerate(tracker.speeds.items()):
                class_name1 = class_names_dict[id]
                cv2.putText(frame, f"{class_name1}: {speeds[0]:.2f} pixels/sec", (10, line_height * (i*3 + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name1}_X: {speeds[1]:.2f} pixels/sec", (10, line_height * (i*3 + 1) + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name1}_Y: {speeds[2]:.2f} pixels/sec", (10, line_height * (i*3 + 1) + (2 * line_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                row += [class_name1, f"{speeds[0]:.2f} pixels/sec", f"{speeds[1]:.2f} pixels/sec", f"{speeds[2]:.2f} pixels/sec"]

            row += [''] * (48 - len(row))   # Fill with empty values if fewer than 12 instruments detected

            # PERFORM OPTICAL FLOW
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < (2 / 3 * original_count):
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

                if p0 is not None:
                    original_count = len(p0)
                    good_new = p0
                else:
                    continue
            else:
                p0 = good_new.reshape(-1, 1, 2)

            motion_diff = good_new - good_old
            avg_motion = np.mean(motion_diff, axis=0) if len(motion_diff) > 0 else np.array([0, 0])

            # Counters for red and total points
            red_points = 0
            total_points = len(good_new)

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()

                motion_mag = np.linalg.norm(motion_diff[i] - avg_motion) if len(motion_diff) > 0 else 0

                if motion_mag <= 10:  # Points moving with the camera/environment
                    color = (0, 0, 255)  # Red
                    red_points += 1
                else:
                    color = (0, 255, 0)  # Green

                cv2.circle(frame, (int(a), int(b)), 5, color, -1)

            # Determine if the majority of points are not red
            threshold = 2  # any points that doesn't align with the overall movement of the scene by X pixels
            if red_points / total_points < (1 - unknown_threshold):
                speed_text = "Overall Camera Speed: unknown"
                speed_text_x_axis = "Camera Speed X: unknown"
                speed_text_y_axis = "Camera Speed Y: unknown"
            else:
                estimated_speed = np.mean([np.linalg.norm(motion_diff[i]) for i in range(total_points) if
                                           np.linalg.norm(motion_diff[i] - avg_motion) <= threshold])
                estimated_speed_x = np.mean(
                    [motion_diff[i][0] for i in range(total_points) if np.linalg.norm(motion_diff[i] - avg_motion) <= threshold])
                estimated_speed_y = np.mean(
                    [motion_diff[i][1] for i in range(total_points) if np.linalg.norm(motion_diff[i] - avg_motion) <= threshold])
                # Format the speed text for overall speed, speed in the x-axis, and speed in the y-axis
                speed_text = f"Overall Camera Speed: {estimated_speed * frame_rate:.2f}pixels/sec"
                speed_text_x_axis = f"Camera Speed X: {estimated_speed_x * frame_rate:.2f}pixels/sec"
                speed_text_y_axis = f"Camera Speed Y: {estimated_speed_y * frame_rate:.2f}pixels/sec"

            current_time = datetime.datetime.now()
            row_camera = [frame_count, current_time, speed_text, speed_text_x_axis, speed_text_y_axis]

            start_x = frame_width - 500
            start_y = 50
            line_height = 30  # Adjust based on font size

            cv2.putText(frame, speed_text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, speed_text_x_axis, (start_x, start_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, speed_text_y_axis, (start_x, start_y + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save all values to a csv file
            csv_writer.writerow(row_camera + row)

            old_gray = frame_gray.copy()

        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

