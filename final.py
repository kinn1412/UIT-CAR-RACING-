import cv2
import numpy as np
import time
from simple_pid import PID
from CEEC_Library import GetStatus, GetSeg, AVControl, CloseSocket

# Constants
CHECKPOINTS = [160, 140, 120]
MAX_STEERING_ANGLE = 25
MIN_STEERING_ANGLE = 10
MAX_SPEED = 40
BASE_SPEED = 1
WHITE_PIXEL_VALUE = 255
CENTER_BIAS = 0.6
LINE_WIDTH = 2
LINE_THICKNESS = 2
SLOPE_INPUT_RANGE = (-5, 5)
ANGLE_OUTPUT_RANGE = (-25, 25)

# Kalman Filter initialization
kalman = cv2.KalmanFilter(2, 1)
kalman.measurementMatrix = np.array([[1, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03
kalman.measurementNoiseCov = np.array([[1]], np.float32) * 0.1

# PID controller parameters
KP = 1.3
KI = 0.02
KD = 0.1

pid_controller = PID(KP, KI, KD, setpoint=0)
pid_controller.sample_time = 0.01
pid_controller.output_limits = (-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

def display_image(gray_image, centers, avg_slope):
    h, w = gray_image.shape
    midpoint = w // 2

    cv2.line(gray_image, (centers[0], CHECKPOINTS[0]), (centers[1], CHECKPOINTS[1]), 70, LINE_THICKNESS)
    cv2.line(gray_image, (centers[2], CHECKPOINTS[2]), (centers[1], CHECKPOINTS[1]), 70, LINE_THICKNESS)
    cv2.line(gray_image, (centers[0], CHECKPOINTS[0]), (midpoint, h - 1), 70, LINE_THICKNESS)

    cv2.line(gray_image, (midpoint, 0), (midpoint, h), 70, LINE_THICKNESS)
    
    cv2.imshow('Processed Image', gray_image)

def find_center(gray_image, checkpoint):
    row_data = gray_image[checkpoint, :]
    detected_line = np.where(row_data == WHITE_PIXEL_VALUE)[0]
    
    if len(detected_line) == 0:
        return gray_image.shape[1] // 2

    min_x, max_x = detected_line[0], detected_line[-1]
    if max_x - min_x >= gray_image.shape[1] - 1:
        return gray_image.shape[1] // 2

    return int((min_x + max_x + 1) * CENTER_BIAS)

def map_slope_to_angle(slope_value):
    input_min, input_max = SLOPE_INPUT_RANGE
    angle_min, angle_max = ANGLE_OUTPUT_RANGE
    return (slope_value - input_min) * (angle_max - angle_min) / (input_max - input_min) + angle_min

def derive_angle(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    gray_image = cv2.normalize(gray_image, None, 0, WHITE_PIXEL_VALUE, cv2.NORM_MINMAX)

    height, width = gray_image.shape
    center_positions = np.array([find_center(gray_image, cp) for cp in CHECKPOINTS])

    slopes = [(center - (width // 2)) / (height - CHECKPOINTS[i]) for i, center in enumerate(center_positions) if center != width // 2]
    
    average_slope = np.mean(slopes) if slopes else 0
    steering_angle = map_slope_to_angle(average_slope)

    # Kalman filter to smooth the steering angle
    kalman.correct(np.array([[np.float32(steering_angle)]]))
    predicted_angle = kalman.predict()[0][0]

    display_image(gray_image, center_positions, average_slope)
    return predicted_angle

def adjust_speed(angle):
    computed_speed = MAX_SPEED - abs(angle) * (MAX_SPEED - BASE_SPEED) / MAX_STEERING_ANGLE
    computed_speed = max(BASE_SPEED, computed_speed)
    
    if abs(angle) > MIN_STEERING_ANGLE:
        computed_speed = BASE_SPEED

    print(f"Adjusted Speed: {computed_speed:.2f}")
    return computed_speed

if __name__ == "__main__":
    try:
        while True:
            time.sleep(0.02)

            current_state = GetStatus()
            segmented_image = GetSeg()

            if segmented_image is None:
                print("Failed to retrieve the segmented image.")
                continue

            angle = derive_angle(segmented_image)
            if angle is None:
                AVControl(speed=BASE_SPEED, angle=0)
                continue

            pid_output = pid_controller(-angle)
            print(f"PID Steering Output: {pid_output:.2f} degrees")
            current_speed = adjust_speed(angle)

            AVControl(speed=current_speed, angle=pid_output)

            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        CloseSocket()
