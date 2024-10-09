from CEEC_Library import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import time

# Cấu hình thông số
class Config:
    UPPER_CHECKPOINT = 160
    MID_CHECKPOINT = 140
    LOWER_CHECKPOINT = 120
    MAX_STEERING_ANGLE = 25
    SLOW_DOWN_ANGLE = 10
    MAX_VEHICLE_SPEED = 28  # Giảm tốc độ tối đa xuống còn 28
    MIN_VEHICLE_SPEED = 1
    WHITE_PIXEL_VALUE = 255
    CENTER_BIAS = 0.6
    LINE_WIDTH = 2
    ANGLE_INPUT_RANGE = (-5, 5)
    ANGLE_OUTPUT_RANGE = (-25, 25)

# PID constants
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, target_value, current_value):
        error = target_value - current_value
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

# Khởi tạo PID controller
pid_controller = PIDController(1.3, 0.02, 0.1)

def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray_image * (Config.WHITE_PIXEL_VALUE / np.max(gray_image))).astype(np.uint8)

def find_line_centers(image, checkpoints):
    centers = []
    for cp in checkpoints:
        pixel_row = image[cp, :]
        white_pixels = np.where(pixel_row == Config.WHITE_PIXEL_VALUE)[0]
        if len(white_pixels) == 0:
            centers.append(int(image.shape[1] / 2))
        else:
            min_x, max_x = white_pixels[0], white_pixels[-1]
            if max_x - min_x >= image.shape[1] - 1:
                centers.append(int(image.shape[1] / 2))
            else:
                centers.append(int((min_x + max_x + 1) * Config.CENTER_BIAS))
    return centers

def calculate_weighted_slope(center_positions, image_width, image_height):
    weights = [0.5, 0.3, 0.2]  # Trọng số cho các checkpoint
    weighted_slopes = []
    midpoint = int(image_width / 2)

    for center, checkpoint, weight in zip(center_positions, [Config.UPPER_CHECKPOINT, Config.MID_CHECKPOINT, Config.LOWER_CHECKPOINT], weights):
        slope = (center - midpoint) / (image_height - checkpoint)
        weighted_slopes.append(slope * weight)
    
    return sum(weighted_slopes)

def compute_steering_angle(image):
    global image_height, image_width
    gray_image = process_image(image)
    image_height, image_width = gray_image.shape
    checkpoints = [Config.UPPER_CHECKPOINT, Config.MID_CHECKPOINT, Config.LOWER_CHECKPOINT]
    center_positions = find_line_centers(gray_image, checkpoints)

    avg_slope = calculate_weighted_slope(center_positions, image_width, image_height)
    steering_angle = (avg_slope - Config.ANGLE_INPUT_RANGE[0]) * (Config.ANGLE_OUTPUT_RANGE[1] - Config.ANGLE_OUTPUT_RANGE[0]) / \
                     (Config.ANGLE_INPUT_RANGE[1] - Config.ANGLE_INPUT_RANGE[0]) + Config.ANGLE_OUTPUT_RANGE[0]

    display_image(gray_image, center_positions, avg_slope)
    return steering_angle

def compute_vehicle_speed(steering_angle, last_angle):
    if abs(steering_angle - last_angle) > Config.SLOW_DOWN_ANGLE:
        return Config.MIN_VEHICLE_SPEED
    adjusted_speed = Config.MAX_VEHICLE_SPEED - abs(steering_angle) * (Config.MAX_VEHICLE_SPEED - Config.MIN_VEHICLE_SPEED) / Config.MAX_STEERING_ANGLE
    return max(Config.MIN_VEHICLE_SPEED, adjusted_speed)

def display_image(processed_image, center_positions, avg_slope):
    height, width = processed_image.shape

    for checkpoint, center in zip([Config.UPPER_CHECKPOINT, Config.MID_CHECKPOINT, Config.LOWER_CHECKPOINT], center_positions):
        cv2.line(processed_image, (0, checkpoint), (width, checkpoint), 100, Config.LINE_WIDTH)
        cv2.circle(processed_image, (center, checkpoint), 5, 200, -1)

    # Vẽ đường dự đoán góc lái
    angle_end_x = int(width / 2 + (height - Config.UPPER_CHECKPOINT) * avg_slope)
    cv2.line(processed_image, (int(width / 2), height), (angle_end_x, Config.UPPER_CHECKPOINT), 50, Config.LINE_WIDTH + 3)

    cv2.imshow('Big Diggers View', cv2.resize(processed_image, (640, 480)))

if __name__ == "__main__":
    try:
        previous_angle = 0
        while True:
            time.sleep(0.02)

            status = GetStatus()
            segmented_image = GetSeg()

            if segmented_image is None:
                print("Không thể nhận diện hình ảnh phân đoạn.")
                continue

            steering_angle = compute_steering_angle(segmented_image)
            if steering_angle is None:
                AVControl(speed=Config.MIN_VEHICLE_SPEED, angle=0)
                continue

            pid_adjusted_angle = pid_controller.compute(0, -steering_angle)
            print(f"PID adjusted angle: {pid_adjusted_angle:.2f} degrees")
            vehicle_speed = compute_vehicle_speed(steering_angle, previous_angle)
            previous_angle = steering_angle
            
            AVControl(speed=vehicle_speed, angle=pid_adjusted_angle)

            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        CloseSocket()
