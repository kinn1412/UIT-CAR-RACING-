from CEEC_Library import GetStatus, GetRaw, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import math

CHECKPOINT = 150
MAX_ANGLE = 40  # Tăng giới hạn góc lái để xe có thể bẻ cua tốt hơn
MAX_SPEED = 50  # Giới hạn tốc độ tối đa

# Hiển thị đường thẳng trung tâm để debug
def showImage(gray_show, center_row):
    h, w = gray_show.shape
    gray_show = cv2.line(gray_show, (center_row, CHECKPOINT), (int(w/2), h-1), 90, 2)
    gray_show = cv2.line(gray_show, (int(w/2), CHECKPOINT), (int(w/2), h-1), 90, 2)
    gray_show = cv2.line(gray_show, (int(w/2), CHECKPOINT), (center_row, CHECKPOINT), 90, 2)
    cv2.imshow('test', gray_show)

# Tính toán góc lái từ hình ảnh segment
def AngCal(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray*(255/np.max(gray))).astype(np.uint8)
    gray_show = gray.copy()

    h, w = gray.shape

    # Lấy dòng tại điểm CHECKPOINT
    line_row = gray[CHECKPOINT, :]
    line = np.where(line_row == 255)[0]

    if len(line) == 0:  # Không có đường phát hiện
        return 0

    min_x = line[0]
    max_x = line[-1]
    center_row = int((max_x + min_x + 1) / 2)

    showImage(gray, center_row)

    # Xử lý ngã tư nếu phát hiện cả chiều rộng ảnh đều là đường
    if max_x - min_x == 319:  # Ngã tư, cần xử lý riêng
        return 0

    # Tính toán góc lái
    x0, y0 = int(w / 2), h
    x1, y1 = center_row, CHECKPOINT
    value = (x1 - x0) / (y0 - y1)
    angle = math.degrees(math.atan(value)) / 3

    print("Value: {} --- Angle: {}".format(value, angle))

    # Giới hạn góc lái trong khoảng -MAX_ANGLE đến MAX_ANGLE
    angle = np.clip(angle, -MAX_ANGLE, MAX_ANGLE)

    return angle

# Điều chỉnh tốc độ dựa trên góc lái
def adjust_speed(angle):
    if abs(angle) > 30:  # Nếu góc lái lớn, giảm tốc mạnh hơn
        return 20
    elif abs(angle) > 15:  # Nếu góc lái trung bình, giảm tốc vừa phải
        return 35
    return MAX_SPEED  # Tốc độ tối đa khi đi thẳng

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()  # Lấy trạng thái của xe
            raw_image = GetRaw()  # Lấy ảnh raw từ camera
            segment_image = GetSeg()  # Lấy ảnh segment

            # Tính toán góc lái từ ảnh segment
            angle = AngCal(segment_image)

            # Điều chỉnh tốc độ theo góc lái
            speed = adjust_speed(angle)

            # Điều khiển xe với tốc độ và góc lái đã tính
            AVControl(speed=speed, angle=angle)

            # Thoát vòng lặp khi nhấn phím 'q'
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        CloseSocket()  # Đóng socket khi kết thúc
