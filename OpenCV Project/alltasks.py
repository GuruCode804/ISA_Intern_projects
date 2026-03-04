import cv2
import numpy as np


# ---------------------------------------
# TASK 1: Landing Pad Finder
# ---------------------------------------
def landing_pad_finder(img_path):
    print("\n--- Task 1: Landing Pad Finder ---")

    img = cv2.imread(img_path)
    if img is None:
        print("Image not loaded!")
        return

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=200
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        largest_circle = circles[np.argmax(circles[:, 2])]
        x, y, r = largest_circle

        cx = np.clip(x, 15, w - 15)
        cy = np.clip(y, 15, h - 15)

        cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
        cv2.line(img, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
        cv2.line(img, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)

    cv2.imshow("Task 1", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Task 1")
    cv2.waitKey(1)


# ---------------------------------------
# TASK 2: Horizon Leveler
# ---------------------------------------
def horizon_leveler(img_path, angle):
    print("\n--- Task 2: Horizon Leveler ---")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (w, h))

    crop_x = int(w * 0.1)
    crop_y = int(h * 0.1)
    cropped = rotated[crop_y:h - crop_y, crop_x:w - crop_x]

    cv2.imshow("Task 2", cropped)
    cv2.waitKey(0)
    cv2.destroyWindow("Task 2")
    cv2.waitKey(1)


# ---------------------------------------
# TASK 3: Obstacle Alert
# ---------------------------------------
def obstacle_alert(img_path):
    print("\n--- Task 3: Obstacle Alert ---")

    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    danger_pixels = np.count_nonzero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    danger_percentage = (danger_pixels / total_pixels) * 100

    print(f"Danger Coverage: {danger_percentage:.2f}%")

    if danger_percentage > 10:
        print("🚨 DANGER: OBSTACLE DETECTED!")

    highlighted = img.copy()
    highlighted[mask != 0] = [0, 0, 255]

    cv2.imshow("Task 3", highlighted)
    cv2.waitKey(0)
    cv2.destroyWindow("Task 3")
    cv2.waitKey(1)


# ---------------------------------------
# TASK 4: Night Vision
# ---------------------------------------
def night_vision(img_path):
    print("\n--- Task 4: Night Vision Booster ---")

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    green_night = np.zeros_like(img)
    green_night[:, :, 1] = enhanced_gray

    cv2.imshow("Task 4", green_night)
    cv2.waitKey(0)
    cv2.destroyWindow("Task 4")
    cv2.waitKey(1)


# ---------------------------------------
# TASK 5: Motion Blur
# ---------------------------------------
def motion_blur(img_path):
    print("\n--- Task 5: Motion Blur ---")

    img = cv2.imread(img_path)

    kernel = np.zeros((1, 15))
    kernel[0, :] = 1 / 15

    blurred = cv2.filter2D(img, -1, kernel)

    cv2.imshow("Task 5", blurred)
    cv2.waitKey(0)
    cv2.destroyWindow("Task 5")
    cv2.waitKey(1)


# ---------------------------------------
# MAIN EXECUTION
# ---------------------------------------
if __name__ == "__main__":

    landing_pad_finder("images/landing_pad.jpg")
    horizon_leveler("images/horizon.jpg", 15)
    obstacle_alert("images/obstacle.jpg")
    night_vision("images/night.jpg")
    motion_blur("images/motion.jpg")

    print("\n✅ All tasks completed successfully.")