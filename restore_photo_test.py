import cv2
from ultralytics import YOLO
import mediapipe as mp

def is_full_body(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("No person detected in the image.")
        return False

    landmarks = results.pose_landmarks.landmark

    try:
        left_ankle_visible = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].visibility > 0.5
        right_ankle_visible = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility > 0.5
    except Exception as e:
        print(f"Error getting landmarks: {e}")
        return False
    
    pose.close()

    if left_ankle_visible or right_ankle_visible:
        return True
    else:
        return False

def modify_image(image):
    # Простое изменение цвета для тестирования
    modified_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return modified_image

def insert_modified_image(original_image, modified_image, x1, y1):
    # Вставляет модифицированное изображение обратно в оригинал
    original_image[y1:y1 + modified_image.shape[0], x1:x1 + modified_image.shape[1]] = modified_image
    return original_image

def crop_person(image_path, base_crop = True, crop_ratio=0.2):
    try:
        model = YOLO("yolo11n.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    results = model(image)

    persons = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls)
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append((x1, y1, x2, y2))

    if not persons:
        print("No person detected in the image.")
        return None

    max_area = 0
    max_person_bbox = None
    for x1, y1, x2, y2 in persons:
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            max_person_bbox = (x1, y1, x2, y2)

    if max_person_bbox:
        x1, y1, x2, y2 = max_person_bbox
        height = y2 - y1
        
        full_body = is_full_body(image_path)

        if not full_body or base_crop:
            crop_y2 = int(y2 + (y2 - y1) * crop_ratio)
            crop_y2 = min(crop_y2, image.shape[0])
            cropped_image = image[y1:crop_y2, x1:x2]
        else:
            knee_level = int(y1 + height * 0.65)
            cropped_image = image[y1:knee_level, x1:x2]

        # Возвращаем координаты и обрезанное изображение
        return cropped_image, (x1, y1, x2, y2)
    else:
        return None, None

if __name__ == "__main__":
    image_path = "test_imgs\jon-performance-pants-5-pocket-navy-performance-pants-velland-370041.jpg"
    base_crop = False
    crop_ratio = 0.2
    cropped_image, coords = crop_person(image_path, base_crop, crop_ratio)
    if cropped_image is not None and coords is not None:
        x1, y1, _, _ = coords
        modified_image = modify_image(cropped_image)
        original_image = cv2.imread(image_path)
        final_image = insert_modified_image(original_image, modified_image, x1, y1)
        
        cv2.namedWindow("Modified Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Modified Image", final_image)
        cv2.imwrite("modified_" + image_path, final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No person detected in the image.")
