from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from logger import setup_logger
import json
import os
import uuid
from gradio_client import Client, handle_file
import asyncio
import logging
from config import Config
import base64
import cv2
from ultralytics import YOLO
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

processing_results = {}
app = FastAPI()
config = Config()
setup_logger()

gradio_client = Client(config.model_name, hf_token=config.model_name)

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
    
def crop_person(image_path, crop_ratio=0.2):
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

        if not full_body:
            crop_y2 = int(y2 + (y2 - y1) * crop_ratio)
            crop_y2 = min(crop_y2, image.shape[0])
            cropped_image = image[y1:crop_y2, x1:x2]
        else:
            knee_level = int(y1 + height * 0.65)
            cropped_image = image[y1:knee_level, x1:x2]
            
        return cropped_image
    else:
        return None

@app.get("/")
async def read_root():
    return {"message": "Hello, API Server"}

@app.get("/products/")
async def get_products():
    """Fetch products from the JSON file."""
    try:
        with open(config.js_data_url, 'r', encoding='utf-8') as f:
            products = json.load(f)
        return JSONResponse(content=products)
    except Exception as e:
        return JSONResponse(content={"error": "Error with loading product from json."}, status_code=500)

@app.post("/upload/")
async def upload_file(
    user_photo: str = Form(...),
    user_photo_extension: str = Form(...),
    product_image: str = Form(...),
    product_image_extension: str = Form(...),
    current_index: int = Form(...),
    product_description: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    task_id = str(uuid.uuid4())

    try:
        user_photo_bytes = base64.b64decode(user_photo)
        product_image_bytes = base64.b64decode(product_image)
        
        user_photo_path = os.path.join(UPLOAD_DIR, f"{task_id}_user_photo{user_photo_extension}")
        product_image_path = os.path.join(UPLOAD_DIR, f"{task_id}_product_image{product_image_extension}")

        with open(user_photo_path, "wb") as f:
            f.write(user_photo_bytes)

        with open(product_image_path, "wb") as f:
            f.write(product_image_bytes)

        background_tasks.add_task(process_files, task_id, user_photo_path, product_image_path, product_description, current_index)

        return JSONResponse(content={"task_id": task_id, "message": "Files loading and started processing."})

    except Exception as e:
        logging.error("Loading files server error: %s", e)
        return JSONResponse(content={"error": "Loading files server error."}, status_code=500)

async def process_files(task_id: str, user_photo_path: str, product_image_path: str,
                        product_description: str, current_index: int):
    processing_results[task_id] = {'status': 'processing'}

    try:
        if not os.path.exists(user_photo_path):
            processing_results[task_id] = {'status': 'error', 'message': f"File {user_photo_path} not found."}
            return
        
        if not os.path.exists(product_image_path):
            processing_results[task_id] = {'status': 'error', 'message': f"File {product_image_path} not found."}
            return

        cropped_image = crop_person(user_photo_path)

        with open(config.js_data_url, 'r', encoding='utf-8') as f:
          products = json.load(f)
          product_id = products[current_index]['cloth_type']
          if product_id == "Верх":
            print("Верх")
            result_gradio = await asyncio.to_thread(gradio_client.predict,
		          src_image_path=handle_file(cropped_image),
		          ref_image_path=handle_file(product_image_path),
		          ref_acceleration=True,
		          step=30,
		          scale=2.5,
		          seed=42,
		          vt_model_type="viton_hd",
		          vt_garment_type="upper_body",
		          api_name="/leffa_predict_vt"
            )
          elif product_id == "Низ":
            print("НИЗ")
            result_gradio = await asyncio.to_thread(gradio_client.predict,
		          src_image_path=handle_file(user_photo_path),
		          ref_image_path=handle_file(product_image_path),
		          ref_acceleration=True,
		          step=30,
		          scale=2.5,
		          seed=42,
		          vt_model_type="dress_code",
		          vt_garment_type="lower_body",
		          api_name="/leffa_predict_vt"
            )
        
        
        image_path = result_gradio[0]
        with open(image_path, "rb") as img_file:
            processed_image_data = img_file.read()

        processed_image_extension = os.path.splitext(image_path)[1]
        processed_image_base64 = base64.b64encode(processed_image_data).decode('utf-8')

        final_processed_image_path = os.path.join(PROCESSED_DIR, f"{task_id}_result{processed_image_extension}")
        with open(final_processed_image_path, "wb") as f:
            f.write(processed_image_data)

        processing_results[task_id] = {'status': 'completed', 'result': processed_image_base64}

    except Exception as e:
        logging.error("Error during processing for task %s: %s", task_id, e)
        processing_results[task_id] = {'status': 'error', 'message': "Error with processing images."}

@app.get("/status/{task_id}") 
async def get_status(task_id: str): 
    if task_id in processing_results: 
        return JSONResponse(
            content={"status": processing_results[task_id]['status'], 
                     "result": processing_results[task_id].get('result') }) 
    else: 
        return JSONResponse(content={"status": "not found"}, status_code=404)
    