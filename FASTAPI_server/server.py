from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
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

        background_tasks.add_task(process_files, task_id, user_photo_path, product_image_path, product_description)

        return JSONResponse(content={"task_id": task_id, "message": "Files loading and started processing."})

    except Exception as e:
        logging.error("Loading files server error: %s", e)
        return JSONResponse(content={"error": "Loading files server error."}, status_code=500)

async def process_files(task_id: str, user_photo_path: str, product_image_path: str,
                        product_description: str):
    processing_results[task_id] = {'status': 'processing'}

    try:
        if not os.path.exists(user_photo_path):
            processing_results[task_id] = {'status': 'error', 'message': f"File {user_photo_path} not found."}
            return
        
        if not os.path.exists(product_image_path):
            processing_results[task_id] = {'status': 'error', 'message': f"File {product_image_path} not found."}
            return

        # result_gradio = await asyncio.to_thread(gradio_client.predict,
        #     dict={"background": file(user_photo_path)},
        #     garm_img=file(product_image_path),
        #     garment_des=product_description,
        #     is_checked=True,
        #     is_checked_crop=False,
        #     denoise_steps=30,
        #     seed=42,
        #     api_name="/tryon"
        # )
        result_gradio = await asyncio.to_thread(gradio_client.predict,
		      src_image_path=handle_file(user_photo_path),
		      ref_image_path=handle_file(product_image_path),
		      ref_acceleration=True,
		      step=35,
		      scale=2.5,
		      seed=42,
		      vt_model_type="viton_hd",
		      vt_garment_type="upper_body",
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
    