# main.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import io
import tempfile
import base64
from PIL import Image
import time

# Import the utility functions (adjust the path if needed)
from util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
)

# Set up device and load the models
device = 'cuda'
model_path = 'weights/icon_detect/model.pt'
print("Loading YOLO model from:", model_path)
som_model = get_yolo_model(model_path)
som_model.to(device)
print(f"YOLO model loaded on {device}")

# Initialize caption model processor (using florence2 as an example)
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence",
    device=device
)
print("Caption model processor loaded.")

# Create the FastAPI app
app = FastAPI(title="OmniParser API")

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    """
    Receives an image file, processes it through OmniParser, and returns:
    - A base64-encoded image (with UI element annotations)
    - A parsed content list (list of dicts containing parsed information)
    """
    # Read uploaded file content
    contents = await file.read()
    
    # Open image with PIL for obtaining dimensions
    image = Image.open(io.BytesIO(contents))
    image_rgb = image.convert('RGB')
    width, height = image.size
    print("Received image size:", image.size)

    # Compute drawing parameters based on image size
    box_overlay_ratio = max(image.size) / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    BOX_THRESHOLD = 0.05

    # Save the image temporarily since our utility functions expect a file path.
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(contents)
        tmp.flush()  # ensure the data is written

        # Run OCR box extraction.
        # Note: Adjust the parameters as needed.
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            tmp.name,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_bbox_rslt

        # Process the image through the YOLO-based model and caption processor.
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            tmp.name,
            som_model,
            BOX_TRESHOLD=BOX_THRESHOLD,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128
        )

    # Return the annotated image (base64 string) and parsed content list.
    return {
        "annotated_image": dino_labeled_img,
        "parsed_content_list": parsed_content_list
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
