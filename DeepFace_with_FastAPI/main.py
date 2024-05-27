from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates
from deepface import DeepFace
import os
import cv2
from fastapi import Request

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/verify")
async def verify(file: UploadFile= File(...)):
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    reference_images_dir = "reference_images"
    results = []

    try:
        for ref_image in os.listdir(reference_images_dir):
            ref_image_path = os.path.join(reference_images_dir, ref_image)
            result = DeepFace.verify(img1_path= file_location, img2_path= ref_image_path)
            if result["verified"]:
                results.append({"filename": ref_image})

        if not results:
            return JSONResponse(content={"message": "No Match Found"}, status_code=404)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    finally:
        os.remove(file_location)

@app.post("/add_reference_image/")
async def add_reference_image(name: str = Form(...), file: UploadFile=File(...)):
    file_location = f"reference_images/{name}.jpg"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    return JSONResponse(content={"message": "Reference image added successfully", "filename": f"{name}.jpg"})

@app.post("/real_time_register")
async def real_time_register(name: str = Form(...)):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        return JSONResponse(content={"error": "Could not open webcam"}, status_code=500)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            return JSONResponse(content={"error": "Failed to capture image"}, status_code=500)

        cv2.imshow("Register - press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            file_location = f"reference_images/{name}.jpg"
            cv2.imwrite(file_location, frame)
            cv2.waitKey(1)
            break
    
    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()
    # need to have waitkey again after destroying all windows
    cv2.waitKey(1)
    return JSONResponse(content={"message": "Reference image added successfully", "filename": f"{name}.jpg"})

@app.post("/real_time_verify/")
async def real_time_verify():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        return JSONResponse(content={"error": "Could not open webcam"}, status_code=500)
    
    reference_images_dir = "reference_images"
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            return JSONResponse(content={"error": "Failed to capture image"}, status_code=500)

        cv2.imshow("Verify - Press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            temp_image_path = "temp/temp_frame.jpg"
            cv2.imwrite(temp_image_path, frame)
            break
    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()
    # need to have waitkey again after destroying all windows
    cv2.waitKey(1)
    
    try:
        for ref_image in os.listdir(reference_images_dir):
            if ref_image == ".DS_Store":
                continue
            ref_image_path = os.path.join(reference_images_dir, ref_image)
            result = DeepFace.verify(img1_path=temp_image_path, img2_path=ref_image_path)  
            if result["verified"]:
                results.append({"filename": ref_image})
        
        os.remove(temp_image_path)

        if not results:
            return JSONResponse(content={"message": "No match found"}, status_code=404)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    
    