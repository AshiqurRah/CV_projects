from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates
from deepface import DeepFace
import os
import cv2
import pickle
from fastapi import Request
from deepface.modules.verification import find_cosine_distance, find_threshold

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/add_reference_image/")
async def add_reference_image(name: str = Form(...), file: UploadFile=File(...)):
    file_location = f"reference_images/{name}.jpg"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    return JSONResponse(content={"message": "Reference image added successfully", "filename": f"{name}.jpg"})

@app.post("/real_time_register")
async def real_time_register(name: str = Form(...)):
    # change paramter for VideoCapture, mostly it is 0 or 1. Depends on the number of camera devices connected to your comp
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        return JSONResponse(content={"error": "Could not open webcam"}, status_code=500)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            return JSONResponse(content={"error": "Failed to capture image"}, status_code=500)

        cv2.imshow("Register - press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            embedding = DeepFace.represent(
                img_path = frame,
                model_name="Facenet512"
            )[0]['embedding']
            embedding_path = os.path.join("embeddings", f"{name}.pkl")
            with open(embedding_path, 'wb') as file:
                pickle.dump(embedding, file)
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            return JSONResponse(content={"error": "Failed to capture image"}, status_code=500)

        cv2.imshow("Verify - Press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()
    # need to have waitkey again after destroying all windows
    cv2.waitKey(1)
    
    # Compute embedding of the captured image using the default DeepFace model (VGG-Face)
    try:
        embedding = DeepFace.represent(frame, model_name="Facenet512")[0]['embedding']
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
    # find best_match and best_distance for checking
    best_match = False
    best_distance = False

    for filename in os.listdir("embeddings"):
        if filename.endswith(".pkl"):
            with open(os.path.join("embeddings", filename), 'rb') as file:
                stored_embedding = pickle.load(file)

            distance = find_cosine_distance(embedding, stored_embedding)
            threshold = find_threshold("Facenet512", 'cosine')

            if distance <= threshold:
                best_match = filename.split(".")[0]
                best_distance = distance

    if best_match:  # Adjust threshold as necessary
        return {"message": f"{best_match} verified successfully", "distance": best_distance, "threshold": threshold}
    else:
        return {"message": "Verification failed, no such person"}

    
    
