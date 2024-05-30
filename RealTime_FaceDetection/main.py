from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from deepface import DeepFace
import os
import cv2
import pickle
import time
from deepface.modules.verification import find_cosine_distance, find_threshold

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

async def get_embeddings():
    # Load embeddings and names
    embeddings = []
    names = []

    embedding_dir = "embeddings"
    for file_name in os.listdir(embedding_dir):
        if file_name.endswith(".pkl"):
            with open(os.path.join(embedding_dir, file_name), "rb") as file:
                data = pickle.load(file)
                embeddings.append(data)
                names.append(file_name.split('.')[0])
    return embeddings, names

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/real_time_register/")
async def real_time_register(name: str = Form(...)):
    # change paramter for VideoCapture, mostly it is 0 or 1. Depends on the number of camera devices connected to your comp
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        return JSONResponse(content={"error": "Could not open webcam"}, status_code=500)
    try:
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
    finally:
        # Release the webcam
        cap.release()
        # Close the image show frame
        cv2.destroyAllWindows()
        # need to have waitkey again after destroying all windows
        cv2.waitKey(1)

    return JSONResponse(content={"message": "Reference data added successfully", "filename": f"{name}.pkl"})

@app.get("/real_time_detection")
async def real_time_detection():
    cap = cv2.VideoCapture(1)
    # To give time for the webcam to start
    time.sleep(2)
    embeddings, names = await get_embeddings()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # detect faces and compute embeddings
            detections = DeepFace.extract_faces(frame, enforce_detection=False)

            threshold = find_threshold("Facenet512", 'cosine')
            for detection in detections:
                # Skip detections with no faces
                if detection["confidence"] == 0:
                    continue
                x, y, w, h, _, _ = detection["facial_area"].values()
                face = rgb_frame[y: y+h, x: x+w]
                try:
                    face_embedding = DeepFace.represent(img_path=face, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                except ValueError as e:
                    print(f"Embedding error: {e}")
                    continue
                
                match_found = False
                
                for stored_embedding,name in zip(embeddings, names):
                    distance = find_cosine_distance(face_embedding, stored_embedding)
                    if distance <= threshold:
                        # green for match
                        color = (0,255,0)
                        text = name
                        match_found = True
                        break
                
                if not match_found:
                    # red for no match
                    color = (0,0,255) 
                    text = 'Unknown'
                
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            cv2.imshow("Real-Time Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the webcam
        cap.release()
        # Close the image show frame
        cv2.destroyAllWindows()
        # need to have waitkey again after destroying all windows
        cv2.waitKey(1)
    
    return JSONResponse(content={"status": "stopped"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)