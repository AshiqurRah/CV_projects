from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from deepface import DeepFace
import os
import cv2
import pickle
import numpy as np
from deepface.modules.verification import find_cosine_distance, find_threshold
from deepface.modules import modeling, preprocessing
from deepface.extendedmodels import Gender, Race

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
            if cv2.waitKey(10) & 0xFF == ord('q'):
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
async def real_time_detection(level: int):
    cap = cv2.VideoCapture(1)
    # get preloaded embeddings
    embeddings, names = await get_embeddings()

    if level == 2:
       # Build model here to prevent re-downloaded of pretrained weights
       age_model = modeling.build_model("Age")
       gender_model = modeling.build_model("Gender")
       race_model = modeling.build_model("Race")

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

                if level == 1:
                
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
                    
                
                elif level == 2:
                    color = (0,255,0)

                    img_content = preprocessing.resize_image(img=detection["face"], target_size=(224,224))

                    # To identify age
                    apparent_age = age_model.predict(img_content)
                    age = int(apparent_age)
                    
                    # To identify gender
                    gender_predictions = gender_model.predict(img_content)
                    gender = Gender.labels[np.argmax(gender_predictions)]

                    # To identify race
                    race_predictions  = race_model.predict(img_content)
                    race = Race.labels[np.argmax(race_predictions)]
                    
                    text = f"Age: {age},Gender: {gender}, Race: {race}"
                    
                else:
                    color = (0,255,0)
                    text = "Person"
                
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
            
            cv2.imshow("Real-Time Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # number: #pax in screen
                number = len(detections) if level == 3 and detections[0]["confidence"] != 0 else 0
                break
    finally:
        # Release the webcam
        cap.release()
        # Close the image show frame
        cv2.destroyAllWindows()
        # need to have waitkey again after destroying all windows
        cv2.waitKey(1)
    if level == 3:
        return JSONResponse(content={"message": f"{number} pax in frame", "status": "stopped"})
    return JSONResponse(content={"status": "stopped"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
