# Real Time Face Detection v2

Real-Time Face Detection v2 is a cutting-edge solution leveraging the DeepFace framework and FastAPI for accurate and efficient face detection and recognition. 
Designed with privacy and customization in mind, this project offers three distinct levels of face detection, catering to various use cases and privacy requirements.


## Table of Contents

- [Key Features](#key-features)
- [Privacy and Customization](#Privacy-and-Customization)
- [Installation](#Installation)
- [Usage](#usage)



## Key Features:
### 1. Multi-Level Face Detection:

- High Level: Perform face detection with registered users, displaying their names for seamless recognition.
- Medium Level: Conduct face detection without user registration, providing insights into age, gender, and race.
- Low Level: Detect faces and provide a count of people present in the frame, ensuring privacy while still providing valuable information.

### 2. Advanced Recognition Techniques:

- Utilizes the state-of-the-art Facenet512 model for recognition and verification.
- Adopts the cosine similarity metric for accurate comparison, ensuring reliable results.

### 3. Efficient Data Storage:
- Stores face data in Pickle files, ensuring fast retrieval and minimal resource consumption.
- Enables seamless integration with existing systems and workflows.

## Privacy and Customization
Real-Time Face Detection v2 prioritizes privacy and customization, allowing users to tailor the detection process according to their specific needs. With multiple levels of detection, users can choose the appropriate level of detail while safeguarding individual privacy.



## Installation

1. Clone the repository:

```bash
 git clone https://github.com/Ashiqurrah/CV_projects.git
```

2. Change directory of root

```bash
cd RealTime_FaceDetection_v2
```

3. Create a new python virtual env [Recommended step]

```bash
python -m venv name_of_your_environment
```

Activate the virtual env
```bash
source name_of_your_environment/bin/activate
```

4. Install dependencies:

```bash
 pip install -r requirements.txt
 ```

In some cases, you might encounter an issue with regards to the installation of wrapt and h5py packages. Ensure that your root virtual environment has already installed these 2 packages.
  ```bash
  pip install wrapt
  pip install h5py
  ```

5. Create the ```embeddings``` folder to store embeddings in pkl format:

```bash
mkdir embeddings
```

## Usage
To run the project, use the following command:
```bash
uvicorn main:app --reload
```
