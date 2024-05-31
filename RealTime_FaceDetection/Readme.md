## Real Time Face Detection

Face detection and recognition using DeepFace framework and FastAPI. Stores data in Pickle file. Recognition and Verification is done using ```Facenet512``` model and similarity metric adopted is ```cosine similarity```.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)


## Installation

1. Clone the repository:

```bash
 git clone https://github.com/Ashiqurrah/CV_projects.git
```

2. Change directory of root

```bash
cd RealTime_FaceDetection
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
