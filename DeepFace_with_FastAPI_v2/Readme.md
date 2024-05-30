## DeepFace with FastAPI v2

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
cd DeepFace_with_FastAPI_v2
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

5. Create the ```embeddings``` folder to store embeddings in pkl format:

```bash
mkdir embeddings
```

## Usage
To run the project, use the following command:
```bash
uvicorn main:app --reload
```
