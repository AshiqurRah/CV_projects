## DeepFace with FastAPI v1

Face detection and recognition using DeepFace framework and FastAPI. Stores data in image form. Recognition and Verification is done using ```VGG-Face``` model and similarity metric adopted is ```cosine similarity```.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


## Installation

1. Clone the repository:

```bash
 git clone https://github.com/Ashiqurrah/CV_projects.git
```

2. Change directory of root

```bash
cd DeepFace_with_FastAPI
```

3. Create a new python virtual env [Optional step]

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

5. Create the ```temp``` and ```reference_images``` folder to store images:

```bash
mkdir temp reference_images
```


## Usage
To run the project, use the following command:
```bash
uvicorn main:app --reload
```
