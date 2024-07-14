# Flask App

This is a simple Flask application with setup instructions for a Conda environment.

## 1. Conda Environment

To set up the Conda environment for this project, follow these steps:

1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already.

2. Create a new Conda environment:
   ```
   conda create -p translation python=3.12
   ```

3. Activate the environment:
   ```
   conda activate translation
   ```

## 2. app.py

The `app.py` file contains the main Flask application code. Here's a basic structure of what it might look like:

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

To run the application:

```
python app.py
```

The app will be available at `http://localhost:5000`.

## 3. requirements.txt

The `requirements.txt` file lists all the Python packages required for this project. To install these dependencies, run:

```
pip install -r requirements.txt
```


## Getting Started

1. Set up and activate the Conda environment as described in section 1.
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Flask app: `python app.py`
4. Open a web browser and go to `http://localhost:5000`

