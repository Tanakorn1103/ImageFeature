FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Install libGL for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app
COPY ./CarsDataset/test /app/CarsDataset/test