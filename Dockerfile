FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y
# install gcc along with requirements as python slim buster is removing it to be lightweight
RUN apt-get update &&  apt-get install -y gcc && pip install -r requirements.txt
CMD ["python3", "app.py"]