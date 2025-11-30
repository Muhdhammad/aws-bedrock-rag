FROM python:3.11-slim

EXPOSE 8088

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cached-dir -r requirements.txt

COPY . ./

ENTRYPOINT [ "streamlit", "run", "main.py", "--server.port=8088", "--server.address=0.0.0.0" ]