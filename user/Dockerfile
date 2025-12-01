FROM python:3.11-slim

EXPOSE 8084

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cached-dir -r requirements.txt

COPY . ./

ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8084", "--server.address=0.0.0.0" ]