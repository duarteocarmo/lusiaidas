FROM python:3.9

WORKDIR /app

COPY app.py . 
COPY assets/index.html ./assets/index.html

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]