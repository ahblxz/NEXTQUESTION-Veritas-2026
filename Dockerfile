# Reproducible, self-contained NEXTQUESTION synthetic research demonstration.
# This is NOT a clinical or production hospital diagnostic device.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY hospital/ hospital/
COPY hospital_train.py app.py ./
RUN python hospital_train.py --out artifacts/hospital --n 4800
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=8s --start-period=25s --retries=3 \
 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/',timeout=5)" || exit 1
CMD ["python", "app.py"]
