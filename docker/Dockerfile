FROM python:3.12-slim

RUN pip install --no-cache-dir \
    fastapi uvicorn pandas numpy matplotlib seaborn scipy scikit-learn statsmodels

COPY server.py /app/server.py
WORKDIR /app

EXPOSE 8765
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765"]
