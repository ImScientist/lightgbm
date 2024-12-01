FROM python:3.12.4-bookworm

WORKDIR /main

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV PYTHONPATH=/main:/main/src

ENTRYPOINT ["python", "src/main.py"]
