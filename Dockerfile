FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r ig_manager/requirements.txt

CMD ["tail", "-f", "/dev/null"]
