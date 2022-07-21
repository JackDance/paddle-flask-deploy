FROM paddlepaddle/paddle:2.3.0

COPY requirements.txt /

RUN python3 -m pip install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . /app

WORKDIR /app

EXPOSE 5002

CMD ["python", "app.py"]
