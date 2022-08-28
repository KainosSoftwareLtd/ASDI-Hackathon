FROM python:3.7-slim-buster

RUN pip3 install pandas==1.4.3 scikit-learn==1.1.2 numpy==1.23.2 multiprocess==0.70.13 boto3==1.24.12

ENV PYTHONUNBUFFERED=TRUE
ADD processing.py /

ENTRYPOINT ["python3", "/processing.py"]