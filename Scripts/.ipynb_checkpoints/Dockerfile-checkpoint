FROM python:3.7-slim-buster

#imstall most recent versions available to Sagemaker (not ideal bit Sagemaker doesn't support versions used locally)
RUN pip3 install pandas scikit-learn numpy multiprocess boto3

ENV PYTHONUNBUFFERED=TRUE
ADD processing.py /

ENTRYPOINT ["python3", "/processing.py"]