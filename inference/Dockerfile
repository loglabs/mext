FROM python:3.9

# set work directory
WORKDIR /src

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONUNBUFFERED=1

# copy project
COPY . .

# install dependencies
RUN pip install --upgrade pip
RUN pip install prometheus_client
RUN pip install grafanalib
RUN pip install scikit-learn
RUN pip install numpy
RUN pip install pandas
RUN pip install git+https://github.com/loglabs/mltrace.git@master
RUN pip install git+https://github.com/loglabs/ttb.git@main
RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2
RUN pip install -e .

# Set credentials

# Expose port 
EXPOSE 1000

# Run the application:
CMD ["python", "./inference/main.py"]