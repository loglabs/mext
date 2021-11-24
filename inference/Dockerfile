FROM python:3.8

# set work directory
WORKDIR /src

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV GIT_PYTHON_REFRESH=quiet

# copy project
COPY . .

# install dependencies
RUN pip install --upgrade pip
RUN pip install prometheus_client
RUN pip install grafanalib
RUN pip install -e .

# Set credentials

# Expose port 
EXPOSE 1000

# Run the application:
CMD ["python", "./inference/scratch.py"]