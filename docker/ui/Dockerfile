# Docker container will have Python version: 3.12-slim.
# Check the Python image in https://hub.docker.com/_/python
FROM python:3.12-slim

# Set working directory to execute our program.
WORKDIR /home

# Copy anything needed to Docker container.
COPY ./requirements.txt ./
COPY ./src ./src/
COPY ./assets ./assets/


# Command to execute when building Docker images.
RUN \
apt-get update && \
apt-get upgrade -y && \
apt-get autoremove -y && \
apt-get clean -y && \
pip install --upgrade pip && \
pip install wheel && \
pip install -r requirements.txt

# Expose port the app runs on.
EXPOSE 8501

# Command to execute when running Docker container.
CMD ["streamlit", "run", "src/ui.py"]


# To build and run the Docker container, use the following commands:
# docker build -t ui_aq .
# docker run -p 8501:8501 ui_aq