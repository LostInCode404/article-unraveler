# Base image
FROM python:3.11.8-slim-bookworm

# Install python dependencies
COPY ../app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Setup app dir
RUN mkdir /app
WORKDIR /app
COPY ./app/*.py ./

# Copy streamlit file
COPY ./app/streamlit.toml ./.streamlit/config.toml

# Run app
CMD [ "streamlit", "run", "main.py" ]
