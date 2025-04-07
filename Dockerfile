FROM python:3.12

WORKDIR /app

COPY pyproject.toml .

# Install dependencies
COPY . .
# Removing the tests package to avoid installing the test dependencies
RUN rm -rf tests
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install

# Copy the rest of the application code to the container
COPY . .

# Set the entry point to run index.py
ENTRYPOINT ["python", "index.py"]