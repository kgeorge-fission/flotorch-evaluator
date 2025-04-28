# Use the official AWS Lambda Python 3.9 runtime base image
FROM --platform=linux/amd64 python:3.9-slim

# Create and set the working directory inside the container
WORKDIR /var/task

COPY pyproject.toml .

# Install dependencies into the /var/task directory (where Lambda expects them)
RUN pip install .

# Copy the necessary files and directories
COPY fargate/ fargate/
COPY evaluator/ evaluator/
COPY fargate/handler/evaluator/fargate_evaluator_handler.py .

# Set environment variables
ENV PYTHONPATH=/var/task
ENV PYTHONUNBUFFERED=1

CMD ["python", "fargate_evaluator_handler.py"]
