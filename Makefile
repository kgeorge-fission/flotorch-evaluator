# List of folders under fargate/handler
FOLDERS := indexing

# The ECR repository URL must be passed as an environment variable
IMAGE_PREFIX ?= $(error IMAGE_PREFIX environment variable is required)

# Default target
.PHONY: all
all: build

# Build Docker images for all folders
.PHONY: build
build:
	@echo "Building the images"
	@for folder in $(FOLDERS); do \
		echo "Building Docker image for $$folder..."; \
		docker build -t $(IMAGE_PREFIX)/$$folder:latest -f fargate/handler/Dockerfile fargate/handler; \
	done

# Push Docker images to ECR
.PHONY: push
push:
	@for folder in $(FOLDERS); do \
		echo "Pushing Docker image for $$folder to ECR..."; \
		docker push $(IMAGE_PREFIX)/$$folder:latest; \
	done

# Clean up dangling images
.PHONY: clean
clean:
	@echo "Cleaning up dangling images..."
	@docker image prune -f

# Remove all built images
.PHONY: remove
remove:
	@for folder in $(FOLDERS); do \
		echo "Removing Docker image for $$folder..."; \
		docker rmi $(IMAGE_PREFIX)/$$folder:latest; \
	done