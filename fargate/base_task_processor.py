import json
import sys
from abc import ABC, abstractmethod
from flotorch_core.logger.global_logger import get_logger

logger = get_logger()

class BaseFargateTaskProcessor(ABC):
    """
    Abstract base class for Fargate task processors.
    """

    def __init__(self, experiment_id: str, execution_id: str):
        """
        Initializes the task processor with experiment and execution id.
        Args:
            experiment_id (str): The experiment ID.
            execution_id (str): The execution ID.
        """
        self.experiment_id = experiment_id
        self.execution_id = execution_id

    @abstractmethod
    def process(self):
        """
        Abstract method to be implemented by subclasses for processing tasks.
        """
        raise NotImplementedError("Subclasses must implement the process method.")

    def send_task_success(self, output: dict):
        """
        Sends task success signal to Step Functions.
        """
        print(json.dumps(output))
        sys.exit(0)

    def send_task_failure(self, error_message: str):
        """
        Sends task failure signal to Step Functions.
        Args:
            error_message (str): The error message to send to Step Functions.
        """
        print(json.dumps({"status": "failure", "output": "Evaluation Failed", "error": error_message}))
        sys.exit(1)
