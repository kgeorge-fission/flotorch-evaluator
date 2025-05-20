import json
import os
from fargate.evaluator_processor import EvaluatorProcessor
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.config.config import Config
from flotorch_core.config.env_config_provider import EnvConfigProvider

logger = get_logger()

# Initialize configuration provider and config
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)


def get_environment_data():
    """
    Fetches task token and input data from environment variables.
    Returns:
        tuple: Task token (str) and input data (dict).
    """
    EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID")
    print(f"EXPERIMENT_ID: {EXPERIMENT_ID}")
    EXECUTION_ID = os.environ.get("EXECUTION_ID")
    print(f"EXECUTION_ID: {EXECUTION_ID}")
    return EXPERIMENT_ID, EXECUTION_ID


def main():
    """
    Main entry point for the Fargate retriever handler.
    """
    try:
        experiment_id, execution_id = get_environment_data()

        # Initialize and process the RetrieverProcessor
        fargate_processor = EvaluatorProcessor(experiment_id=experiment_id, execution_id=execution_id)
        fargate_processor.process()
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        raise


if __name__ == "__main__":
    main()
