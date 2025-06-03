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
    logger.info(f"EXPERIMENT_ID: {EXPERIMENT_ID}")
    EXECUTION_ID = os.environ.get("EXECUTION_ID")
    logger.info(f"EXECUTION_ID: {EXECUTION_ID}")
    CONFIG_DATA_RAW = os.environ.get("CONFIG_DATA")
    CONFIG_DATA = {}
    if CONFIG_DATA_RAW:
        try:
            parsed_data = json.loads(CONFIG_DATA_RAW)
            if isinstance(parsed_data, dict):
                CONFIG_DATA = parsed_data
            else:
                logger.info(
                    f"Warning: CONFIG_DATA environment variable was parsed but is not a dictionary. Type: {type(parsed_data)}. Using default empty dict."
                )
        except Exception as e:  # Catches both JSONDecodeError and TypeError
            logger.info(
                f"Error processing CONFIG_DATA environment variable: {e}. Using default empty dictionary."
            )
    else:
        logger.info("CONFIG_DATA environment variable not set or is empty. Using default empty dictionary.")

    return EXPERIMENT_ID, EXECUTION_ID, CONFIG_DATA


def main():
    """
    Main entry point for the Fargate retriever handler.
    """
    try:
        experiment_id, execution_id, config_data = get_environment_data()
        # Initialize and process the RetrieverProcessor
        fargate_processor = EvaluatorProcessor(experiment_id=experiment_id, execution_id=execution_id, config_data=config_data)
        fargate_processor.process()
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        raise


if __name__ == "__main__":
    main()
