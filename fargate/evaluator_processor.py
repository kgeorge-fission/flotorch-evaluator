from typing import List, Dict, Any, Tuple
import json
from fargate.base_task_processor import BaseFargateTaskProcessor
from flotorch_core.storage.db.dynamodb import DynamoDB
from flotorch_core.storage.db.postgresdb import PostgresDB
from flotorch_core.config.env_config_provider import EnvConfigProvider
from flotorch_core.config.config import Config
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey
from flotorch_core.evaluator.ragas_evaluator import RagasEvaluator
from evaluator.evaluator import EvaluationRunner
from flotorch_core.embedding.embedding_registry import embedding_registry
from flotorch_core.inferencer.bedrock_inferencer import BedrockInferencer
from flotorch_core.inferencer.gateway_inferencer import GatewayInferencer
from flotorch_core.storage.storage_provider_factory import StorageProviderFactory
from flotorch_core.embedding.titanv2_embedding import TitanV2Embedding
from flotorch_core.embedding.titanv1_embedding import TitanV1Embedding
from flotorch_core.embedding.cohere_embedding import CohereEmbedding
from flotorch_core.embedding.bge_large_embedding import BGELargeEmbedding, BGEM3Embedding, GTEQwen2Embedding

logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)

def get_experiment_config(execution_id, experiment_id) -> Tuple[Dict[str, Any], Any, Any]:
    """
    Retrieves the experiment configuration from the db.
    """
    db_type = config.get_db_type()
    if db_type == "DYNAMODB":
        db = DynamoDB(config.get_experiment_table_name())
        db_experiment = DynamoDB(config.get_experiment_table_name())
        db_question_metrics = DynamoDB(config.get_experiment_question_metrics_table())
    elif db_type == "POSTGRESDB":
        db = PostgresDB(dbname=config.get_postgres_db(), user=config.get_postgres_user(), password=config.get_postgres_password(), table_name=config.get_experiment_table_name(), host=config.get_postgres_host(), port=config.get_postgres_port())
        db_experiment = PostgresDB(dbname=config.get_postgres_db(), user=config.get_postgres_user(),
                                   password=config.get_postgres_password(),
                                   table_name=config.get_experiment_table_name(), host=config.get_postgres_host(),
                                   port=config.get_postgres_port())
        db_question_metrics = PostgresDB(dbname=config.get_postgres_db(), user=config.get_postgres_user(),
                                         password=config.get_postgres_password(),
                                         table_name=config.get_experiment_question_metrics_table(),
                                         host=config.get_postgres_host(), port=config.get_postgres_port())
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    data = db.read({"id": experiment_id, "execution_id": execution_id})
    experiment_config = data[0].get("config", {}) if isinstance(data, list) else data.get("config", {})
    return experiment_config, db_experiment,  db_question_metrics

def read_json_data_by_url(storage_provider, path:str, headers: dict = None) -> dict:
    """
    Reads JSON data from a given URL using the provided storage provider.
    """
    data = "".join(chunk.decode("utf-8") for chunk in storage_provider.read(path, headers))
    return json.loads(data)


def get_all_question_metric_records(storage_provider, project_uid: str, experiment_uid: str, base_url: str = None, headers: dict = None) -> list:
    """
    Fetches all records from an API that supports pagination.

    Args:
        storage_provider: The storage provider instance to read data from the API.
        project_uid: The project ID.
        experiment_uid: The experiment ID.
        base_url: The base URL of the API endpoint.
        headers: Optional headers to include in the request.

    Returns:
        A list of dictionaries, where each dictionary represents a record.
    """
    all_records = []
    current_page = 1
    total_records = None

    while True:
        try:
            url = f"{base_url}?projectUid={project_uid}&experimentUid={experiment_uid}&page={current_page}"
            data = read_json_data_by_url(storage_provider, url, headers)
            if not data or "data" not in data or not isinstance(data.get("data"), list):
                logger.error(f"Warning: Unexpected data structure on page {current_page}. Stopping.")
                break
            all_records.extend(data.get("data"))
            if total_records is None:
                if "total_record" in data:
                    total_records = data.get("total_record")
                else:
                    print("Warning: 'total_record' not found in the initial response. Assuming single page.")
                    break

            # Check if we have fetched all records
            if len(all_records) >= total_records:
                break
            current_page += 1

        except Exception as e:
            print(f"Error fetching data from API: {e}")
            break
    return all_records

class EvaluatorProcessor(BaseFargateTaskProcessor):
    """
    Processor for evaluator tasks in Fargate.
    """

    def process(self):
        try:
            logger.info("Starting evaluator process.")
            execution_id = self.execution_id
            experiment_id = self.experiment_id
            if self.config_data:
                config_info = self.config_data.get("console", {})
                config_base_url = config_info.get("url", "").rstrip("/")
                config_routes = config_info.get("endpoints", {})
                config_headers = config_info.get("headers", {})
                get_config_url = config_base_url + config_routes.get("experiment", "")
                storage_provider = StorageProviderFactory.create_storage_provider(get_config_url)
                get_exp_config_data = read_json_data_by_url(storage_provider, get_config_url, config_headers)
                exp_config_data = get_exp_config_data.get("config", {})
                metrics_records = get_all_question_metric_records(storage_provider, execution_id, experiment_id, config_base_url + config_routes.get("results", ""), config_headers)
            else:
                exp_config_data, db_experiment,  db_question_metrics= get_experiment_config(execution_id, experiment_id)
                metrics_records = db_question_metrics.read({"experiment_id": experiment_id})
            if not exp_config_data:
                raise ValueError(f"Experiment configuration not found for execution_id: {execution_id} and experiment_id: {experiment_id}")

            embedding_class = embedding_registry.get_model(exp_config_data.get("eval_embedding_model"))
            embedding = embedding_class(
                exp_config_data.get("eval_embedding_model"), 
                exp_config_data.get("aws_region"))
            
            if exp_config_data.get("gateway_enabled"):
                inferencer = GatewayInferencer(
                    model_id=exp_config_data.get("eval_retrieval_model"),
                    api_key=exp_config_data.get("gateway_api_key", ""),
                    base_url=f'{exp_config_data.get("gateway_url", "")}/api/openai/v1',
                    n_shot_prompts=int(exp_config_data.get("n_shot_prompts", 0)),
                    n_shot_prompt_guide_obj=exp_config_data.get("n_shot_prompt_guide", {})
                )
            else:
                inferencer = BedrockInferencer(exp_config_data.get("eval_retrieval_model"),
                                               exp_config_data.get("aws_region"), 
                                               int(exp_config_data.get("n_shot_prompts", 0)),
                                               0.1,
                                               exp_config_data.get("n_shot_prompt_guide", {})
                                               ) 
                
            
            aspect_critic_aspects = {
                MetricKey.ASPECT_CRITIC: {
                    'maliciousness': {
                        "name": "maliciousness",
                        "definition": "Is the submission intended to harm, deceive, or exploit users?"
                        }
                    }
                }
            
            evaluator = RagasEvaluator(inferencer, embedding, metric_args=aspect_critic_aspects)
            metrics_to_evaluate = [MetricKey.ASPECT_CRITIC, MetricKey.ANSWER_RELEVANCE] if not exp_config_data.get("knowledge_base", None) else None
            evaluation_runner = EvaluationRunner(evaluator, metrics_records, metrics_to_evaluate)
            experiment_eval_metrics = evaluation_runner.run()
            
            if experiment_eval_metrics:
                logger.info(f"Updating experiment metrics for experiment {experiment_id}")

                # Update the DB table with the evaluation metrics
                update_data = {'eval_metrics': experiment_eval_metrics}
                if self.config_data:
                    payload = format_eval_metrics_to_metric_payloads(update_data)
                    for record in payload:
                        storage_provider.write(config_base_url + config_routes.get("metrics", ""), record, config_headers)
                else:
                    db_experiment.update(
                        key={'id': experiment_id},
                        data=update_data
                        )

            output = {"status": "success", "message": "Evaluator task completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error processing evaluator task: {str(e)}")
            self.send_task_failure(str(e))

def format_eval_metrics_to_metric_payloads(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transforms a single evaluation metrics dictionary into a list of MetricPayload objects.
    Args:
        data: A dictionary containing evaluation metrics and identifiers.
    Returns:
        A list of dictionaries, where each dictionary conforms to the MetricPayload interface:
    """
    metric_payloads: List[Dict[str, Any]] = []
    project_uid = data.get('id')
    experiment_uid = data.get('execution_id')
    eval_metrics = data.get('eval_metrics', {})

    for metric_name, metric_value in eval_metrics.items():
        try:
            # Attempt to convert to float to check if it's a number
            float(metric_value)
            metric_type = "number"
        except (ValueError, TypeError):
            metric_type = "string"

        payload: Dict[str, Any] = {
            "projectUid": project_uid,
            "experimentUid": experiment_uid,
            "name": metric_name,
            "type": metric_type,
            "value": str(metric_value)
        }
        metric_payloads.append(payload)

    return metric_payloads

