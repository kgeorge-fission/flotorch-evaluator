from typing import Dict, Optional
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
from flotorch_core.embedding.titanv2_embedding import TitanV2Embedding
from flotorch_core.embedding.titanv1_embedding import TitanV1Embedding
from flotorch_core.embedding.cohere_embedding import CohereEmbedding
from flotorch_core.embedding.bge_large_embedding import BGELargeEmbedding, BGEM3Embedding, GTEQwen2Embedding

logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)
db_type = config.get_db_type()

def get_experiment_config(execution_id, experiment_id) -> Dict:
    """
    Retrieves the experiment configuration from the db.
    """
    if db_type == "DYNAMODB":
        db = DynamoDB(config.get_experiment_table_name())
    elif db_type == "POSTGRESDB":
        db = PostgresDB(dbname=config.get_postgres_db(), user=config.get_postgres_user(), password=config.get_postgres_password(), table_name=config.get_experiment_table_name(), host=config.get_postgres_host(), port=config.get_postgres_port())
    data = db.read({"id": experiment_id, "execution_id": execution_id})
    if data:
        experiment_config = data[0].get("config", {}) if isinstance(data, list) else data.get("config", {})
        return experiment_config
    return None
    

class EvaluatorProcessor(BaseFargateTaskProcessor):
    """
    Processor for evaluator tasks in Fargate.
    """

    def process(self):
        try:
            logger.info("Starting evaluator process.")
            execution_id = self.execution_id
            experiment_id = self.experiment_id

            exp_config_data = get_experiment_config(execution_id, experiment_id)
            if not exp_config_data:
                raise ValueError(f"Experiment configuration not found for execution_id: {execution_id} and experiment_id: {experiment_id}")
            if db_type == "DYNAMODB":
                db_experiment = DynamoDB(config.get_experiment_table_name())
                db_question_metrics = DynamoDB(config.get_experiment_question_metrics_table())
            elif db_type == "POSTGRESDB":
                db_experiment = PostgresDB(dbname=config.get_postgres_db(), user=config.get_postgres_user(), password=config.get_postgres_password(), table_name=config.get_experiment_table_name(), host=config.get_postgres_host(), port=config.get_postgres_port())
                db_question_metrics = PostgresDB(dbname=config.get_postgres_db(), user=config.get_postgres_user(), password=config.get_postgres_password(), table_name=config.get_experiment_question_metrics_table(), host=config.get_postgres_host(), port=config.get_postgres_port())
            
            metrics_records = db_question_metrics.read({"experiment_id": experiment_id})            
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
                db_experiment.update(
                    key={'id': experiment_id},
                    data=update_data
                    )

            output = {"status": "success", "message": "Evaluator task completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error processing evaluator task: {str(e)}")
            self.send_task_failure(str(e))
