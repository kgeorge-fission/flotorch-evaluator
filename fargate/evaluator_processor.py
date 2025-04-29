from fargate.base_task_processor import BaseFargateTaskProcessor
from flotorch_core.storage.db.dynamodb import DynamoDB
from flotorch_core.config.env_config_provider import EnvConfigProvider
from flotorch_core.config.config import Config
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey
from flotorch_core.evaluator.ragas_evaluator import RagasEvaluator
from evaluator.evaluator import EvaluationRunner
from flotorch_core.embedding.embedding_registry import embedding_registry
from flotorch_core.inferencer.inferencer_provider_factory import InferencerProviderFactory
from flotorch_core.embedding.titanv2_embedding import TitanV2Embedding
from flotorch_core.embedding.titanv1_embedding import TitanV1Embedding
from flotorch_core.embedding.cohere_embedding import CohereEmbedding
from flotorch_core.embedding.bge_large_embedding import BGELargeEmbedding, BGEM3Embedding, GTEQwen2Embedding

logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)


class EvaluatorProcessor(BaseFargateTaskProcessor):
    """
    Processor for evaluator tasks in Fargate.
    """

    def process(self):
        try:
            logger.info("Starting evaluator process.")
            exp_config_data = self.input_data
            
            experiment_id = exp_config_data.get('experiment_id')
             
            dynamo_db_experiment = DynamoDB(config.get_experiment_table_name())
            dynamo_db_question_metrics = DynamoDB(config.get_experiment_question_metrics_table())
            
            metrics_records = dynamo_db_question_metrics.read({"experiment_id": experiment_id})
            
            if exp_config_data.get("knowledge_base", False) and not exp_config_data.get("bedrock_knowledge_base", False):
                embedding_class = embedding_registry.get_model(exp_config_data.get("embedding_model"))
                embedding = embedding_class(
                    exp_config_data.get("embedding_model"), 
                    exp_config_data.get("aws_region"), 
                    int(exp_config_data.get("vector_dimension")))
                is_opensearch_required = True
            else:
                embedding = None
                is_opensearch_required = False
            
            inferencer = InferencerProviderFactory.create_inferencer_provider(
                exp_config_data.get("gateway_enabled", False),
                f'{exp_config_data.get("gateway_url", "")}/api/openai/v1',
                exp_config_data.get("gateway_api_key", ""),
                exp_config_data.get("eval_service"),
                exp_config_data.get("eval_retrieval_model"), 
                exp_config_data.get("aws_region"), 
                config.get_sagemaker_arn_role(),
                int(exp_config_data.get("n_shot_prompts", 0)), 
                float(exp_config_data.get("temp_retrieval_llm", 0)), 
                exp_config_data.get("n_shot_prompt_guide_obj")
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

            evaluation_runner = EvaluationRunner(evaluator, metrics_records)
            experiment_eval_metrics = evaluation_runner.run()
            
            if experiment_eval_metrics:
                logger.info(f"Updating experiment metrics for experiment {experiment_id}")

                # Update the DynamoDB table with the evaluation metrics
                update_data = {'eval_metrics': experiment_eval_metrics}
                dynamo_db_experiment.update(
                    key={'id': experiment_id},
                    data=update_data
                    )

            output = {"status": "success", "message": "Evaluator task completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error processing evaluator task: {str(e)}")
            raise
