from fargate.base_task_processor import BaseFargateTaskProcessor
from flotorch_core.storage.db.dynamodb import DynamoDB
from flotorch_core.evaluation.ragas import ragas_llm_bedrock_eval
from flotorch_core.evaluation.ragas import ragas_llm_eval_custom_gateway
from flotorch_core.evaluation.eval_factory import EvalFactory
from flotorch_core.config.env_config_provider import EnvConfigProvider
from flotorch_core.config.config import Config
from flotorch_core.logger.global_logger import get_logger
from evaluator.evaluator import Evaluator

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
            
            key_condition = "experiment_id = :exp_id"
            expression_values = {":exp_id": experiment_id}
            
            metrics_records = dynamo_db_question_metrics.query(
                key_condition_expression=key_condition,
                expression_attribute_values=expression_values,
                index_name=config.get_experimentid_index()
                )
            
            evaluator = EvalFactory.create_evaluator(
                exp_config_data.get('aws_region'),
                exp_config_data.get('eval_embedding_model'),
                exp_config_data.get('eval_retrieval_model'),
                exp_config_data.get('knowledge_base'),
                exp_config_data.get('eval_service') ,
                exp_config_data.get('gateway_enabled', False),
                f'{exp_config_data.get("gateway_url", "")}/api/openai/v1',
                exp_config_data.get('gateway_api_key', ""),
                float(exp_config_data.get("temp_retrieval_llm", 0)), 
                )
            
            
            evaluator = Evaluator(evaluator, metrics_records)
            experiment_eval_metrics = evaluator.evaluate()
            
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
