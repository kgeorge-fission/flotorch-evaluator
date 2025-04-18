from pydantic import BaseModel, Field
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Union
from flotorch_core.evaluator.evaluation_item import EvaluationItem

@dataclass
class EvaluationMetrics():
    faithfulness_score: Optional[float] = 0.0
    context_precision_score: Optional[float] = 0.0
    aspect_critic_score: Optional[float] = 0.0
    answers_relevancy_score: Optional[float] = 0.0
    string_similarity: Optional[float] = 0.0
    context_recall: Optional[float] = 0.0
    rouge_score: Optional[float] = 0.0


    def from_dict(self, metrics_dict: Dict[str, str]) -> 'EvaluationMetrics':
        """Convert metrics from DynamoDB format to EvaluationMetrics"""
        return EvaluationMetrics(
            faithfulness_score=float(metrics_dict.get('faithfulness', '0.0')),
            context_precision_score=float(metrics_dict.get('llm_context_precision_with_reference', '0.0')),
            aspect_critic_score=float(metrics_dict.get('maliciousness', '0.0')),
            answers_relevancy_score=float(metrics_dict.get('answer_relevancy', '0.0')),
            string_similarity=float(metrics_dict.get('String_Similarity', '0.0')),
            context_recall=float(metrics_dict.get('Context_Recall', '0.0')),
            rouge_score=float(metrics_dict.get('Rouge_Score', '0.0'))
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            'faithfulness_score': str(self.faithfulness_score),
            'context_precision_score': str(self.context_precision_score),
            'aspect_critic_score': str(self.aspect_critic_score),
            'answers_relevancy_score': str(self.answers_relevancy_score),
            'string_similarity_score': str(self.string_similarity),
            'context_recall_score': str(self.context_recall),
            'rouge_score': str(self.rouge_score)
        }
    
    def to_dynamo_format(self) -> dict:
        return {
            'eval_metrics': {
                'string_similarity_score': str(self.string_similarity) if self.string_similarity is not None else '0.0',
                'context_recall_score': str(self.context_recall) if self.context_recall is not None else '0.0',
                'rouge_score': str(self.rouge_score) if self.rouge_score is not None else '0.0',
                'faithfulness_score': str(self.faithfulness_score) if self.faithfulness_score is not None else '0.0',
                'context_precision_score': str(self.context_precision_score) if self.context_precision_score is not None else '0.0',
                'aspect_critic_score': str(self.aspect_critic_score) if self.aspect_critic_score is not None else '0.0',
                'answers_relevancy_score': str(self.answers_relevancy_score) if self.answers_relevancy_score is not None else '0.0'
            }
        }
    
    def to_dynamo_format(self) -> Dict[str, Dict[str, str]]:
        """Convert metrics to DynamoDB format"""
        return {
            'Faithfulness': {'S': str(self.faithfulness_score) if self.faithfulness_score is not None else '0.0'},
            'Context_Precision': {'S': str(self.context_precision_score) if self.context_precision_score is not None else '0.0'},
            'Aspect_Critic': {'S': str(self.aspect_critic_score) if self.aspect_critic_score is not None else '0.0'},
            'Answers_Relevancy': {'S': str(self.answers_relevancy_score) if self.answers_relevancy_score is not None else '0.0'},
            'String_Similarity': {'S': str(self.string_similarity) if self.string_similarity is not None else '0.0'},
            'Context_Precision': {'S': str(self.context_precision) if self.context_precision is not None else '0.0'},
            'Context_Recall': {'S': str(self.context_recall) if self.context_recall is not None else '0.0'},
            'Rouge_Score': {'S': str(self.rouge_score) if self.rouge_score is not None else '0.0'}
        }
        
    
class EvaluationRunner:
    def __init__(self, evaluator, metric_records, metrics=None) -> None:
        self.evaluator = evaluator
        self.metric_records = metric_records
        self.metrics = metrics
    
    def run(self):
        
        metric_records = [
            EvaluationItem(
                question=record['question'],
                generated_answer=record['generated_answer'],
                expected_answer=record['gt_answer'],
                context=record.get('reference_contexts')
            ) for record in self.metric_records
            ]
            
        metrics = self.evaluator.evaluate(metric_records, self.metrics)
        
        experiment_eval_metrics = {}
        if metrics:
            experiment_eval_metrics = metrics._repr_dict
            experiment_eval_metrics = {key: round(value, 2) if isinstance(value, float) else value for key, value in experiment_eval_metrics.items()}        
            experiment_eval_metrics = EvaluationMetrics().from_dict(experiment_eval_metrics).to_dict()
            
        return experiment_eval_metrics