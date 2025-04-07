from pydantic import BaseModel, Field
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Union
from flotorch_core.evaluation.ragas import ragas_llm_bedrock_eval
from flotorch_core.evaluation.ragas import ragas_llm_eval_custom_gateway

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
        

class ExperimentQuestionMetrics(BaseModel):
    id : str = Field(default_factory=lambda: str(uuid.uuid4()), description="The unique identifier for the question")
    execution_id : str = Field(..., description="The execution id")
    experiment_id: str = Field(..., description="The unique identifier for the experiment")
    timestamp: datetime = Field(default_factory=datetime.now, description="The timestamp of the experiment")
    question: str = Field(..., description="The question that was asked")
    gt_answer: str = Field(..., description="The answer that was given")
    generated_answer: str = Field(default='', description="The answer that was generated")
    reference_contexts:  Optional[List[str]] = Field(..., description="The reference contexts retrieved from vectorstore") 
    query_metadata: Optional[Dict[str, int]] = Field(..., description="The metadata during querying")
    answer_metadata: Optional[Dict[str, int]] = Field(..., description="The metadata during answer generation")
    reference_contexts: Optional[List[str]] = Field(..., description="The reference contexts retrieved from vectorstore") 
    guardrail_input_assessment: Optional[Union[List[Dict], Dict]] = Field(default=None, description="Input guardrail assessment results")
    guardrail_context_assessment: Optional[Union[List[Dict], Dict]] = Field(default=None, description="Context guardrail assessment results")
    guardrail_output_assessment: Optional[Union[List[Dict], Dict]] = Field(default=None, description="Output guardrail assessment results")
    guardrail_id: Optional[str] = Field(default=None, description="The guardrail id that was used")
    guardrail_blocked: Optional[str] = Field(default=None, description="Input or Output blocked by Guardrail")


    @staticmethod
    def _format_guardrail_assessment(assessment: Union[List[Dict], Dict]) -> Dict:
        """Format guardrail assessment for DynamoDB"""
        if not assessment:
            return None

        # Convert single dictionary to list if necessary
        if isinstance(assessment, dict):
            assessment = [assessment]

        formatted_assessment = []
        for item in assessment:
            formatted_item = {}

            # Format Topic Policy
            if 'topicPolicy' in item:
                formatted_item['topicPolicy'] = {
                    'M': {
                        'topics': {
                            'L': [
                                {
                                    'M': {
                                        'name': {'S': topic.get('name', '')},
                                        'type': {'S': topic.get('type', '')},
                                        'action': {'S': topic.get('action', '')}
                                    }
                                }
                                for topic in item['topicPolicy'].get('topics', [])
                            ]
                        }
                    }
                }

            # Format Content Policy
            if 'contentPolicy' in item:
                formatted_item['contentPolicy'] = {
                    'M': {
                        'filters': {
                            'L': [
                                {
                                    'M': {
                                        'type': {'S': filter.get('type', '')},
                                        'confidence': {'S': filter.get('confidence', '')},
                                        'filterStrength': {'S': filter.get('filterStrength', '')},
                                        'action': {'S': filter.get('action', '')}
                                    }
                                }
                                for filter in item['contentPolicy'].get('filters', [])
                            ]
                        }
                    }
                }

            # Format Word Policy
            if 'wordPolicy' in item:
                word_policy = {'M': {}}

                if 'customWords' in item['wordPolicy']:
                    word_policy['M']['customWords'] = {
                        'L': [
                            {
                                'M': {
                                    'match': {'S': word.get('match', '')},
                                    'action': {'S': word.get('action', '')}
                                }
                            }
                            for word in item['wordPolicy'].get('customWords', [])
                        ]
                    }

                if 'managedWordLists' in item['wordPolicy']:
                    word_policy['M']['managedWordLists'] = {
                        'L': [
                            {
                                'M': {
                                    'match': {'S': word.get('match', '')},
                                    'type': {'S': word.get('type', '')},
                                    'action': {'S': word.get('action', '')}
                                }
                            }
                            for word in item['wordPolicy'].get('managedWordLists', [])
                        ]
                    }

                formatted_item['wordPolicy'] = word_policy

            # Format Sensitive Information Policy
            if 'sensitiveInformationPolicy' in item:
                sensitive_info_policy = {'M': {}}

                if 'piiEntities' in item['sensitiveInformationPolicy']:
                    sensitive_info_policy['M']['piiEntities'] = {
                        'L': [
                            {
                                'M': {
                                    'match': {'S': entity.get('match', '')},
                                    'type': {'S': entity.get('type', '')},
                                    'action': {'S': entity.get('action', '')}
                                }
                            }
                            for entity in item['sensitiveInformationPolicy'].get('piiEntities', [])
                        ]
                    }

                if 'regexes' in item['sensitiveInformationPolicy']:
                    sensitive_info_policy['M']['regexes'] = {
                        'L': [
                            {
                                'M': {
                                    'name': {'S': regex.get('name', '')},
                                    'match': {'S': regex.get('match', '')},
                                    'regex': {'S': regex.get('regex', '')},
                                    'action': {'S': regex.get('action', '')}
                                }
                            }
                            for regex in item['sensitiveInformationPolicy'].get('regexes', [])
                        ]
                    }

                formatted_item['sensitiveInformationPolicy'] = sensitive_info_policy

            # Format Contextual Grounding Policy
            if 'contextualGroundingPolicy' in item:
                formatted_item['contextualGroundingPolicy'] = {
                    'M': {
                        'filters': {
                            'L': [
                                {
                                    'M': {
                                        'type': {'S': filter.get('type', '')},
                                        'threshold': {'N': str(filter.get('threshold', 0))},
                                        'score': {'N': str(filter.get('score', 0))},
                                        'action': {'S': filter.get('action', '')}
                                    }
                                }
                                for filter in item['contextualGroundingPolicy'].get('filters', [])
                            ]
                        }
                    }
                }

            # Format Invocation Metrics
            if 'invocationMetrics' in item:
                metrics = item['invocationMetrics']
                formatted_item['invocationMetrics'] = {
                    'M': {
                        'guardrailProcessingLatency': {'N': str(metrics.get('guardrailProcessingLatency', 0))},
                        'usage': {
                            'M': {
                                'topicPolicyUnits': {'N': str(metrics.get('usage', {}).get('topicPolicyUnits', 0))},
                                'contentPolicyUnits': {'N': str(metrics.get('usage', {}).get('contentPolicyUnits', 0))},
                                'wordPolicyUnits': {'N': str(metrics.get('usage', {}).get('wordPolicyUnits', 0))},
                                'sensitiveInformationPolicyUnits': {'N': str(metrics.get('usage', {}).get('sensitiveInformationPolicyUnits', 0))},
                                'sensitiveInformationPolicyFreeUnits': {'N': str(metrics.get('usage', {}).get('sensitiveInformationPolicyFreeUnits', 0))},
                                'contextualGroundingPolicyUnits': {'N': str(metrics.get('usage', {}).get('contextualGroundingPolicyUnits', 0))}
                            }
                        }
                    }
                }

            formatted_assessment.append({'M': formatted_item})

        return {'L': formatted_assessment}

    def to_dynamo_item(self) -> Dict[str, Dict[str, str]]:
        """Convert to DynamoDB item format."""
        item = {
            'id' : {'S': self.id},
            'execution_id': {'S': self.execution_id},
            'experiment_id': {'S': self.experiment_id},
            'timestamp': {'S': self.timestamp.isoformat()},
            'question': {'S': self.question},
            'gt_answer': {'S': self.gt_answer},
            'generated_answer': {'S': self.generated_answer},
            'reference_contexts': {
                'L': [{'S': context} for context in self.reference_contexts]
            },
            'query_metadata': {
                'M': {key: {'N': str(value)} for key, value in self.query_metadata.items()}
            },
            'answer_metadata': {
                'M': {key: {'N': str(value)} for key, value in self.answer_metadata.items()}
            }
        } 

        # Format and add guardrail assessments if they exist
        if self.guardrail_input_assessment is not None:
            item['guardrail_input_assessment'] = self._format_guardrail_assessment(self.guardrail_input_assessment)

        if self.guardrail_context_assessment is not None:
            item['guardrail_context_assessment'] = self._format_guardrail_assessment(self.guardrail_context_assessment)

        if self.guardrail_output_assessment is not None:
            item['guardrail_output_assessment'] = self._format_guardrail_assessment(self.guardrail_output_assessment)

        if self.guardrail_id is not None:
            item['guardrail_id'] = {'S': self.guardrail_id}

        if self.guardrail_blocked is not None:
            item['guardrail_blocked'] = {'S': self.guardrail_blocked}

        return item
    
    
class Evaluator:
    def __init__(self, evaluator, metric_records) -> None:
        self.evaluator = evaluator
        self.metric_records = metric_records
    
    def evaluate(self):
        
        metric_records = [ExperimentQuestionMetrics(**question) for question in self.metric_records]
        metrics = self.evaluator.evaluate(metric_records)
        
        experiment_eval_metrics = {}
        if metrics:
            experiment_eval_metrics = metrics._repr_dict
            experiment_eval_metrics = {key: round(value, 2) if isinstance(value, float) else value for key, value in experiment_eval_metrics.items()}        
            experiment_eval_metrics = EvaluationMetrics().from_dict(experiment_eval_metrics).to_dict()
            
        return experiment_eval_metrics