import openai
import uuid
from django.conf import settings
from .models import SelfExplanation, DatasetExplanation
from training_config.models import TrainingConfiguration, DatasetProfile

class SelfExplanationService:
    """
    A service to generate natural language explanations for training configurations.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        openai.api_key = self.api_key

    def generate_explanation(self, training_config: TrainingConfiguration) -> SelfExplanation:
        """
        Generates and saves a self-explanation for a given TrainingConfiguration.

        Args:
            training_config: The TrainingConfiguration instance to explain.

        Returns:
            A new SelfExplanation instance.
        """
        prompt = self._build_prompt(training_config)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert MLOps assistant. Your task is to provide a clear, concise, and insightful explanation of a given model training configuration. Focus on the rationale behind the choices and their interplay."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            explanation_text = response.choices[0].message['content'].strip()
        except Exception as e:
            explanation_text = f"Error generating explanation: {str(e)}"

        explanation, _ = SelfExplanation.objects.update_or_create(
            training_configuration=training_config,
            defaults={'explanation_text': explanation_text}
        )
        return explanation

    def _build_prompt(self, config: TrainingConfiguration) -> str:
        """Builds the detailed prompt for the LLM."""
        
        prompt = f"""
        Please provide a self-explanation for the following training configuration named '{config.name}'.

        **Objective:** Analyze the configuration choices and explain the likely reasoning and trade-offs.

        **Configuration Details:**
        - **Base Model:** {config.base_model.name} ({config.base_model.parameter_count:,} parameters, {config.base_model.context_length} context length).
        - **Adapter:** {config.adapter.name} ({config.adapter.get_adapter_type_display()}).
        - **Dataset:** {config.dataset.name} ({config.dataset.total_tokens:,} total tokens).
        - **Hardware:** {config.hardware.name} ({config.hardware.memory_gb}GB VRAM, {config.hardware.fp16_tflops} FP16 TFLOPs).
        - **Precision:** {config.precision}

        **Key Hyperparameters:**
        - **Learning Rate:** {config.learning_rate}
        - **Batch Size:** {config.batch_size}
        - **Epochs:** {config.epochs}
        - **Optimizer:** {config.optimizer}

        **Analysis Required:**
        1.  **Overall Strategy:** What is the high-level goal of this configuration? (e.g., fast iteration, cost-efficiency, maximum performance).
        2.  **Component Synergy:** How do the chosen model, adapter, and hardware interact? Is the hardware a good fit for the model size and batch size?
        3.  **Hyperparameter Rationale:** Explain the choice of learning rate, batch size, and epochs. Are they typical for this kind of setup? What are the implications?
        4.  **Potential Risks & Bottlenecks:** What are the potential risks? (e.g., VRAM limitations, slow training, overfitting/underfitting).
        5.  **Estimated Outcome:** Based on the estimations (Cost: ${config.estimated_cost_usd}, Time: {config.estimated_time_hours:.2f} hours), is this a cost-effective and time-efficient run?

        Provide the explanation in clear, well-structured markdown.
        """
        return prompt

    def regenerate_explanation(self, explanation_id: uuid.UUID) -> SelfExplanation:
        """
        Finds an existing explanation and regenerates its content.
        """
        explanation = SelfExplanation.objects.get(id=explanation_id)
        return self.generate_explanation(explanation.training_configuration)

    def generate_dataset_explanation(self, dataset_profile: DatasetProfile) -> DatasetExplanation:
        """
        Generates and saves a self-explanation for a given DatasetProfile.

        Args:
            dataset_profile: The DatasetProfile instance to explain.

        Returns:
            A new DatasetExplanation instance.
        """
        prompt = self._build_dataset_prompt(dataset_profile)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert data scientist and ML researcher. Your task is to provide a clear, comprehensive analysis of a dataset, focusing on its characteristics, potential applications, limitations, and biases."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            explanation_text = response.choices[0].message['content'].strip()
        except Exception as e:
            explanation_text = f"Error generating dataset explanation: {str(e)}"

        explanation, _ = DatasetExplanation.objects.update_or_create(
            dataset_profile=dataset_profile,
            defaults={'explanation_text': explanation_text}
        )
        return explanation

    def _build_dataset_prompt(self, dataset: DatasetProfile) -> str:
        """Builds the detailed prompt for analyzing a dataset."""
        
        prompt = f"""
        Please provide a comprehensive analysis for the following dataset: '{dataset.name}'.

        **Objective:** Analyze the dataset characteristics and explain its potential applications, limitations, and considerations.

        **Dataset Details:**
        - **Type:** {dataset.get_dataset_type_display()}
        - **Size:** {dataset.num_examples:,} examples
        - **Total Tokens:** {dataset.total_tokens:,} tokens
        - **Average Tokens per Example:** {dataset.avg_tokens_per_example}
        - **Storage Size:** {dataset.size_gb:.2f} GB
        - **HuggingFace Dataset:** {dataset.hf_dataset_name or 'Not specified'}
        - **Description:** {dataset.description or 'No description provided'}

        **Analysis Required:**
        1. **Dataset Characteristics:** What can we infer about the nature and quality of this dataset from its size, token distribution, and type?
        
        2. **Potential Applications:** What types of ML tasks and models would this dataset be most suitable for? Consider the dataset type, size, and token characteristics.
        
        3. **Training Considerations:** 
           - Is the dataset size appropriate for training different model sizes?
           - What are the implications of the token distribution?
           - How might the dataset size affect training time and computational requirements?
        
        4. **Potential Limitations and Biases:**
           - What biases might be present in a dataset of this type and size?
           - What are the potential limitations for certain use cases?
           - What preprocessing considerations should be kept in mind?
        
        5. **Quality Assessment:** Based on the available metrics, how would you assess the overall quality and utility of this dataset?

        Provide the analysis in clear, well-structured markdown format.
        """
        return prompt

