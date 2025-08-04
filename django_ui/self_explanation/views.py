from django.http import JsonResponse, HttpResponseBadRequest
from django.views import View
import json
import uuid
from .services import SelfExplanationService
from training_config.models import TrainingConfiguration, DatasetProfile
from .models import SelfExplanation, DatasetExplanation

class GenerateExplanationView(View):
    """
    API View to trigger the generation of a self-explanation for a TrainingConfiguration.
    """
    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            config_id = data.get('training_configuration_id')
            if not config_id:
                return HttpResponseBadRequest("Missing 'training_configuration_id'")
            
            training_config = TrainingConfiguration.objects.get(id=config_id)
            
            service = SelfExplanationService()
            explanation = service.generate_explanation(training_config)
            
            return JsonResponse({
                'message': 'Explanation generated successfully.',
                'explanation_id': explanation.id,
                'explanation_text': explanation.explanation_text
            }, status=201)
            
        except TrainingConfiguration.DoesNotExist:
            return JsonResponse({'error': 'TrainingConfiguration not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

class ExplanationDetailView(View):
    """
    API View to retrieve or regenerate a specific self-explanation.
    """
    def get(self, request, explanation_id, *args, **kwargs):
        try:
            explanation = SelfExplanation.objects.get(id=explanation_id)
            return JsonResponse({
                'id': explanation.id,
                'training_configuration_id': explanation.training_configuration.id,
                'explanation_text': explanation.explanation_text,
                'created_at': explanation.created_at,
                'updated_at': explanation.updated_at,
            })
        except SelfExplanation.DoesNotExist:
            return JsonResponse({'error': 'Explanation not found.'}, status=404)

    def post(self, request, explanation_id, *args, **kwargs):
        """
        Triggers regeneration of the explanation.
        """
        try:
            service = SelfExplanationService()
            explanation = service.regenerate_explanation(explanation_id)
            return JsonResponse({
                'message': 'Explanation regenerated successfully.',
                'explanation_id': explanation.id,
                'explanation_text': explanation.explanation_text
            })
        except SelfExplanation.DoesNotExist:
            return JsonResponse({'error': 'Explanation not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

class GenerateDatasetExplanationView(View):
    """
    API View to trigger the generation of a self-explanation for a DatasetProfile.
    """
    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            dataset_id = data.get('dataset_profile_id')
            if not dataset_id:
                return HttpResponseBadRequest("Missing 'dataset_profile_id'")
            
            dataset_profile = DatasetProfile.objects.get(id=dataset_id)
            
            service = SelfExplanationService()
            explanation = service.generate_dataset_explanation(dataset_profile)
            
            return JsonResponse({
                'message': 'Dataset explanation generated successfully.',
                'explanation_id': explanation.id,
                'explanation_text': explanation.explanation_text
            }, status=201)
            
        except DatasetProfile.DoesNotExist:
            return JsonResponse({'error': 'DatasetProfile not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

class DatasetExplanationDetailView(View):
    """
    API View to retrieve or regenerate a specific dataset explanation.
    """
    def get(self, request, explanation_id, *args, **kwargs):
        try:
            explanation = DatasetExplanation.objects.get(id=explanation_id)
            return JsonResponse({
                'id': explanation.id,
                'dataset_profile_id': explanation.dataset_profile.id,
                'explanation_text': explanation.explanation_text,
                'created_at': explanation.created_at,
                'updated_at': explanation.updated_at,
            })
        except DatasetExplanation.DoesNotExist:
            return JsonResponse({'error': 'Dataset explanation not found.'}, status=404)
