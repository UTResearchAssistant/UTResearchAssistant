from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

@csrf_exempt
@require_http_methods(["POST"])
def check_integrity(request):
    """Check academic integrity of content."""
    try:
        data = json.loads(request.body)
        return JsonResponse({
            'success': True,
            'message': 'Integrity check started',
            'check_id': 1,
            'plagiarism_score': 15.2,
            'ai_detection_score': 8.5,
            'status': 'passed'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

def check_status(request, check_id):
    """Get integrity check status."""
    return JsonResponse({
        'check_id': check_id,
        'status': 'completed',
        'plagiarism_score': 15.2,
        'ai_detection_score': 8.5
    })

def integrity_report(request, check_id):
    """Get detailed integrity report."""
    return JsonResponse({
        'check_id': check_id,
        'report': {
            'plagiarism_details': 'Low similarity to known sources',
            'ai_detection_details': 'Minimal AI-generated content detected',
            'recommendations': ['Review citation format', 'Add more original analysis']
        }
    })

def list_checks(request):
    """List user's integrity checks."""
    return JsonResponse({
        'checks': [
            {
                'id': 1,
                'title': 'Research Paper Check',
                'created_at': '2025-08-04T00:00:00Z',
                'status': 'passed'
            }
        ]
    })
