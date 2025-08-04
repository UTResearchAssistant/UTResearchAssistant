from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

def list_projects(request):
    """List collaboration projects."""
    return JsonResponse({
        'projects': [
            {
                'id': 1,
                'title': 'AI Research Project',
                'members': ['user1', 'user2'],
                'created_at': '2025-08-04T00:00:00Z',
                'status': 'active'
            }
        ]
    })

@csrf_exempt
@require_http_methods(["POST"])
def create_project(request):
    """Create a new collaboration project."""
    try:
        data = json.loads(request.body)
        return JsonResponse({
            'success': True,
            'message': 'Project created',
            'project_id': 1
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def invite_member(request, project_id):
    """Invite member to project."""
    try:
        data = json.loads(request.body)
        return JsonResponse({
            'success': True,
            'message': 'Invitation sent',
            'project_id': project_id
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

def project_activity(request, project_id):
    """Get project activity feed."""
    return JsonResponse({
        'project_id': project_id,
        'activities': [
            {
                'user': 'john_doe',
                'action': 'added citation',
                'timestamp': '2025-08-04T12:00:00Z'
            }
        ]
    })

@require_http_methods(["DELETE"])
def leave_project(request, project_id):
    """Leave a collaboration project."""
    return JsonResponse({
        'success': True,
        'message': f'Left project {project_id}'
    })
