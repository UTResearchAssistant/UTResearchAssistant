from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

def list_alerts(request):
    """List user's alerts."""
    return JsonResponse({
        'alerts': [
            {
                'id': 1,
                'title': 'New Research Paper Available',
                'message': 'A new paper matching your interests has been published',
                'type': 'info',
                'created_at': '2025-08-04T00:00:00Z',
                'read': False
            }
        ]
    })

@csrf_exempt
@require_http_methods(["POST"])
def create_alert(request):
    """Create a new alert."""
    try:
        data = json.loads(request.body)
        return JsonResponse({
            'success': True,
            'message': 'Alert created',
            'alert_id': 1
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@csrf_exempt
@require_http_methods(["PUT"])
def mark_read(request, alert_id):
    """Mark alert as read."""
    return JsonResponse({
        'success': True,
        'message': f'Alert {alert_id} marked as read'
    })

@require_http_methods(["DELETE"])
def delete_alert(request, alert_id):
    """Delete an alert."""
    return JsonResponse({
        'success': True,
        'message': f'Alert {alert_id} deleted'
    })

def alert_settings(request):
    """Get user's alert settings."""
    return JsonResponse({
        'settings': {
            'email_notifications': True,
            'push_notifications': False,
            'research_updates': True,
            'collaboration_invites': True
        }
    })

@csrf_exempt
@require_http_methods(["PUT"])
def update_settings(request):
    """Update alert settings."""
    try:
        data = json.loads(request.body)
        return JsonResponse({
            'success': True,
            'message': 'Settings updated'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)
