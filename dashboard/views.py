from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Alert
from .fusion_engine import fuse


@api_view(['POST'])
def classify(request):
    """
    POST /api/classify/
    Accepts flow features + ML prediction + Snort alert
    Returns fusion decision.
    """
    data = request.data

    try:
        result = fuse(
            src_ip        = data.get('src_ip'),
            dst_ip        = data.get('dst_ip'),
            src_port      = data.get('src_port'),
            dst_port      = data.get('dst_port'),
            protocol      = data.get('protocol'),
            ml_prediction = int(data.get('ml_prediction', 0)),
            ml_confidence = float(data.get('ml_confidence', 0.0)),
            snort_alert   = bool(data.get('snort_alert', False)),
            snort_sid     = data.get('snort_sid'),
            ml_model_used = data.get('ml_model_used', 'RandomForest'),
        )
        return Response(result, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def alert_list(request):
    """
    GET /api/alerts/
    Returns last 100 alerts.
    """
    alerts = Alert.objects.all()[:100]
    data = [{
        'id'           : a.id,
        'timestamp'    : a.timestamp,
        'src_ip'       : a.src_ip,
        'dst_ip'       : a.dst_ip,
        'decision'     : a.decision,
        'source_tag'   : a.source_tag,
        'ml_confidence': a.ml_confidence,
        'snort_alert'  : a.snort_alert,
        'snort_sid'    : a.snort_sid,
    } for a in alerts]
    return Response(data)


@api_view(['GET'])
def dashboard_view(request):
    """
    GET /
    Main dashboard page.
    """
    total   = Alert.objects.count()
    threats = Alert.objects.filter(decision='THREAT').count()
    alerts  = Alert.objects.filter(decision='ALERT').count()
    benign  = Alert.objects.filter(decision='BENIGN').count()

    context = {
        'total'  : total,
        'threats': threats,
        'alerts' : alerts,
        'benign' : benign,
        'recent' : Alert.objects.all()[:20],
    }
    return render(request, 'dashboard/index.html', context)