from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Alert
from .fusion_engine import fuse
from .ml_engine import predict, get_models_status, load_models
import subprocess
import csv
import os

# Load models at startup
load_models()


# ── Dashboard main view ───────────────────────────────────────
def dashboard_view(request):
    total   = Alert.objects.count()
    threats = Alert.objects.filter(decision='THREAT').count()
    alerts  = Alert.objects.filter(decision='ALERT').count()
    benign  = Alert.objects.filter(decision='BENIGN').count()

    # Recent alerts for table
    recent = Alert.objects.all()[:10]

    # ML model breakdown
    rf_count   = Alert.objects.filter(ml_model_used__contains='RandomForest').count()
    xgb_count  = Alert.objects.filter(ml_model_used__contains='XGBoost').count()
    lstm_count = Alert.objects.filter(ml_model_used__contains='LSTM').count()

    # Source tag breakdown
    both_count   = Alert.objects.filter(source_tag='BOTH').count()
    ml_count     = Alert.objects.filter(source_tag='ML_ONLY').count()
    snort_count  = Alert.objects.filter(source_tag='SNORT_ONLY').count()

    context = {
        'total'      : total,
        'threats'    : threats,
        'alerts'     : alerts,
        'benign'     : benign,
        'recent'     : recent,
        'rf_count'   : rf_count,
        'xgb_count'  : xgb_count,
        'lstm_count' : lstm_count,
        'both_count' : both_count,
        'ml_count'   : ml_count,
        'snort_count': snort_count,
        'models'     : get_models_status(),
    }
    return render(request, 'dashboard/index.html', context)


# ── Alerts page with pagination and filters ───────────────────
def alerts_page(request):
    queryset = Alert.objects.all()

    # Filters
    decision   = request.GET.get('decision', '')
    source_tag = request.GET.get('source_tag', '')
    search     = request.GET.get('search', '')

    if decision:
        queryset = queryset.filter(decision=decision)
    if source_tag:
        queryset = queryset.filter(source_tag=source_tag)
    if search:
        queryset = queryset.filter(src_ip__icontains=search) | \
                   queryset.filter(dst_ip__icontains=search)

    # Pagination — 20 per page
    paginator = Paginator(queryset, 20)
    page_num  = request.GET.get('page', 1)
    page_obj  = paginator.get_page(page_num)

    context = {
        'page_obj'   : page_obj,
        'decision'   : decision,
        'source_tag' : source_tag,
        'search'     : search,
        'total'      : queryset.count(),
        'decisions'  : ['THREAT', 'ALERT', 'BENIGN'],
        'source_tags': ['BOTH', 'ML_ONLY', 'SNORT_ONLY', 'NONE'],
    }
    return render(request, 'dashboard/alerts.html', context)


# ── Alert detail view ─────────────────────────────────────────
def alert_detail(request, alert_id):
    alert = get_object_or_404(Alert, id=alert_id)
    
    # Get previous and next alert
    prev_alert = Alert.objects.filter(id__lt=alert_id).order_by('-id').first()
    next_alert = Alert.objects.filter(id__gt=alert_id).order_by('id').first()

    context = {
        'alert'     : alert,
        'prev_alert': prev_alert,
        'next_alert': next_alert,
    }
    return render(request, 'dashboard/alert_detail.html', context)


# ── REST API endpoints ────────────────────────────────────────
@api_view(['POST'])
def classify(request):
    import json as json_lib
    
    # Parse body directly to get all fields
    try:
        data = json_lib.loads(request.body.decode('utf-8'))
    except Exception:
        data = dict(request.data)
    
    print('DEBUG keys count:', len(data))
    print('Flow Duration present:', 'Flow Duration' in data)
    print('DEBUG keys:', list(data.keys())[:5])
    if 'Flow Duration' in data or 'Destination Port' in data:
        ml_result      = predict(dict(data))
        ml_prediction  = ml_result['ml_prediction']
        ml_confidence  = ml_result['ml_confidence']
        ml_model_used  = ml_result['ml_model_used']
        attack_type    = ml_result['attack_type']
        detection_mode = ml_result['detection_mode']
    else:
        ml_prediction  = int(data.get('ml_prediction', 0))
        ml_confidence  = float(data.get('ml_confidence', 0.0))
        ml_model_used  = data.get('ml_model_used', 'RandomForest')
        attack_type    = data.get('attack_type', 'BENIGN')
        detection_mode = data.get('detection_mode', 'cascade')

    try:
        result = fuse(
            src_ip         = data.get('src_ip'),
            dst_ip         = data.get('dst_ip'),
            src_port       = data.get('src_port'),
            dst_port       = data.get('dst_port'),
            protocol       = data.get('protocol'),
            ml_prediction  = ml_prediction,
            ml_confidence  = ml_confidence,
            snort_alert    = bool(data.get('snort_alert', False)),
            snort_sid      = data.get('snort_sid'),
            ml_model_used  = ml_model_used,
            attack_type    = attack_type,
            detection_mode = detection_mode,
        )
        return Response(result, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
@api_view(['GET'])
def alert_list(request):
    alerts = Alert.objects.all()[:100]
    data = [{
        'id'            : a.id,
        'timestamp'     : a.timestamp,
        'src_ip'        : a.src_ip,
        'dst_ip'        : a.dst_ip,
        'decision'      : a.decision,
        'source_tag'    : a.source_tag,
        'ml_confidence' : a.ml_confidence,
        'ml_model_used' : a.ml_model_used,
        'attack_type'   : a.attack_type,
        'detection_mode': a.detection_mode,
        'snort_alert'   : a.snort_alert,
        'snort_sid'     : a.snort_sid,
    } for a in alerts]
    return Response(data)


@api_view(['GET'])
def stats_api(request):
    from django.db.models import Count

    # Attack type distribution (multiclass only)
    attack_types = Alert.objects.filter(
        detection_mode__in=['multiclass', 'cascade']
    ).exclude(
        attack_type__in=['BENIGN', 'Unknown Attack']
    ).values('attack_type').annotate(
        count=Count('attack_type')
    ).order_by('-count')

    attack_dist = {item['attack_type']: item['count'] for item in attack_types}

    return Response({
        'total'      : Alert.objects.count(),
        'threats'    : Alert.objects.filter(decision='THREAT').count(),
        'alerts'     : Alert.objects.filter(decision='ALERT').count(),
        'benign'     : Alert.objects.filter(decision='BENIGN').count(),
        'both'       : Alert.objects.filter(source_tag='BOTH').count(),
        'ml_only'    : Alert.objects.filter(source_tag='ML_ONLY').count(),
        'snort_only' : Alert.objects.filter(source_tag='SNORT_ONLY').count(),
        'rf_count'   : Alert.objects.filter(ml_model_used__contains='RandomForest').count(),
        'xgb_count'  : Alert.objects.filter(ml_model_used__contains='XGBoost').count(),
        'lstm_count' : Alert.objects.filter(ml_model_used__contains='LSTM').count(),
        'attack_dist': attack_dist,
    })


@api_view(['GET'])
def models_status(request):
    return Response(get_models_status())


def ml_models_page(request):
    """GET /models/ — ML Models benchmark page."""
    
    RESULTS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )

    def read_csv(filename):
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            return {}
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows[0] if rows else {}

    def read_features(filename, top=10):
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            return []
        features = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= top:
                    break
                features.append({
                    'name'      : row['feature'],
                    'importance': round(float(row['importance']), 4),
                    'pct'       : round(float(row['importance']) * 100, 2)
                })
        return features

    rf_metrics   = read_csv('rf_metrics.csv')
    xgb_metrics  = read_csv('xgb_metrics.csv')
    lstm_metrics = read_csv('lstm_metrics.csv')
    rf_features  = read_features('rf_feature_importances.csv')
    xgb_features = read_features('xgb_feature_importances.csv')

    models_status = get_models_status()

    context = {
        'rf_metrics'   : rf_metrics,
        'xgb_metrics'  : xgb_metrics,
        'lstm_metrics' : lstm_metrics,
        'rf_features'  : rf_features,
        'xgb_features' : xgb_features,
        'models_status': models_status,
        'models_list'  : [
            ('Random Forest', 'RandomForest', 'RandomForest'),
            ('XGBoost',       'XGBoost',      'XGBoost'),
            ('LSTM',          'LSTM',          'LSTM'),
        ],
    }
    return render(request, 'dashboard/ml_models.html', context)


def snort_rules_page(request):
    """GET /snort/ — Snort status and alerts."""
    from django.db.models import Count

    snort_active = os.path.exists('/var/run/ids-snort-running')

    total_snort   = Alert.objects.filter(snort_alert=True).count()
    snort_only    = Alert.objects.filter(source_tag='SNORT_ONLY').count()
    both_detected = Alert.objects.filter(source_tag='BOTH').count()

    top_sids = Alert.objects.exclude(snort_sid=None)\
                            .values('snort_sid')\
                            .annotate(count=Count('snort_sid'))\
                            .order_by('-count')[:10]

    snort_db_alerts = Alert.objects.filter(snort_alert=True)\
                                   .order_by('-timestamp')[:20]

    context = {
        'snort_active'   : snort_active,
        'top_sids'       : top_sids,
        'snort_db_alerts': snort_db_alerts,
        'total_snort'    : total_snort,
        'snort_only'     : snort_only,
        'both_detected'  : both_detected,
        'rules_count'    : 133866,
        'snort_version'  : '2.9.20',
        'ruleset'        : 'Emerging Threats Open',
        'interface'      : 'eth0',
        'home_net'       : '10.35.111.10/32',
    }
    return render(request, 'dashboard/snort_rules.html', context)

def fusion_engine_page(request):
    """GET /fusion/ — Fusion Engine rules and statistics."""
    from django.db.models import Count, Avg

    # Rule application counts
    both_count      = Alert.objects.filter(source_tag='BOTH').count()
    ml_only_count   = Alert.objects.filter(source_tag='ML_ONLY').count()
    snort_only_count = Alert.objects.filter(source_tag='SNORT_ONLY').count()
    none_count      = Alert.objects.filter(source_tag='NONE').count()
    total           = Alert.objects.count()

    # Vote distribution
    votes_1 = Alert.objects.filter(ml_model_used__contains='+').exclude(
        ml_model_used__contains='RandomForest+XGBoost+LSTM'
    ).count()
    votes_2 = Alert.objects.filter(ml_model_used='RandomForest+XGBoost').count() + \
              Alert.objects.filter(ml_model_used='RandomForest+LSTM').count() + \
              Alert.objects.filter(ml_model_used='XGBoost+LSTM').count()
    votes_3 = Alert.objects.filter(ml_model_used='RandomForest+XGBoost+LSTM').count()

    # Model detection rates
    rf_detections   = Alert.objects.filter(ml_model_used__contains='RandomForest').count()
    xgb_detections  = Alert.objects.filter(ml_model_used__contains='XGBoost').count()
    lstm_detections = Alert.objects.filter(ml_model_used__contains='LSTM').count()

    # Average confidence per decision
    avg_conf_threat = Alert.objects.filter(decision='THREAT').aggregate(
        avg=Avg('ml_confidence'))['avg'] or 0
    avg_conf_alert  = Alert.objects.filter(decision='ALERT').aggregate(
        avg=Avg('ml_confidence'))['avg'] or 0

    # Attack type breakdown
    attack_dist = Alert.objects.exclude(
        attack_type__in=['BENIGN', 'Unknown Attack', '']
    ).values('attack_type').annotate(
        count=Count('attack_type')
    ).order_by('-count')

    context = {
        'both_count'       : both_count,
        'ml_only_count'    : ml_only_count,
        'snort_only_count' : snort_only_count,
        'none_count'       : none_count,
        'total'            : total,
        'votes_1'          : votes_1,
        'votes_2'          : votes_2,
        'votes_3'          : votes_3,
        'rf_detections'    : rf_detections,
        'xgb_detections'   : xgb_detections,
        'lstm_detections'  : lstm_detections,
        'avg_conf_threat'  : round(avg_conf_threat * 100, 1),
        'avg_conf_alert'   : round(avg_conf_alert * 100, 1),
        'attack_dist'      : attack_dist,
        'conf_threshold'   : 50,
    }
    return render(request, 'dashboard/fusion_engine.html', context)