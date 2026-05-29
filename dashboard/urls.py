from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('alerts/', views.alerts_page, name='alerts'),
    path('alerts/<int:alert_id>/', views.alert_detail, name='alert_detail'),
    path('models/', views.ml_models_page, name='ml_models'),
    path('api/classify/', views.classify, name='classify'),
    path('api/alerts/', views.alert_list, name='alert_list'),
    path('api/models/', views.models_status, name='models_status'),
    path('api/stats/', views.stats_api, name='stats_api'),
    path('snort/', views.snort_rules_page, name='snort_rules'),
    path('fusion/', views.fusion_engine_page, name='fusion_engine'),
]