from django.contrib import admin
from .models import Alert

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = [
        'timestamp', 'src_ip', 'dst_ip',
        'decision', 'source_tag', 'ml_confidence', 'snort_alert'
    ]
    list_filter = ['decision', 'source_tag', 'snort_alert']
    search_fields = ['src_ip', 'dst_ip', 'snort_sid']
    ordering = ['-timestamp']