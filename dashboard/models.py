from django.db import models

class Alert(models.Model):
    
    # Timestamp
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Network flow info
    src_ip   = models.GenericIPAddressField(null=True, blank=True)
    dst_ip   = models.GenericIPAddressField(null=True, blank=True)
    src_port = models.IntegerField(null=True, blank=True)
    dst_port = models.IntegerField(null=True, blank=True)
    protocol = models.CharField(max_length=10, null=True, blank=True)
    
    # ML Detection
    ml_prediction = models.IntegerField(default=0)
    ml_confidence = models.FloatField(default=0.0)
    ml_model_used = models.CharField(max_length=50, default='RandomForest')
    
    # Multiclass detection
    attack_type    = models.CharField(max_length=50, default='BENIGN')
    detection_mode = models.CharField(max_length=15, default='binary')

    # Snort Detection
    snort_alert = models.BooleanField(default=False)
    snort_sid   = models.CharField(max_length=50, null=True, blank=True)
    
    # Fusion Engine Decision
    DECISION_CHOICES = [
        ('BENIGN', 'Benign'),
        ('ALERT',  'Alert'),
        ('THREAT', 'Threat'),
    ]
    decision = models.CharField(max_length=10, choices=DECISION_CHOICES, default='BENIGN')
    
    SOURCE_CHOICES = [
        ('ML_ONLY',    'ML Only'),
        ('SNORT_ONLY', 'Snort Only'),
        ('BOTH',       'Both'),
        ('NONE',       'None'),
    ]
    source_tag = models.CharField(max_length=15, choices=SOURCE_CHOICES, default='NONE')

    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"[{self.timestamp}] {self.decision} — {self.source_tag}"