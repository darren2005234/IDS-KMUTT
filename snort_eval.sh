#!/bin/bash
source ~/ids_env/bin/activate
cd ~/ids_kmutt

SLICES=(
  "ddos:DDoS:~/pcap/ddos_pure.pcap"
  "portscan:PortScan:~/pcap/portscan_pure.pcap"
  "botnet:Botnet:~/pcap/botnet_pure.pcap"
  "ftp:FTP-Patator:~/pcap/ftp_patator_pure.pcap"
  "ssh:SSH-Patator:~/pcap/ssh_patator_pure.pcap"
  "dos:DoS:~/pcap/dos_pure.pcap"
  "heartbleed:Heartbleed:~/pcap/heartbleed_pure.pcap"
  "web:WebAttack:~/pcap/web_attack_pure.pcap"
)

for entry in "${SLICES[@]}"; do
  IFS=':' read -r key label pcap <<< "$entry"

  echo ""
  echo "════════════════════════════════════════════"
  echo "  $label  ($pcap)"
  echo "════════════════════════════════════════════"

  # Vider la base
  python manage.py shell -c \
    "from dashboard.models import Alert; Alert.objects.all().delete()" \
    2>/dev/null

  # Rejouer
  sudo tcpreplay -i eth0 --mbps=10 $pcap > /dev/null 2>&1
  echo "  Replay terminé — attente pipeline (90s)..."
  sleep 90

  # Interroger
  python manage.py shell -c "
from dashboard.models import Alert
from django.db.models import Count

total = Alert.objects.count()
threats = Alert.objects.filter(decision='THREAT').count()
both   = Alert.objects.filter(decision='THREAT', snort_alert=True).count()
ml_only= Alert.objects.filter(decision='THREAT', snort_alert=False).count()
snort_only = Alert.objects.filter(snort_alert=True, ml_prediction=0).count()
snort_total= Alert.objects.filter(snort_alert=True).count()

print(f'  Total flux     : {total}')
print(f'  THREAT total   : {threats}')
print(f'  BOTH (ML+Snort): {both}')
print(f'  ML_ONLY        : {ml_only}')
print(f'  SNORT_ONLY     : {snort_only}')
print(f'  Snort a tiré   : {snort_total} fois au total')
print()
print('  Top SIDs Snort :')
for r in Alert.objects.filter(snort_alert=True).values('snort_sid').annotate(n=Count('id')).order_by('-n')[:5]:
    sid = r['snort_sid']
    n   = r['n']
    print(f'    SID {sid}: {n} alertes')
" 2>/dev/null

  # Texte des top règles
  echo "  Règles correspondantes :"
  python manage.py shell -c "
from dashboard.models import Alert
from django.db.models import Count
sids = [r['snort_sid'] for r in Alert.objects.filter(snort_alert=True).values('snort_sid').annotate(n=Count('id')).order_by('-n')[:3] if r['snort_sid']]
for sid in sids:
    print(f'    SID {sid}:', end=' ')
    import subprocess
    result = subprocess.run(['sudo','grep','-rh',f'sid:{sid};','/etc/snort/rules/'], capture_output=True, text=True)
    line = result.stdout.strip().split('\n')[0] if result.stdout.strip() else 'non trouvée'
    msg = ''
    import re
    m = re.search(r'msg:\"([^\"]+)\"', line)
    if m: msg = m.group(1)
    print(msg or 'règle non trouvée')
" 2>/dev/null

done

echo ""
echo "=== Test Snort terminé ==="
