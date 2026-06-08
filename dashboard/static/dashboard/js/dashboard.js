const REFRESH_INTERVAL = 5000;
let currentMode = localStorage.getItem('ids_detection_mode') || 'binary';

// ── Charts ───────────────────────────────────────────────────
const trafficChart = new Chart(
    document.getElementById('trafficChart').getContext('2d'), {
    type: 'doughnut',
    data: {
        labels: ['BENIGN', 'ALERT', 'THREAT'],
        datasets: [{
            data: [0, 0, 0],
            backgroundColor: ['#22c55e', '#f97316', '#ef4444'],
            borderColor: '#111827', borderWidth: 3, hoverOffset: 6
        }]
    },
    options: {
        cutout: '65%',
        plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 16, font: { size: 12 } } } }
    }
});

const sourceChart = new Chart(
    document.getElementById('sourceChart').getContext('2d'), {
    type: 'bar',
    data: {
        labels: ['ML Only', 'Snort Only', 'Both'],
        datasets: [{
            data: [0, 0, 0],
            backgroundColor: [
                'rgba(59,130,246,0.7)',
                'rgba(168,85,247,0.7)',
                'rgba(239,68,68,0.7)',
            ],
            borderColor: ['#3b82f6', '#a855f7', '#ef4444'],
        }]
    },
    options: {
        scales: {
            x: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(30,45,69,0.8)' } },
            y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(30,45,69,0.8)' }, beginAtZero: true }
        },
        plugins: { legend: { display: false } }
    }
});
const attackTypeChart = new Chart(
    document.getElementById('attackTypeChart').getContext('2d'), {
    type: 'doughnut',
    data: {
        labels: [],
        datasets: [{
            data: [],
            backgroundColor: [
                '#ef4444', '#dc2626', '#f97316',
                '#f59e0b', '#a855f7', '#6366f1',
                '#64748b', '#22c55e'
            ],
            borderColor: '#111827', borderWidth: 3, hoverOffset: 4
        }]
    },
    options: {
        cutout: '60%',
        plugins: {
            legend: {
                position: 'right',
                labels: { color: '#94a3b8', font: { size: 11 }, padding: 10 }
            }
        }
    }
});

// ── Mode selector ────────────────────────────────────────────
function setMode(mode) {
    currentMode = mode;
    localStorage.setItem('ids_detection_mode', mode);

    const btnBinary = document.getElementById('btn-binary');
    const btnMulti  = document.getElementById('btn-multi');
    const thAttack  = document.getElementById('th-attack-type');

    if (mode === 'multiclass') {
        btnBinary.style.background = 'transparent';
        btnBinary.style.color      = '#64748b';
        btnMulti.style.background  = '#1d7adb';
        btnMulti.style.color       = 'white';
        thAttack.style.display     = '';
    } else {
        btnBinary.style.background = '#1d7adb';
        btnBinary.style.color      = 'white';
        btnMulti.style.background  = 'transparent';
        btnMulti.style.color       = '#64748b';
        thAttack.style.display     = 'none';
    }
    const attackCard = document.getElementById('attack-type-card');
    if (mode === 'multiclass') {
        attackCard.style.display = '';
    } else {
        attackCard.style.display = 'none';
    }

    refreshDashboard();
}

// ── Helpers ───────────────────────────────────────────────────
function confBar(value) {
    const pct   = Math.round(value * 100);
    const level = pct >= 85 ? 'high' : pct >= 60 ? 'med' : 'low';
    return `<div class="conf-bar">
        <div class="bar"><div class="fill ${level}" style="width:${pct}%"></div></div>
        <span>${pct}%</span>
    </div>`;
}

function attackTypeBadge(type) {
    const colors = {
        'BENIGN'     : '#22c55e',
        'DoS'        : '#ef4444',
        'DDoS'       : '#dc2626',
        'PortScan'   : '#f97316',
        'Brute Force': '#f59e0b',
        'Web Attack' : '#a855f7',
        'Botnet'     : '#6366f1',
        'Rare Attack': '#64748b',
        'ATTACK'     : '#ef4444',
    };
    const color = colors[type] || '#64748b';
    return `<span style="font-size:0.75rem; padding:2px 8px; border-radius:4px;
        background:${color}22; color:${color}; border:1px solid ${color}44;
        font-weight:600;">${type || '—'}</span>`;
}

function renderRow(a) {
    const ts = new Date(a.timestamp).toLocaleString('en-GB', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        timeZone: 'Asia/Bangkok'
    });

    const attackTypeCell = currentMode === 'multiclass'
        ? `<td>${attackTypeBadge(a.attack_type)}</td>` : '';

    return `<tr style="cursor:pointer;" onclick="window.location='/alerts/${a.id}/'">
        <td style="color:#64748b">#${a.id}</td>
        <td>${ts}</td>
        <td style="font-family:monospace">${a.src_ip || '—'}</td>
        <td style="font-family:monospace">${a.dst_ip || '—'}</td>
        <td><span class="badge ${a.decision}">${a.decision}</span></td>
        <td><span class="tag ${a.source_tag}">${a.source_tag.replace('_', ' ')}</span></td>
        ${attackTypeCell}
        <td>${confBar(a.ml_confidence)}</td>
        <td style="color:#64748b;font-family:monospace">${a.snort_sid || '—'}</td>
    </tr>`;
}

// ── Main refresh ──────────────────────────────────────────────
async function refreshDashboard() {
    try {
        // Real stats from DB
        const statsRes = await fetch('/api/stats/');
        const stats    = await statsRes.json();

        document.getElementById('stat-total').textContent  = stats.total;
        document.getElementById('stat-threat').textContent = stats.threats;
        document.getElementById('stat-alert').textContent  = stats.alerts;
        document.getElementById('stat-benign').textContent = stats.benign;

        trafficChart.data.datasets[0].data = [stats.benign, stats.alerts, stats.threats];
        trafficChart.update('none');

        sourceChart.data.datasets[0].data = [stats.ml_only, stats.snort_only, stats.both];
        sourceChart.update('none');
        // Attack type chart (multiclass)
        if (stats.attack_dist && Object.keys(stats.attack_dist).length > 0) {
            attackTypeChart.data.labels   = Object.keys(stats.attack_dist);
            attackTypeChart.data.datasets[0].data = Object.values(stats.attack_dist);
            attackTypeChart.update('none');
        }

        // Recent alerts
        const alertsRes = await fetch('/api/alerts/');
        const alerts    = await alertsRes.json();

        // Filter by mode if needed
        const filtered = currentMode === 'multiclass'
            ? alerts.filter(a => a.detection_mode === 'multiclass' || a.detection_mode === 'cascade')
            : alerts.filter(a => a.detection_mode === 'binary' || a.detection_mode === 'cascade' || !a.detection_mode);

        const tbody = document.getElementById('alert-tbody');
        tbody.innerHTML = filtered.length === 0
            ? `<tr><td colspan="9"><div class="empty-state">No ${currentMode} alerts yet.</div></td></tr>`
            : filtered.map(renderRow).join('');

        document.getElementById('last-refresh').textContent =
            'Updated ' + new Date().toLocaleTimeString();

    } catch(e) {
        console.error('Dashboard refresh error:', e);
    }
}

// ── Start ─────────────────────────────────────────────────────
setMode(currentMode);
setInterval(refreshDashboard, REFRESH_INTERVAL);
