// ── IDS-KMUTT Dashboard — Main JS ────────────────────────────
const REFRESH_INTERVAL = 5000; // 5 seconds

// ── Chart initialization ─────────────────────────────────────
const trafficChart = new Chart(
    document.getElementById('trafficChart').getContext('2d'), {
    type: 'doughnut',
    data: {
        labels: ['BENIGN', 'ALERT', 'THREAT'],
        datasets: [{
            data: [0, 0, 0],
            backgroundColor: ['#22c55e', '#f97316', '#ef4444'],
            borderColor: '#111827',
            borderWidth: 3,
            hoverOffset: 6
        }]
    },
    options: {
        cutout: '65%',
        plugins: {
            legend: {
                position: 'bottom',
                labels: { color: '#94a3b8', padding: 16, font: { size: 12 } }
            }
        }
    }
});

const sourceChart = new Chart(
    document.getElementById('sourceChart').getContext('2d'), {
    type: 'bar',
    data: {
        labels: ['ML Only', 'Snort Only', 'Both', 'None'],
        datasets: [{
            label: 'Alerts',
            data: [0, 0, 0, 0],
            backgroundColor: [
                'rgba(59,130,246,0.7)',
                'rgba(168,85,247,0.7)',
                'rgba(239,68,68,0.7)',
                'rgba(100,116,139,0.3)'
            ],
            borderColor: [
                '#3b82f6', '#a855f7', '#ef4444', '#64748b'
            ],
            borderWidth: 1,
            borderRadius: 4,
        }]
    },
    options: {
        scales: {
            x: {
                ticks: { color: '#94a3b8', font: { size: 11 } },
                grid: { color: 'rgba(30,45,69,0.8)' }
            },
            y: {
                ticks: { color: '#94a3b8', font: { size: 11 } },
                grid: { color: 'rgba(30,45,69,0.8)' },
                beginAtZero: true
            }
        },
        plugins: { legend: { display: false } }
    }
});

// ── Confidence bar helper ────────────────────────────────────
function confBar(value) {
    const pct   = Math.round(value * 100);
    const level = pct >= 85 ? 'high' : pct >= 60 ? 'med' : 'low';
    return `
        <div class="conf-bar">
            <div class="bar"><div class="fill ${level}" style="width:${pct}%"></div></div>
            <span>${pct}%</span>
        </div>`;
}

// ── Render alert row ─────────────────────────────────────────
function renderRow(a) {
    const ts = new Date(a.timestamp).toLocaleString('en-GB', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
    return `
        <tr>
            <td style="color:#64748b">#${a.id}</td>
            <td>${ts}</td>
            <td style="font-family:monospace">${a.src_ip || '—'}</td>
            <td style="font-family:monospace">${a.dst_ip || '—'}</td>
            <td><span class="badge ${a.decision}">${a.decision}</span></td>
            <td><span class="tag ${a.source_tag}">${a.source_tag.replace('_', ' ')}</span></td>
            <td>${confBar(a.ml_confidence)}</td>
            <td style="color:#64748b;font-family:monospace">${a.snort_sid || '—'}</td>
        </tr>`;
}

// ── Main refresh function ────────────────────────────────────
async function refreshDashboard() {
    try {
        const res    = await fetch('/api/alerts/');
        const alerts = await res.json();

        // Stats
        const threats = alerts.filter(a => a.decision === 'THREAT').length;
        const alrts   = alerts.filter(a => a.decision === 'ALERT').length;
        const benign  = alerts.filter(a => a.decision === 'BENIGN').length;

        document.getElementById('stat-total').textContent  = alerts.length;
        document.getElementById('stat-threat').textContent = threats;
        document.getElementById('stat-alert').textContent  = alrts;
        document.getElementById('stat-benign').textContent = benign;

        // Traffic chart
        trafficChart.data.datasets[0].data = [benign, alrts, threats];
        trafficChart.update('none');

        // Source chart
        const mlOnly    = alerts.filter(a => a.source_tag === 'ML_ONLY').length;
        const snortOnly = alerts.filter(a => a.source_tag === 'SNORT_ONLY').length;
        const both      = alerts.filter(a => a.source_tag === 'BOTH').length;
        const none      = alerts.filter(a => a.source_tag === 'NONE').length;
        sourceChart.data.datasets[0].data = [mlOnly, snortOnly, both, none];
        sourceChart.update('none');

        // Table
        const tbody = document.getElementById('alert-tbody');
        tbody.innerHTML = alerts.length === 0
            ? `<tr><td colspan="8"><div class="empty-state">
                🛡️ No alerts yet — system monitoring active
               </div></td></tr>`
            : alerts.map(renderRow).join('');

        // Last refresh
        document.getElementById('last-refresh').textContent =
            'Updated ' + new Date().toLocaleTimeString();

    } catch(e) {
        console.error('Dashboard refresh error:', e);
    }
}

// ── Start ────────────────────────────────────────────────────
refreshDashboard();
setInterval(refreshDashboard, REFRESH_INTERVAL);