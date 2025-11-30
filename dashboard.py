#!/usr/bin/env python3
"""
📊 LIVE TRAINING DASHBOARD

Real-time training metrics visualization on port 3000.

Usage:
    python dashboard.py                    # Auto-find latest run
    python dashboard.py --run checkpoints/curriculum_150M_*
    
Then open: http://localhost:3000
"""

import argparse
import json
import re
import time
from pathlib import Path
from threading import Thread
from datetime import datetime

from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

# Global state
CURRENT_RUN = None
LOG_DATA = {
    "steps": [],
    "losses": [],
    "ppls": [],
    "lrs": [],
    "tok_s": [],
    "tokens_b": [],
    "grad_norms": [],
    "gpu_mem": [],
    "samples": [],
    "config": {},
    "status": "initializing",
    "current_phase": "",
    "eta": "",
    "last_update": "",
}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🧠 Mini-Model Training</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 24px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 24px;
            color: #fff;
            letter-spacing: -0.5px;
        }
        h1 span { 
            background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.1));
            border: 1px solid rgba(139,92,246,0.2);
            padding: 16px 24px;
            border-radius: 16px;
            margin-bottom: 24px;
        }
        .status { display: flex; align-items: center; gap: 12px; font-weight: 500; }
        .status-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.running { background: #22c55e; box-shadow: 0 0 12px rgba(34,197,94,0.5); }
        .status-dot.stopped { background: #ef4444; animation: none; }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(0.95); }
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e, #1e1e3a);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 20px 16px;
            border-radius: 16px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'JetBrains Mono', monospace;
        }
        .metric-label { 
            color: #888; 
            font-size: 0.75em; 
            margin-top: 8px; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }
        .chart-card {
            background: linear-gradient(135deg, #1a1a2e, #1e1e3a);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 20px;
            border-radius: 16px;
        }
        .chart-title { 
            font-size: 0.9em; 
            margin-bottom: 16px; 
            color: #a0a0a0;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        canvas { max-height: 220px; }
        .samples-section {
            background: linear-gradient(135deg, #1a1a2e, #1e1e3a);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 20px;
            border-radius: 16px;
        }
        .sample {
            background: rgba(0,0,0,0.4);
            padding: 16px;
            border-radius: 12px;
            margin: 12px 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85em;
            line-height: 1.6;
            white-space: pre-wrap;
            border-left: 3px solid #8b5cf6;
        }
        .sample-header { 
            color: #a855f7; 
            margin-bottom: 8px; 
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        }
        .phase-indicator {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .eta { 
            font-family: 'JetBrains Mono', monospace;
            color: #888;
            font-size: 0.9em;
        }
        .progress-section {
            background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.05));
            border: 1px solid rgba(139,92,246,0.2);
            padding: 16px 24px;
            border-radius: 12px;
            margin-bottom: 24px;
        }
        .progress-bar {
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #a855f7);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .progress-text {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            color: #888;
        }
        @media (max-width: 1200px) {
            .metrics-grid { grid-template-columns: repeat(3, 1fr); }
        }
        @media (max-width: 768px) {
            .charts-grid { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><span>Mini-Model</span> Training Dashboard</h1>
        
        <div class="status-bar">
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
            <div>
                <span class="phase-indicator" id="phase">PHASE 1</span>
            </div>
            <div class="eta">ETA: <span id="eta">--</span></div>
        </div>
        
        <div class="progress-section">
            <div class="progress-text">
                <span>Progress: <span id="progressPct">0%</span></span>
                <span><span id="tokensB">0.00</span>B / 3.0B tokens</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="step">-</div>
                <div class="metric-label">Step</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="loss">-</div>
                <div class="metric-label">Loss</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="ppl">-</div>
                <div class="metric-label">Perplexity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="tokS">-</div>
                <div class="metric-label">Tokens/sec</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="lr">-</div>
                <div class="metric-label">Learning Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="gpuMem">-</div>
                <div class="metric-label">GPU Memory</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-title">📉 Loss</div>
                <canvas id="lossChart"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">📈 Perplexity</div>
                <canvas id="pplChart"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">⚡ Throughput</div>
                <canvas id="tokChart"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">📊 Gradient Norm</div>
                <canvas id="gradChart"></canvas>
            </div>
        </div>
        
        <div class="samples-section">
            <div class="chart-title">📝 Recent Samples</div>
            <div id="samples">No samples yet...</div>
        </div>
    </div>
    
    <script>
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: true,
            animation: { duration: 0 },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#888' } },
                y: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#888' } }
            },
            plugins: { legend: { display: false } }
        };
        
        const lossChart = new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#8b5cf6', borderWidth: 2, pointRadius: 0, tension: 0.3 }] },
            options: chartOptions
        });
        
        const pplChart = new Chart(document.getElementById('pplChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#22c55e', borderWidth: 2, pointRadius: 0, tension: 0.3 }] },
            options: chartOptions
        });
        
        const tokChart = new Chart(document.getElementById('tokChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#f59e0b', borderWidth: 2, pointRadius: 0, tension: 0.3 }] },
            options: chartOptions
        });
        
        const gradChart = new Chart(document.getElementById('gradChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#ef4444', borderWidth: 2, pointRadius: 0, tension: 0.3 }] },
            options: chartOptions
        });
        
        async function updateData() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();
                
                // Update status
                const dot = document.getElementById('statusDot');
                const statusText = document.getElementById('statusText');
                if (data.status === 'running') {
                    dot.className = 'status-dot running';
                    statusText.textContent = 'Training';
                } else {
                    dot.className = 'status-dot stopped';
                    statusText.textContent = data.status;
                }
                
                // Update metrics
                if (data.steps.length > 0) {
                    const last = data.steps.length - 1;
                    document.getElementById('step').textContent = data.steps[last].toLocaleString();
                    document.getElementById('loss').textContent = data.losses[last].toFixed(3);
                    document.getElementById('ppl').textContent = data.ppls[last].toFixed(1);
                    document.getElementById('tokS').textContent = Math.round(data.tok_s[last]).toLocaleString();
                    document.getElementById('tokensB').textContent = data.tokens_b[last].toFixed(2);
                    document.getElementById('gpuMem').textContent = data.gpu_mem[last].toFixed(1) + ' GB';
                    
                    // Learning rate
                    if (data.lrs && data.lrs[last]) {
                        document.getElementById('lr').textContent = data.lrs[last];
                    }
                    
                    // Progress bar (3B tokens target)
                    const progress = (data.tokens_b[last] / 3.0) * 100;
                    document.getElementById('progressBar').style.width = progress.toFixed(1) + '%';
                    document.getElementById('progressPct').textContent = progress.toFixed(1) + '%';
                }
                
                document.getElementById('eta').textContent = data.eta || '--';
                document.getElementById('phase').textContent = (data.current_phase || 'phase 1').toUpperCase();
                
                // Update charts (show last 200 points)
                const maxPoints = 200;
                const steps = data.steps.slice(-maxPoints);
                
                lossChart.data.labels = steps;
                lossChart.data.datasets[0].data = data.losses.slice(-maxPoints);
                lossChart.update();
                
                pplChart.data.labels = steps;
                pplChart.data.datasets[0].data = data.ppls.slice(-maxPoints);
                pplChart.update();
                
                tokChart.data.labels = steps;
                tokChart.data.datasets[0].data = data.tok_s.slice(-maxPoints);
                tokChart.update();
                
                gradChart.data.labels = steps;
                gradChart.data.datasets[0].data = data.grad_norms.slice(-maxPoints);
                gradChart.update();
                
                // Update samples
                if (data.samples.length > 0) {
                    document.getElementById('samples').innerHTML = data.samples.slice(-3).map(s => 
                        `<div class="sample"><div class="sample-header">${s.header}</div>${s.text}</div>`
                    ).join('');
                }
                
            } catch (e) {
                console.error('Update failed:', e);
            }
        }
        
        // Update every 2 seconds
        setInterval(updateData, 2000);
        updateData();
    </script>
</body>
</html>
'''


def parse_log_line(line: str) -> dict:
    """Parse a log line into metrics."""
    # Match: step=  1000 | tokens=0.03B | loss=2.34 | ppl=10.4 | lr=5.00e-04 | grad=0.45 | tok/s=50000 | gpu=18.2GB | eta=2.1h | phase=tin+cos
    pattern = r'step=\s*(\d+).*?(?:tokens=(\d+\.?\d*)B)?.*?loss=(\d+\.?\d+).*?ppl=(\d+\.?\d+).*?lr=(\S+).*?grad=(\d+\.?\d+).*?tok/s=(\d+).*?gpu=(\d+\.?\d+).*?eta=(\S+)(?:.*?phase=(\S+))?'
    
    match = re.search(pattern, line)
    if match:
        return {
            "step": int(match.group(1)),
            "tokens_b": float(match.group(2)) if match.group(2) else 0,
            "loss": float(match.group(3)),
            "ppl": float(match.group(4)),
            "lr": match.group(5),
            "grad_norm": float(match.group(6)),
            "tok_s": int(match.group(7)),
            "gpu_mem": float(match.group(8)),
            "eta": match.group(9),
            "phase": match.group(10) if match.group(10) else "",
        }
    return None


def parse_sample(lines: list, start_idx: int) -> dict:
    """Parse a sample generation from log lines."""
    header = lines[start_idx].strip()
    text_lines = []
    for i in range(start_idx + 1, min(start_idx + 5, len(lines))):
        if lines[i].strip().startswith("step=") or lines[i].strip().startswith("📝"):
            break
        text_lines.append(lines[i].strip())
    return {"header": header, "text": "\n".join(text_lines)}


def watch_log_file():
    """Background thread to watch log file."""
    global LOG_DATA, CURRENT_RUN
    
    last_size = 0
    
    while True:
        try:
            if CURRENT_RUN is None:
                time.sleep(2)
                continue
            
            log_file = Path(CURRENT_RUN) / "training.log"
            if not log_file.exists():
                LOG_DATA["status"] = "waiting for log..."
                time.sleep(2)
                continue
            
            # Check if file changed
            current_size = log_file.stat().st_size
            if current_size == last_size:
                time.sleep(1)
                continue
            
            last_size = current_size
            LOG_DATA["status"] = "running"
            
            # Read and parse log
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            steps, losses, ppls, lrs, tok_s, tokens_b, grad_norms, gpu_mem = [], [], [], [], [], [], [], []
            samples = []
            current_phase = ""
            eta = ""
            
            for i, line in enumerate(lines):
                metrics = parse_log_line(line)
                if metrics:
                    steps.append(metrics["step"])
                    losses.append(metrics["loss"])
                    ppls.append(metrics["ppl"])
                    lrs.append(metrics["lr"])
                    tok_s.append(metrics["tok_s"])
                    tokens_b.append(metrics["tokens_b"])
                    grad_norms.append(metrics["grad_norm"])
                    gpu_mem.append(metrics["gpu_mem"])
                    eta = metrics["eta"]
                    if metrics["phase"]:
                        current_phase = metrics["phase"]
                
                # Check for samples
                if "📝 Sample" in line:
                    sample = parse_sample(lines, i)
                    if sample["text"]:
                        samples.append(sample)
            
            LOG_DATA.update({
                "steps": steps,
                "losses": losses,
                "ppls": ppls,
                "lrs": lrs,
                "tok_s": tok_s,
                "tokens_b": tokens_b,
                "grad_norms": grad_norms,
                "gpu_mem": gpu_mem,
                "samples": samples,
                "eta": eta,
                "current_phase": current_phase,
                "last_update": datetime.now().isoformat(),
            })
            
        except Exception as e:
            LOG_DATA["status"] = f"error: {str(e)[:50]}"
        
        time.sleep(1)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/data')
def get_data():
    return jsonify(LOG_DATA)


def find_latest_run(base_dir: str = "checkpoints") -> str:
    """Find the most recent training run."""
    base = Path(base_dir)
    if not base.exists():
        return None
    
    runs = sorted(base.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
    for run in runs:
        if (run / "training.log").exists() or (run / "config.json").exists():
            return str(run)
    
    return str(runs[0]) if runs else None


def main():
    global CURRENT_RUN
    
    parser = argparse.ArgumentParser(description="Live Training Dashboard")
    parser.add_argument("--run", type=str, default=None, help="Path to run directory")
    parser.add_argument("--port", type=int, default=3000, help="Port to run on")
    args = parser.parse_args()
    
    # Find run directory
    if args.run:
        CURRENT_RUN = args.run
    else:
        CURRENT_RUN = find_latest_run()
    
    if CURRENT_RUN:
        print(f"📊 Watching: {CURRENT_RUN}")
    else:
        print("📊 No run found yet, will auto-detect...")
    
    # Start log watcher thread
    watcher = Thread(target=watch_log_file, daemon=True)
    watcher.start()
    
    print(f"\n🚀 Dashboard running at: http://localhost:{args.port}")
    print("   Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == "__main__":
    main()
