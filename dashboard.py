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
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 20px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.1);
            padding: 15px 25px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .status { display: flex; align-items: center; gap: 10px; }
        .status-dot {
            width: 12px; height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.running { background: #00ff88; }
        .status-dot.stopped { background: #ff4444; animation: none; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #00d9ff;
        }
        .metric-label { color: #888; font-size: 0.9em; margin-top: 5px; }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
        }
        .chart-title { font-size: 1.1em; margin-bottom: 15px; color: #00d9ff; }
        canvas { max-height: 250px; }
        .samples-section {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
        }
        .sample {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            border-left: 3px solid #00ff88;
        }
        .sample-header { color: #00d9ff; margin-bottom: 5px; font-weight: bold; }
        .phase-indicator {
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: #1a1a2e;
        }
        @media (max-width: 768px) {
            .charts-grid { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Mini-Model Training Dashboard</h1>
        
        <div class="status-bar">
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
            <div>
                <span class="phase-indicator" id="phase">Phase 1</span>
            </div>
            <div id="eta">ETA: --</div>
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
                <div class="metric-value" id="tokensB">-</div>
                <div class="metric-label">Tokens (B)</div>
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
            data: { labels: [], datasets: [{ data: [], borderColor: '#00d9ff', tension: 0.1 }] },
            options: chartOptions
        });
        
        const pplChart = new Chart(document.getElementById('pplChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#00ff88', tension: 0.1 }] },
            options: chartOptions
        });
        
        const tokChart = new Chart(document.getElementById('tokChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#ff9f43', tension: 0.1 }] },
            options: chartOptions
        });
        
        const gradChart = new Chart(document.getElementById('gradChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#ff6b6b', tension: 0.1 }] },
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
                    statusText.textContent = 'Training...';
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
                }
                
                document.getElementById('eta').textContent = 'ETA: ' + (data.eta || '--');
                document.getElementById('phase').textContent = data.current_phase || 'Phase 1';
                
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
