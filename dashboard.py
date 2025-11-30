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
    <title>Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #fafafa;
            color: #1a1a1a;
            min-height: 100vh;
            padding: 32px 48px;
            line-height: 1.5;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid #e5e5e5;
        }
        h1 {
            font-size: 1.5em;
            font-weight: 600;
            color: #1a1a1a;
            letter-spacing: -0.02em;
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 24px;
        }
        .status { 
            display: flex; 
            align-items: center; 
            gap: 8px; 
            font-size: 0.875em;
            color: #666;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
        }
        .status-dot.running { background: #22c55e; }
        .status-dot.stopped { background: #ef4444; }
        .phase-badge {
            background: #f3f4f6;
            border: 1px solid #e5e5e5;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.75em;
            font-weight: 500;
            color: #374151;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .eta-text {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.875em;
            color: #666;
        }
        
        /* Progress */
        .progress-section {
            background: #fff;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .progress-label {
            font-size: 0.875em;
            font-weight: 500;
            color: #374151;
        }
        .progress-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.875em;
            color: #666;
        }
        .progress-bar {
            height: 6px;
            background: #e5e5e5;
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: #2563eb;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        
        /* Metrics */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: #fff;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 20px;
        }
        .metric-label {
            font-size: 0.75em;
            font-weight: 500;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .metric-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.5em;
            font-weight: 500;
            color: #1a1a1a;
        }
        
        /* Charts */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .chart-card {
            background: #fff;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 20px;
        }
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .chart-title {
            font-size: 0.875em;
            font-weight: 500;
            color: #374151;
        }
        .chart-hint {
            font-size: 0.75em;
            color: #9ca3af;
        }
        .chart-container {
            height: 240px;
        }
        
        /* Samples - Enhanced */
        .samples-section {
            background: #fff;
            border: 1px solid #e5e5e5;
            border-radius: 12px;
            padding: 0;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .samples-header-bar {
            padding: 16px 24px;
            border-bottom: 1px solid #e5e5e5;
            background: #f8fafc;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .samples-title {
            font-size: 0.95em;
            font-weight: 600;
            color: #1e293b;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .samples-container {
            padding: 24px;
            background: #f8fafc; /* Light background */
            max-height: 600px;
            overflow-y: auto;
        }
        .sample-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        .sample-card:last-child { margin-bottom: 0; }
        .sample-meta {
            background: #f1f5f9;
            padding: 8px 16px;
            font-family: 'Inter', sans-serif;
            font-size: 0.75em;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #e2e8f0;
        }
        .sample-content {
            padding: 16px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            color: #334155;
            white-space: pre-wrap;
        }
        .prompt-text {
            color: #2563eb; /* Blue for prompt */
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Training Dashboard</h1>
            <div class="header-right">
                <div class="status">
                    <div class="status-dot" id="statusDot"></div>
                    <span id="statusText">Connecting</span>
                </div>
                <span class="phase-badge" id="phase">Phase 1</span>
                <span class="eta-text">ETA: <span id="eta">--</span></span>
            </div>
        </div>
        
        <div class="progress-section">
            <div class="progress-header">
                <span class="progress-label">Training Progress</span>
                <span class="progress-value"><span id="tokensM">0</span>M tokens (<span id="progressPct">0.0</span>%) → 3B target</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Step</div>
                <div class="metric-value" id="step">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Loss</div>
                <div class="metric-value" id="loss">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Perplexity</div>
                <div class="metric-value" id="ppl">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Throughput</div>
                <div class="metric-value" id="tokS">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Learning Rate</div>
                <div class="metric-value" id="lr">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Memory</div>
                <div class="metric-value" id="gpuMem">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Util</div>
                <div class="metric-value" id="gpuUtil">—</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-title">Loss</span>
                    <span class="chart-hint">Drag to zoom • Double-click to reset</span>
                </div>
                <div class="chart-container" id="lossChart"></div>
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-title">Perplexity</span>
                    <span class="chart-hint">Drag to zoom • Double-click to reset</span>
                </div>
                <div class="chart-container" id="pplChart"></div>
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-title">Throughput (tokens/sec)</span>
                    <span class="chart-hint">Drag to zoom • Double-click to reset</span>
                </div>
                <div class="chart-container" id="tokChart"></div>
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <span class="chart-title">GPU Utilization (%)</span>
                    <span class="chart-hint">Drag to zoom • Double-click to reset</span>
                </div>
                <div class="chart-container" id="utilChart"></div>
            </div>
        </div>
        
        <div class="samples-section">
            <div class="samples-header-bar">
                <span class="samples-title">✨ Generated Samples</span>
                <span style="font-size: 0.8em; color: #64748b;">Updates every 500 steps</span>
            </div>
            <div class="samples-container" id="samples">
                <div style="color: #64748b; text-align: center; padding: 40px;">Waiting for first samples...</div>
            </div>
        </div>
    </div>
    
    <script>
        // Plotly chart configuration
        const layout = {
            margin: { l: 50, r: 20, t: 10, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'Inter, sans-serif', size: 11, color: '#6b7280' },
            xaxis: { 
                gridcolor: '#e5e5e5', 
                linecolor: '#e5e5e5',
                tickfont: { size: 10 }
            },
            yaxis: { 
                gridcolor: '#e5e5e5', 
                linecolor: '#e5e5e5',
                tickfont: { size: 10 }
            },
            hovermode: 'x unified'
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false
        };
        
        // Initialize empty charts
        Plotly.newPlot('lossChart', [{x: [], y: [], type: 'scatter', mode: 'lines', line: {color: '#2563eb', width: 2}}], layout, config);
        Plotly.newPlot('pplChart', [{x: [], y: [], type: 'scatter', mode: 'lines', line: {color: '#059669', width: 2}}], layout, config);
        Plotly.newPlot('tokChart', [{x: [], y: [], type: 'scatter', mode: 'lines', line: {color: '#d97706', width: 2}}], layout, config);
        Plotly.newPlot('utilChart', [{x: [], y: [], type: 'scatter', mode: 'lines', line: {color: '#8b5cf6', width: 2}}], layout, config);
        
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
                    const step = data.steps[last];
                    document.getElementById('step').textContent = step.toLocaleString();
                    document.getElementById('loss').textContent = data.losses[last].toFixed(3);
                    document.getElementById('ppl').textContent = data.ppls[last].toFixed(1);
                    document.getElementById('tokS').textContent = Math.round(data.tok_s[last]).toLocaleString();
                    document.getElementById('gpuMem').textContent = data.gpu_mem[last].toFixed(1) + ' GB';
                    document.getElementById('gpuUtil').textContent = (data.gpu_utils ? data.gpu_utils[last] : 0) + '%';
                    
                    if (data.lrs && data.lrs[last]) {
                        document.getElementById('lr').textContent = data.lrs[last];
                    }
                    
                    // Calculate tokens from step (batch=64 * seq=512 = 32768 tokens/step)
                    const tokens = step * 32768;
                    const tokensM = Math.round(tokens / 1e6);
                    document.getElementById('tokensM').textContent = tokensM.toLocaleString();
                    
                    const progress = (tokens / 3e9) * 100;
                    document.getElementById('progressBar').style.width = Math.min(progress, 100).toFixed(1) + '%';
                    document.getElementById('progressPct').textContent = progress.toFixed(2);
                }
                
                document.getElementById('eta').textContent = data.eta || '--';
                
                // Format phase nicely
                const phase = data.current_phase || 'phase 1';
                const phaseMap = {
                    'tin+cos': 'Stories + Cosmo',
                    'tin+cos+ope+pyt': 'Stories + Math + Code', 
                    'cos+ope+pyt+met': 'Full Curriculum'
                };
                document.getElementById('phase').textContent = phaseMap[phase] || phase.toUpperCase();
                
                // Update Plotly charts
                if (data.steps.length > 0) {
                    Plotly.react('lossChart', [{x: data.steps, y: data.losses, type: 'scatter', mode: 'lines', line: {color: '#2563eb', width: 2}, hovertemplate: 'Step %{x}<br>Loss: %{y:.4f}<extra></extra>'}], layout, config);
                    Plotly.react('pplChart', [{x: data.steps, y: data.ppls, type: 'scatter', mode: 'lines', line: {color: '#059669', width: 2}, hovertemplate: 'Step %{x}<br>PPL: %{y:.2f}<extra></extra>'}], layout, config);
                    Plotly.react('tokChart', [{x: data.steps, y: data.tok_s, type: 'scatter', mode: 'lines', line: {color: '#d97706', width: 2}, hovertemplate: 'Step %{x}<br>Tok/s: %{y:,.0f}<extra></extra>'}], layout, config);
                    Plotly.react('utilChart', [{x: data.steps, y: data.gpu_utils || [], type: 'scatter', mode: 'lines', line: {color: '#8b5cf6', width: 2}, hovertemplate: 'Step %{x}<br>Util: %{y}%<extra></extra>'}], layout, config);
                }
                
                // Update samples
                if (data.samples && data.samples.length > 0) {
                    // Take last 6 samples (2 sets of 3 prompts)
                    const recentSamples = data.samples.slice(-6).reverse();
                    
                    document.getElementById('samples').innerHTML = recentSamples.map(s => {
                        // Highlight the prompt part if possible (simple heuristic)
                        let content = s.text;
                        const promptMatch = content.match(/^(.*?)(?=\s)/); // Try to find first word/sentence
                        
                        return `
                        <div class="sample-card">
                            <div class="sample-meta">
                                <span>${s.header}</span>
                                <span>MODEL OUTPUT</span>
                            </div>
                            <div class="sample-content">${content}</div>
                        </div>
                        `;
                    }).join('');
                }
                
            } catch (e) {
                console.error('Update failed:', e);
            }
        }
        
        setInterval(updateData, 2000);
        updateData();
    </script>
</body>
</html>
'''


def parse_log_line(line: str) -> dict:
    """Parse a log line into metrics."""
    # Match: step=  1000 | tokens=0.03B | loss=2.34 | ppl=10.4 | lr=5.00e-04 | grad=0.45 | tok/s=50000 | gpu_mem=18.2GB | gpu_util=98% | eta=2.1h | phase=tin+cos
    pattern = r'step=\s*(\d+).*?(?:tokens=(\d+\.?\d*)B)?.*?loss=(\d+\.?\d+).*?ppl=(\d+\.?\d+).*?lr=(\S+).*?grad=(\d+\.?\d+).*?tok/s=(\d+).*?gpu_mem=(\d+\.?\d+)GB.*?gpu_util=(\d+)%.*?eta=(\S+)(?:.*?phase=(\S+))?'
    
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
            "gpu_util": int(match.group(9)),
            "eta": match.group(10),
            "phase": match.group(11) if match.group(11) else "",
        }
    return None


def parse_sample(lines: list, start_idx: int) -> dict:
    """Parse a sample generation from log lines."""
    header = lines[start_idx].strip()
    text_lines = []
    for i in range(start_idx + 1, min(start_idx + 10000, len(lines))):
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
            
            steps, losses, ppls, lrs, tok_s, tokens_b, grad_norms, gpu_mem, gpu_utils = [], [], [], [], [], [], [], [], []
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
                    gpu_utils.append(metrics["gpu_util"])
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
                "gpu_utils": gpu_utils,
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
