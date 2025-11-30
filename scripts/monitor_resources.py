#!/usr/bin/env python3
"""
Resource monitoring script for MLX training.

Tracks:
- Memory usage (system + estimated GPU via unified memory)
- CPU usage
- Thermal state (if available)

Run alongside training: python scripts/monitor_resources.py

For detailed GPU metrics, install asitop:
  pip install asitop
  sudo asitop
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Run: pip install psutil")


def get_memory_info():
    """Get memory usage information."""
    if not HAS_PSUTIL:
        return {}
    
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024**3),
        "used_gb": mem.used / (1024**3),
        "available_gb": mem.available / (1024**3),
        "percent": mem.percent,
    }


def get_cpu_info():
    """Get CPU usage information."""
    if not HAS_PSUTIL:
        return {}
    
    return {
        "percent": psutil.cpu_percent(interval=0.1),
        "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
    }


def get_thermal_info():
    """Get thermal information (macOS specific)."""
    try:
        # Try to get thermal state via pmset
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.lower()
        
        # Parse thermal level
        if "normal" in output:
            return {"state": "normal", "throttling": False}
        elif "moderate" in output:
            return {"state": "moderate", "throttling": True}
        elif "heavy" in output or "sleeping" in output:
            return {"state": "heavy", "throttling": True}
        else:
            return {"state": "unknown", "throttling": None}
    except Exception:
        return {"state": "unavailable", "throttling": None}


def get_gpu_memory_pressure():
    """
    Estimate GPU memory pressure.
    On Apple Silicon, GPU uses unified memory, so high system memory = high GPU memory.
    """
    if not HAS_PSUTIL:
        return {}
    
    mem = psutil.virtual_memory()
    
    # Estimate based on available memory
    # For 24GB M4: 
    # - Good: >10GB available
    # - Moderate: 5-10GB available  
    # - High: <5GB available
    available_gb = mem.available / (1024**3)
    
    if available_gb > 10:
        pressure = "low"
    elif available_gb > 5:
        pressure = "moderate"
    else:
        pressure = "high"
    
    return {
        "estimated_gpu_available_gb": available_gb,
        "pressure": pressure,
    }


def format_status_line(mem, cpu, thermal, gpu):
    """Format a single status line."""
    parts = []
    
    if mem:
        parts.append(f"RAM: {mem['used_gb']:.1f}/{mem['total_gb']:.1f}GB ({mem['percent']:.0f}%)")
    
    if cpu:
        parts.append(f"CPU: {cpu['percent']:.0f}%")
    
    if thermal and thermal.get("state") != "unavailable":
        state = thermal["state"]
        if thermal.get("throttling"):
            parts.append(f"Thermal: ⚠️ {state}")
        else:
            parts.append(f"Thermal: {state}")
    
    if gpu:
        parts.append(f"GPU mem pressure: {gpu['pressure']}")
    
    return " | ".join(parts)


def monitor_loop(interval: float = 2.0, duration: float = None):
    """Main monitoring loop."""
    print("=" * 70)
    print("Resource Monitor - Press Ctrl+C to stop")
    print("=" * 70)
    print(f"Interval: {interval}s" + (f" | Duration: {duration}s" if duration else ""))
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        while True:
            # Gather metrics
            mem = get_memory_info()
            cpu = get_cpu_info()
            thermal = get_thermal_info()
            gpu = get_gpu_memory_pressure()
            
            # Format and print
            timestamp = datetime.now().strftime("%H:%M:%S")
            status = format_status_line(mem, cpu, thermal, gpu)
            print(f"[{timestamp}] {status}")
            
            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print("\nDuration reached, stopping.")
                break
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def print_system_info():
    """Print detailed system information."""
    print("=" * 70)
    print("System Information")
    print("=" * 70)
    
    if HAS_PSUTIL:
        # Memory
        mem = psutil.virtual_memory()
        print(f"\nMemory:")
        print(f"  Total:     {mem.total / (1024**3):.1f} GB")
        print(f"  Available: {mem.available / (1024**3):.1f} GB")
        print(f"  Used:      {mem.used / (1024**3):.1f} GB ({mem.percent:.1f}%)")
        
        # CPU
        print(f"\nCPU:")
        print(f"  Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        freq = psutil.cpu_freq()
        if freq:
            print(f"  Frequency: {freq.current:.0f} MHz (max: {freq.max:.0f} MHz)")
    
    # Thermal
    thermal = get_thermal_info()
    print(f"\nThermal:")
    print(f"  State: {thermal['state']}")
    if thermal.get("throttling"):
        print("  ⚠️  System may be thermal throttling!")
    
    # MLX check
    print(f"\nMLX:")
    try:
        import mlx.core as mx
        print(f"  Version: {mx.__version__}")
        print(f"  Default device: {mx.default_device()}")
    except ImportError:
        print("  Not installed")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Monitor system resources during MLX training")
    parser.add_argument("--interval", "-i", type=float, default=2.0, 
                       help="Update interval in seconds (default: 2)")
    parser.add_argument("--duration", "-d", type=float, default=None,
                       help="Total duration to monitor in seconds (default: infinite)")
    parser.add_argument("--info", action="store_true",
                       help="Print system info and exit")
    
    args = parser.parse_args()
    
    if args.info:
        print_system_info()
    else:
        print_system_info()
        print()
        monitor_loop(interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    if not HAS_PSUTIL:
        print("Installing psutil...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "-q"])
        import psutil
        HAS_PSUTIL = True
    
    main()
