"""
Real-time Performance Monitor
Run this in a separate terminal while your backend is running
"""

import psutil
import time
import os
from datetime import datetime

def get_gpu_info():
    """Get NVIDIA GPU information"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
        
        return {
            'utilization': gpu_util.gpu,
            'memory_used_mb': mem_info.used / 1024**2,
            'memory_total_mb': mem_info.total / 1024**2,
            'memory_percent': (mem_info.used / mem_info.total) * 100,
            'temperature': temp
        }
    except:
        return None

def find_python_processes():
    """Find Python processes related to your app"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'run.py' in cmdline or 'flask' in cmdline.lower():
                    processes.append(proc)
        except:
            pass
    return processes

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_size(bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

def main():
    print("🔍 Starting Performance Monitor...")
    print("📊 Monitoring your AI system...")
    print("Press Ctrl+C to stop\n")
    time.sleep(2)
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("=" * 80)
            print(f"🚀 AI SYSTEM PERFORMANCE MONITOR - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)
            
            # Overall system stats
            cpu_percent = psutil.cpu_percent(interval=0.5, percpu=False)
            cpu_per_core = psutil.cpu_percent(interval=0, percpu=True)
            memory = psutil.virtual_memory()
            
            print("\n📊 SYSTEM OVERVIEW:")
            print(f"  CPU Usage:        {cpu_percent:.1f}%")
            print(f"  Memory Usage:     {memory.percent:.1f}% ({format_size(memory.used)} / {format_size(memory.total)})")
            print(f"  Available Memory: {format_size(memory.available)}")
            
            # Per-core CPU usage
            print(f"\n  CPU Cores: ", end="")
            for i, usage in enumerate(cpu_per_core):
                if usage > 80:
                    color = "🔴"
                elif usage > 50:
                    color = "🟡"
                else:
                    color = "🟢"
                print(f"{color}{usage:.0f}%", end="  ")
            print()
            
            # GPU info
            gpu_info = get_gpu_info()
            if gpu_info:
                print("\n🎮 GPU STATUS:")
                print(f"  GPU Usage:        {gpu_info['utilization']:.1f}%")
                print(f"  VRAM Usage:       {gpu_info['memory_percent']:.1f}% ({gpu_info['memory_used_mb']:.0f} MB / {gpu_info['memory_total_mb']:.0f} MB)")
                print(f"  Temperature:      {gpu_info['temperature']}°C")
            else:
                print("\n🎮 GPU: Not detected or not NVIDIA")
            
            # Python processes
            processes = find_python_processes()
            if processes:
                print("\n🐍 PYTHON PROCESSES:")
                print(f"{'PID':<10} {'CPU %':<10} {'Memory %':<12} {'Memory (MB)':<15}")
                print("-" * 50)
                
                total_cpu = 0
                total_mem = 0
                
                for proc in processes:
                    try:
                        proc_info = proc.as_dict(attrs=['pid', 'cpu_percent', 'memory_percent', 'memory_info'])
                        pid = proc_info['pid']
                        cpu = proc_info['cpu_percent']
                        mem_percent = proc_info['memory_percent']
                        mem_mb = proc_info['memory_info'].rss / 1024**2
                        
                        total_cpu += cpu
                        total_mem += mem_percent
                        
                        print(f"{pid:<10} {cpu:<10.1f} {mem_percent:<12.1f} {mem_mb:<15.1f}")
                    except:
                        pass
                
                print("-" * 50)
                print(f"{'TOTAL':<10} {total_cpu:<10.1f} {total_mem:<12.1f}")
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                print("\n💾 DISK I/O:")
                print(f"  Read:  {format_size(disk_io.read_bytes)}")
                print(f"  Write: {format_size(disk_io.write_bytes)}")
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                print("\n🌐 NETWORK I/O:")
                print(f"  Sent:     {format_size(net_io.bytes_sent)}")
                print(f"  Received: {format_size(net_io.bytes_recv)}")
            
            # Bottleneck detection
            print("\n🚨 BOTTLENECK ANALYSIS:")
            bottlenecks = []
            
            if cpu_percent > 85:
                bottlenecks.append("⚠️  HIGH CPU USAGE - Consider optimizing inference or reducing frame rate")
            
            if memory.percent > 85:
                bottlenecks.append("⚠️  HIGH MEMORY USAGE - Risk of system slowdown")
            
            if gpu_info and gpu_info['utilization'] < 20 and gpu_info['utilization'] > 0:
                bottlenecks.append("💡 LOW GPU USAGE - YOLO might be running on CPU")
            
            if gpu_info and gpu_info['memory_percent'] > 90:
                bottlenecks.append("⚠️  HIGH GPU MEMORY - Risk of OOM errors")
            
            if bottlenecks:
                for msg in bottlenecks:
                    print(f"  {msg}")
            else:
                print("  ✅ No major bottlenecks detected")
            
            # Tips
            print("\n💡 OPTIMIZATION TIPS:")
            if cpu_percent > 70:
                print("  • Increase YOLO_INFERENCE_INTERVAL (currently 1, try 2-3)")
                print("  • Reduce frame resolution in stream_manager.py")
                print("  • Enable frame skipping (set frame_skip_ratio > 0)")
            
            if gpu_info and gpu_info['utilization'] < 10:
                print("  • YOLO is likely running on CPU - check device='cpu' settings")
                print("  • Consider enabling GPU if available")
            
            if memory.percent > 70:
                print("  • Consider reducing tracker max_age and nn_budget")
                print("  • Increase cleanup frequency")
            
            print("\n" + "=" * 80)
            print("⏳ Refreshing in 2 seconds... (Press Ctrl+C to stop)")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped.")

if __name__ == "__main__":
    main()