import time
import psutil
import threading
from collections import defaultdict
import numpy as np

class PerformanceProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
        self.enabled = True
        self.lock = threading.Lock()
        
        # System metrics
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        
        # Start system monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_system(self):
        """Monitor system resources in background"""
        while self.enabled:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.5)
                self.cpu_samples.append(cpu_percent)
                if len(self.cpu_samples) > 120:  # Keep last 60 seconds
                    self.cpu_samples.pop(0)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_samples.append(memory.percent)
                if len(self.memory_samples) > 120:
                    self.memory_samples.pop(0)
                
                # Try to get GPU usage (NVIDIA only)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_samples.append(gpu_util.gpu)
                    if len(self.gpu_samples) > 120:
                        self.gpu_samples.pop(0)
                except:
                    pass
                
            except Exception as e:
                pass
            
            time.sleep(0.5)
    
    def measure(self, name):
        """Context manager for timing operations"""
        return TimingContext(self, name)
    
    def record_time(self, name, duration):
        """Record a timing measurement"""
        if self.enabled:
            with self.lock:
                self.timings[name].append(duration)
                if len(self.timings[name]) > 1000:  # Keep last 1000 samples
                    self.timings[name].pop(0)
    
    def increment(self, name):
        """Increment a counter"""
        if self.enabled:
            with self.lock:
                self.counters[name] += 1
    
    def get_report(self):
        """Generate performance report"""
        with self.lock:
            report = {
                'timings': {},
                'counters': dict(self.counters),
                'system': {
                    'cpu_avg': np.mean(self.cpu_samples) if self.cpu_samples else 0,
                    'cpu_max': np.max(self.cpu_samples) if self.cpu_samples else 0,
                    'memory_avg': np.mean(self.memory_samples) if self.memory_samples else 0,
                    'memory_max': np.max(self.memory_samples) if self.memory_samples else 0,
                }
            }
            
            if self.gpu_samples:
                report['system']['gpu_avg'] = np.mean(self.gpu_samples)
                report['system']['gpu_max'] = np.max(self.gpu_samples)
            
            # Calculate timing statistics
            for name, times in self.timings.items():
                if times:
                    report['timings'][name] = {
                        'avg_ms': np.mean(times) * 1000,
                        'min_ms': np.min(times) * 1000,
                        'max_ms': np.max(times) * 1000,
                        'p95_ms': np.percentile(times, 95) * 1000,
                        'count': len(times),
                        'total_sec': sum(times)
                    }
            
            return report
    
    def print_report(self):
        """Print formatted performance report"""
        report = self.get_report()
        
        print("\n" + "="*80)
        print("🔍 PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        # System resources
        print("\n📊 SYSTEM RESOURCES:")
        print(f"  CPU Usage:    Avg: {report['system']['cpu_avg']:.1f}%  |  Max: {report['system']['cpu_max']:.1f}%")
        print(f"  Memory Usage: Avg: {report['system']['memory_avg']:.1f}%  |  Max: {report['system']['memory_max']:.1f}%")
        if 'gpu_avg' in report['system']:
            print(f"  GPU Usage:    Avg: {report['system']['gpu_avg']:.1f}%  |  Max: {report['system']['gpu_max']:.1f}%")
        
        # Timing breakdown
        print("\n⏱️  TIMING BREAKDOWN (sorted by total time):")
        timings_sorted = sorted(
            report['timings'].items(),
            key=lambda x: x[1]['total_sec'],
            reverse=True
        )
        
        print(f"{'Operation':<40} {'Avg (ms)':<12} {'Max (ms)':<12} {'Count':<10} {'Total (s)':<12}")
        print("-" * 96)
        
        for name, stats in timings_sorted[:20]:  # Top 20
            print(f"{name:<40} {stats['avg_ms']:<12.2f} {stats['max_ms']:<12.2f} {stats['count']:<10} {stats['total_sec']:<12.2f}")
        
        # Identify bottlenecks
        print("\n🚨 BOTTLENECK ANALYSIS:")
        bottlenecks = []
        
        for name, stats in timings_sorted:
            # Operations taking >50ms on average
            if stats['avg_ms'] > 50:
                bottlenecks.append((name, stats['avg_ms'], 'SLOW'))
            # Operations with high variance (max > 3x avg)
            elif stats['max_ms'] > stats['avg_ms'] * 3:
                bottlenecks.append((name, stats['max_ms'], 'SPIKY'))
        
        if bottlenecks:
            for name, time_ms, issue_type in bottlenecks[:10]:
                print(f"  ⚠️  {name}: {time_ms:.1f}ms ({issue_type})")
        else:
            print("  ✅ No significant bottlenecks detected")
        
        # FPS calculation
        print("\n🎥 FRAME PROCESSING:")
        if 'frame_processing' in report['timings']:
            fps = 1.0 / report['timings']['frame_processing']['avg_ms'] * 1000
            print(f"  Average FPS: {fps:.1f}")
            print(f"  Frame time: {report['timings']['frame_processing']['avg_ms']:.1f}ms")
        
        # Counters
        if report['counters']:
            print("\n📈 COUNTERS:")
            for name, count in sorted(report['counters'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {name}: {count:,}")
        
        print("\n" + "="*80)
    
    def reset(self):
        """Reset all statistics"""
        with self.lock:
            self.timings.clear()
            self.counters.clear()
            self.cpu_samples.clear()
            self.memory_samples.clear()
            self.gpu_samples.clear()

class TimingContext:
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.profiler.record_time(self.name, duration)

# Global profiler instance
_profiler = PerformanceProfiler()

def get_profiler():
    return _profiler

def measure(name):
    return _profiler.measure(name)

def increment(name):
    _profiler.increment(name)