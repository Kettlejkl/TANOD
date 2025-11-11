import React, { useState, useEffect } from 'react';
import { Monitor, Activity } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

const SystemHealthPanel = () => {
  const [pulse, setPulse] = useState(false);
  const [health, setHealth] = useState({
    fps: 0,
    detectionAccuracy: 0,
    anomalyPrecision: 0,
    uptime: 0,
    idSwitchRate: 0,
    activeUIDs: 0
  });
  const [isLoading, setIsLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState('connected');

  // Pulse animation
  useEffect(() => {
    const interval = setInterval(() => {
      setPulse(true);
      setTimeout(() => setPulse(false), 300);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  // Fetch system health data
  const fetchSystemHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/system-health`);
      const data = await response.json();
      
      console.log('💚 System health data:', data);
      
      if (data.success) {
        setHealth({
          fps: data.health?.fps || 0,
          detectionAccuracy: data.health?.detectionAccuracy || 0,
          anomalyPrecision: data.health?.anomalyPrecision || 0,
          uptime: data.health?.uptime || 0,
          idSwitchRate: data.health?.idSwitchRate || 0,
          activeUIDs: data.health?.activeUIDs || 0
        });
        setConnectionStatus('connected');
      }
      
      if (isLoading) setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch system health:', error);
      setConnectionStatus('disconnected');
      setIsLoading(false);
    }
  };

  // Initial fetch and polling
  useEffect(() => {
    fetchSystemHealth();
    const interval = setInterval(fetchSystemHealth, 3000); // Update every 3 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    if (connectionStatus === 'disconnected') return 'bg-red-400';
    if (health.fps < 15) return 'bg-yellow-400';
    return 'bg-emerald-400';
  };

  const getMetricColor = (value, thresholds) => {
    if (value >= thresholds.good) return 'text-emerald-400';
    if (value >= thresholds.warning) return 'text-yellow-400';
    return 'text-red-400';
  };

  if (isLoading) {
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-lg h-full flex items-center justify-center">
        <Activity className="w-8 h-8 text-emerald-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg h-full flex flex-col">
      <div className="bg-slate-800 border-b border-slate-600 px-4 py-3">
        <h3 className="font-semibold text-slate-100 flex items-center gap-2 text-sm">
          <Monitor className={`w-4 h-4 text-emerald-400 transition-transform duration-300 ${pulse ? 'scale-125' : ''}`} />
          System Health
          <span 
            className={`ml-auto w-2 h-2 rounded-full transition-opacity duration-300 ${getStatusColor()} ${pulse ? 'opacity-100' : 'opacity-50'}`} 
            title={connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
          />
        </h3>
      </div>
      <div className="flex-1 p-3">
        <div className="grid grid-cols-2 gap-2">
          {/* FPS */}
          <div className="bg-slate-800/50 rounded p-2 text-center transition-all duration-300 hover:bg-slate-800/70 hover:scale-105 cursor-pointer">
            <div className={`text-lg font-bold transition-all duration-300 ${pulse ? 'scale-110' : ''} ${getMetricColor(health.fps, { good: 20, warning: 15 })}`}>
              {health.fps}
            </div>
            <div className="text-xs text-slate-400">FPS</div>
          </div>

          {/* Detection Accuracy */}
          <div className="bg-slate-800/50 rounded p-2 text-center transition-all duration-300 hover:bg-slate-800/70 hover:scale-105 cursor-pointer">
            <div className={`text-lg font-bold transition-all duration-300 ${pulse ? 'scale-110' : ''} ${getMetricColor(health.detectionAccuracy, { good: 85, warning: 70 })}`}>
              {health.detectionAccuracy}%
            </div>
            <div className="text-xs text-slate-400">Detection</div>
          </div>

          {/* Anomaly Precision */}
          <div className="bg-slate-800/50 rounded p-2 text-center transition-all duration-300 hover:bg-slate-800/70 hover:scale-105 cursor-pointer">
            <div className={`text-lg font-bold transition-all duration-300 ${pulse ? 'scale-110' : ''} ${getMetricColor(health.anomalyPrecision, { good: 80, warning: 65 })}`}>
              {health.anomalyPrecision}%
            </div>
            <div className="text-xs text-slate-400">Precision</div>
          </div>

          {/* Uptime */}
          <div className="bg-slate-800/50 rounded p-2 text-center transition-all duration-300 hover:bg-slate-800/70 hover:scale-105 cursor-pointer">
            <div className={`text-lg font-bold transition-all duration-300 ${pulse ? 'scale-110' : ''} ${getMetricColor(health.uptime, { good: 95, warning: 85 })}`}>
              {health.uptime}%
            </div>
            <div className="text-xs text-slate-400">Uptime</div>
          </div>

          {/* ID Switch Rate */}
          <div className="bg-slate-800/50 rounded p-2 text-center transition-all duration-300 hover:bg-slate-800/70 hover:scale-105 cursor-pointer">
            <div className={`text-lg font-bold transition-all duration-300 ${pulse ? 'scale-110' : ''} text-orange-400`}>
              {health.idSwitchRate}
            </div>
            <div className="text-xs text-slate-400">ID Switch</div>
          </div>

          {/* Active UIDs */}
          <div className="bg-slate-800/50 rounded p-2 text-center transition-all duration-300 hover:bg-slate-800/70 hover:scale-105 cursor-pointer">
            <div className={`text-lg font-bold transition-all duration-300 ${pulse ? 'scale-110' : ''} text-cyan-400`}>
              {health.activeUIDs}
            </div>
            <div className="text-xs text-slate-400">Active UIDs</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemHealthPanel;