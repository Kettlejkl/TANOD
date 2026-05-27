import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp, Zap, Target, FileText, Download, ArrowUp, ArrowDown, Lock } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, BarChart, Bar, Tooltip } from 'recharts';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

const AnalyticsDashboard = ({ onExportClick }) => {
  const [analytics, setAnalytics] = useState({
    totalIncidents: 0,
    avgResponseTime: 0,
    trackedPersons: 0,
    uniqueVisitors: 0,
    incidentTrends: [],
    incidentHotspots: []
  });
  
  const [animatedMetrics, setAnimatedMetrics] = useState({
    totalIncidents: 0,
    avgResponseTime: 0,
    trackedPersons: 0,
    uniqueVisitors: 0
  });
  
  const [selectedReport, setSelectedReport] = useState(null);
  const [hoveredHotspot, setHoveredHotspot] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const hasAnimated = useRef(false);

  // Check authentication status
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    setIsAuthenticated(!!token);
  }, []);

  // Fetch analytics data
  const fetchAnalytics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/analytics`);
      const data = await response.json();
      
      if (data.success) {
        setAnalytics(data.analytics || {
          totalIncidents: 0,
          avgResponseTime: 0,
          trackedPersons: 0,
          uniqueVisitors: 0,
          incidentTrends: [],
          incidentHotspots: []
        });
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
      setIsLoading(false);
    }
  };

  // Initial fetch and polling
  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 5000);
    return () => clearInterval(interval);
  }, []);

  // Initial count-up animation on mount
  useEffect(() => {
    if (hasAnimated.current || isLoading) return;
    
    hasAnimated.current = true;
    const duration = 1000;
    const steps = 60;
    const interval = duration / steps;
    let currentStep = 0;

    const timer = setInterval(() => {
      currentStep++;
      const progress = Math.min(currentStep / steps, 1);
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      
      setAnimatedMetrics({
        totalIncidents: Math.floor((analytics.totalIncidents || 0) * easeProgress),
        avgResponseTime: parseFloat(((analytics.avgResponseTime || 0) * easeProgress).toFixed(1)),
        trackedPersons: Math.floor((analytics.trackedPersons || 0) * easeProgress),
        uniqueVisitors: Math.floor((analytics.uniqueVisitors || 0) * easeProgress)
      });

      if (currentStep >= steps) {
        clearInterval(timer);
      }
    }, interval);

    return () => clearInterval(timer);
  }, [isLoading, analytics.totalIncidents, analytics.avgResponseTime, analytics.trackedPersons, analytics.uniqueVisitors]);

  // Update metrics smoothly when analytics changes
  useEffect(() => {
    if (!hasAnimated.current) return;
    
    setAnimatedMetrics({
      totalIncidents: analytics.totalIncidents || 0,
      avgResponseTime: analytics.avgResponseTime || 0,
      trackedPersons: analytics.trackedPersons || 0,
      uniqueVisitors: analytics.uniqueVisitors || 0
    });
  }, [analytics.totalIncidents, analytics.avgResponseTime, analytics.trackedPersons, analytics.uniqueVisitors]);

  // Calculate trend indicators
  const getTrend = () => {
    const trends = analytics.incidentTrends || [];
    if (trends.length < 2) return 0;
    const latest = trends[trends.length - 1]?.incidents || 0;
    const previous = trends[trends.length - 2]?.incidents || 0;
    return latest - previous;
  };

  const trend = getTrend();

  // Export functions
  const exportBehaviors = async () => {
    setIsExporting(true);
    try {
      const now = new Date();
      const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      
      const response = await fetch(`${API_BASE_URL}/analytics/behaviors/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_time: yesterday.toISOString(),
          end_time: now.toISOString(),
          filename: `behaviors_${now.toISOString().split('T')[0]}.csv`
        })
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `behaviors_${now.toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('No behavior data available to export');
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed: ' + error.message);
    } finally {
      setIsExporting(false);
    }
  };

  const exportDetections = async () => {
    setIsExporting(true);
    try {
      const now = new Date();
      const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      
      const response = await fetch(`${API_BASE_URL}/analytics/detections/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_time: yesterday.toISOString(),
          end_time: now.toISOString(),
          filename: `detections_${now.toISOString().split('T')[0]}.csv`
        })
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `detections_${now.toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('No detection data available to export');
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed: ' + error.message);
    } finally {
      setIsExporting(false);
    }
  };

  const exportJourneys = async () => {
    setIsExporting(true);
    try {
      const now = new Date();
      const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      
      const response = await fetch(`${API_BASE_URL}/analytics/journeys/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_time: yesterday.toISOString(),
          end_time: now.toISOString(),
          filename: `journeys_${now.toISOString().split('T')[0]}.csv`
        })
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `journeys_${now.toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('No journey data available to export');
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed: ' + error.message);
    } finally {
      setIsExporting(false);
    }
  };

  const handleExport = (reportType) => {
    // Check authentication first
    if (!isAuthenticated) {
      if (onExportClick) {
        onExportClick(); // Trigger login modal
      }
      return;
    }

    setSelectedReport(reportType);
    
    switch(reportType) {
      case 'behaviors':
        exportBehaviors();
        break;
      case 'detections':
        exportDetections();
        break;
      case 'journeys':
        exportJourneys();
        break;
      default:
        break;
    }
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs">
          <p className="text-slate-300">{`${payload[0].value} incidents`}</p>
        </div>
      );
    }
    return null;
  };

  const CustomBarTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs">
          <p className="text-slate-300 font-semibold">{payload[0].payload.zone}</p>
          <p className="text-purple-400">{`${payload[0].value} incidents`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid grid-cols-2 grid-rows-2 gap-3 h-full">
      {/* Incident Trends */}
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-slate-100 flex items-center gap-2 text-sm">
            <TrendingUp className="w-4 h-4 text-blue-400" />
            Incident Trends
          </h3>
          {trend !== 0 && (
            <span className={`flex items-center gap-1 text-xs font-medium ${
              trend > 0 ? 'text-red-400' : 'text-emerald-400'
            }`}>
              {trend > 0 ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
              {Math.abs(trend)}
            </span>
          )}
        </div>
        <div className="h-24">
          {(analytics.incidentTrends || []).length === 0 ? (
            <div className="flex items-center justify-center h-full text-slate-500 text-xs">
              No trend data available
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={analytics.incidentTrends}>
                <defs>
                  <linearGradient id="redGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <Tooltip content={<CustomTooltip />} />
                <Area 
                  type="monotone" 
                  dataKey="incidents" 
                  stroke="#ef4444" 
                  fill="url(#redGradient)" 
                  strokeWidth={2}
                  animationDuration={800}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
      
      {/* Hotspots */}
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-3">
        <h3 className="font-semibold text-slate-100 mb-2 flex items-center gap-2 text-sm">
          <Zap className="w-4 h-4 text-purple-400" />
          Hotspots
          {hoveredHotspot && (
            <span className="text-xs text-slate-400 ml-auto">
              {hoveredHotspot}
            </span>
          )}
        </h3>
        <div className="h-24">
          {(analytics.incidentHotspots || []).length === 0 ? (
            <div className="flex items-center justify-center h-full text-slate-500 text-xs">
              No hotspot data available
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart 
                data={analytics.incidentHotspots}
                onMouseMove={(e) => {
                  if (e.activePayload) {
                    setHoveredHotspot(e.activePayload[0].payload.zone);
                  }
                }}
                onMouseLeave={() => setHoveredHotspot(null)}
              >
                <XAxis dataKey="zone" tick={{ fontSize: 9, fill: '#94a3b8' }} />
                <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <Tooltip content={<CustomBarTooltip />} />
                <Bar 
                  dataKey="count" 
                  fill="#8b5cf6" 
                  radius={[2, 2, 0, 0]}
                  animationDuration={800}
                />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
      
      {/* Key Metrics */}
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-3">
        <h3 className="font-semibold text-slate-100 mb-2 flex items-center gap-2 text-sm">
          <Target className="w-4 h-4 text-emerald-400" />
          Key Metrics
        </h3>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-slate-800/50 rounded p-2 text-center hover:bg-slate-800 transition-colors cursor-pointer">
            <div className="text-lg font-bold text-red-400 transition-all duration-300">
              {animatedMetrics.totalIncidents}
            </div>
            <div className="text-xs text-slate-400">Total</div>
          </div>
          <div className="bg-slate-800/50 rounded p-2 text-center hover:bg-slate-800 transition-colors cursor-pointer">
            <div className="text-lg font-bold text-orange-400 transition-all duration-300">
              {animatedMetrics.avgResponseTime}m
            </div>
            <div className="text-xs text-slate-400">Response</div>
          </div>
          <div className="bg-slate-800/50 rounded p-2 text-center hover:bg-slate-800 transition-colors cursor-pointer">
            <div className="text-lg font-bold text-cyan-400 transition-all duration-300">
              {animatedMetrics.trackedPersons}
            </div>
            <div className="text-xs text-slate-400">Tracked</div>
          </div>
          <div className="bg-slate-800/50 rounded p-2 text-center hover:bg-slate-800 transition-colors cursor-pointer">
            <div className="text-lg font-bold text-emerald-400 transition-all duration-300">
              {animatedMetrics.uniqueVisitors}
            </div>
            <div className="text-xs text-slate-400">Visitors</div>
          </div>
        </div>
      </div>
      
      {/* Reports - Export Real Data */}
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-slate-100 flex items-center gap-2 text-sm">
            <FileText className="w-4 h-4 text-indigo-400" />
            Export Data
          </h3>
          {isExporting && (
            <span className="text-xs text-yellow-400 animate-pulse">Exporting...</span>
          )}
          {!isAuthenticated && (
            <span className="text-xs text-orange-400 flex items-center gap-1">
              <Lock className="w-3 h-3" />
              Login Required
            </span>
          )}
        </div>
        <div className="space-y-1">
          <button 
            onClick={() => handleExport('behaviors')}
            disabled={isExporting}
            className={`w-full text-left p-2 rounded text-xs transition-colors flex items-center justify-between ${
              selectedReport === 'behaviors' 
                ? 'bg-indigo-600 text-white' 
                : 'hover:bg-slate-800 text-slate-300'
            } ${isExporting ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <span>🚨 Behavior Events (CSV)</span>
            {isAuthenticated ? (
              <Download className="w-3 h-3" />
            ) : (
              <Lock className="w-3 h-3 text-orange-400" />
            )}
          </button>
          <button 
            onClick={() => handleExport('detections')}
            disabled={isExporting}
            className={`w-full text-left p-2 rounded text-xs transition-colors flex items-center justify-between ${
              selectedReport === 'detections' 
                ? 'bg-indigo-600 text-white' 
                : 'hover:bg-slate-800 text-slate-300'
            } ${isExporting ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <span>👤 Person Detections (CSV)</span>
            {isAuthenticated ? (
              <Download className="w-3 h-3" />
            ) : (
              <Lock className="w-3 h-3 text-orange-400" />
            )}
          </button>
          <button 
            onClick={() => handleExport('journeys')}
            disabled={isExporting}
            className={`w-full text-left p-2 rounded text-xs transition-colors flex items-center justify-between ${
              selectedReport === 'journeys' 
                ? 'bg-indigo-600 text-white' 
                : 'hover:bg-slate-800 text-slate-300'
            } ${isExporting ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <span>🚶 Person Journeys (CSV)</span>
            {isAuthenticated ? (
              <Download className="w-3 h-3" />
            ) : (
              <Lock className="w-3 h-3 text-orange-400" />
            )}
          </button>
        </div>
        <div className="mt-2 text-xs text-slate-500 border-t border-slate-700 pt-2">
          {isAuthenticated ? (
            'Exports last 24 hours of data'
          ) : (
            <span className="text-orange-400">Login to export data</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;