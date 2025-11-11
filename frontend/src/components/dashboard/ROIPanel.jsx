import React, { useState, useEffect } from 'react';
import { Target, TrendingUp, TrendingDown, Minus, RefreshCw, Activity, Users, Clock } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

const ROIPanel = ({ refreshInterval = 5000 }) => {
  const [roiData, setRoiData] = useState({});
  const [prevRoiData, setPrevRoiData] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [error, setError] = useState(null);

  // Fetch ROI data from API
  const fetchROIData = async (showRefreshIndicator = false) => {
    try {
      if (showRefreshIndicator) setIsRefreshing(true);
      
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/occupancy`);
      const data = await response.json();
      
      console.log('🎯 ROI Panel fetched data:', data);
      
      if (data.success) {
        setPrevRoiData(roiData);
        setRoiData(data.occupancy || {});
        setLastUpdate(new Date());
        setError(null);
      }
      
      if (isLoading) setIsLoading(false);
    } catch (err) {
      console.error('Failed to fetch ROI data:', err);
      setError('Failed to connect to server');
    } finally {
      setIsRefreshing(false);
    }
  };

  // Initial fetch and setup polling
  useEffect(() => {
    fetchROIData();

    const intervalId = setInterval(() => {
      fetchROIData(false);
    }, refreshInterval);

    return () => clearInterval(intervalId);
  }, [refreshInterval]);

  const handleManualRefresh = () => {
    fetchROIData(true);
  };

  const getOccupancyPercentage = (current, capacity) => {
    if (!capacity) return 0;
    return Math.round((current / capacity) * 100);
  };

  const getOccupancyColor = (percentage) => {
    if (percentage >= 90) return 'text-red-400 bg-red-500/10 border-red-500/50';
    if (percentage >= 70) return 'text-orange-400 bg-orange-500/10 border-orange-500/50';
    if (percentage >= 50) return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/50';
    return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/50';
  };

  const getOccupancyStatus = (percentage) => {
    if (percentage >= 90) return 'Critical';
    if (percentage >= 70) return 'High';
    if (percentage >= 50) return 'Moderate';
    return 'Normal';
  };

  const getTrendIcon = (roi) => {
    const currentData = roiData[roi];
    const previousData = prevRoiData[roi];
    
    const current = currentData?.current || 0;
    const previous = previousData?.current || 0;
    
    if (current > previous) return <TrendingUp className="w-3 h-3 text-emerald-400" />;
    if (current < previous) return <TrendingDown className="w-3 h-3 text-red-400" />;
    return <Minus className="w-3 h-3 text-slate-500" />;
  };

  const getLastUpdateText = () => {
    if (!lastUpdate) return 'Never';
    const seconds = Math.floor((Date.now() - lastUpdate) / 1000);
    if (seconds < 10) return 'Just now';
    if (seconds < 60) return `${seconds}s ago`;
    return `${Math.floor(seconds / 60)}m ago`;
  };

  const getTotalOccupancy = () => {
    let totalCurrent = 0;
    let totalCapacity = 0;
    
    Object.values(roiData).forEach(data => {
      totalCurrent += data.current || 0;
      totalCapacity += data.capacity || 0;
    });
    
    return { totalCurrent, totalCapacity };
  };

  const { totalCurrent, totalCapacity } = getTotalOccupancy();
  const overallPercentage = getOccupancyPercentage(totalCurrent, totalCapacity);

  console.log('🎯 ROI Panel state:', { roiData, totalCurrent, totalCapacity, overallPercentage });

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg h-full flex flex-col">
      {/* Header */}
      <div className="bg-slate-800 border-b border-slate-600 px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-slate-100 flex items-center gap-2 text-sm">
            <Target className="w-4 h-4 text-cyan-400" />
            ROI Occupancy
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handleManualRefresh}
              disabled={isRefreshing}
              className="p-1 hover:bg-slate-700 rounded transition-colors disabled:opacity-50"
              title="Refresh"
            >
              <RefreshCw className={`w-4 h-4 text-slate-400 ${isRefreshing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Last Update & Overall Stats */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-500">Updated: {getLastUpdateText()}</span>
          {error && (
            <span className="text-red-400">{error}</span>
          )}
        </div>

        {/* Overall Occupancy Bar */}
        {totalCapacity > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-400">Overall Occupancy</span>
              <span className={`text-xs font-bold ${getOccupancyColor(overallPercentage).split(' ')[0]}`}>
                {totalCurrent}/{totalCapacity} ({overallPercentage}%)
              </span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all duration-500 ${
                  overallPercentage >= 90 ? 'bg-red-500' :
                  overallPercentage >= 70 ? 'bg-orange-500' :
                  overallPercentage >= 50 ? 'bg-yellow-500' :
                  'bg-emerald-500'
                }`}
                style={{ width: `${Math.min(overallPercentage, 100)}%` }}
              />
            </div>
          </div>
        )}
      </div>
      
      {/* ROI List */}
      <div className="flex-1 overflow-y-auto p-3">
        {isLoading ? (
          <div className="text-center py-8">
            <Activity className="w-12 h-12 mx-auto mb-3 text-cyan-400 animate-spin" />
            <p className="text-slate-400 text-sm">Loading occupancy data...</p>
          </div>
        ) : Object.keys(roiData).length === 0 ? (
          <div className="text-center py-8">
            <Target className="w-12 h-12 mx-auto mb-3 text-slate-600" />
            <p className="font-semibold text-slate-400 mb-1">No ROI Data</p>
            <p className="text-slate-500 text-sm">Waiting for detections in geo-fences...</p>
          </div>
        ) : (
          <div className="space-y-3">
            {Object.entries(roiData).map(([roi, data]) => {
              const current = data.current || 0;
              const capacity = data.capacity || 50;
              const trend = data.trend || 'stable';
              
              const percentage = getOccupancyPercentage(current, capacity);
              const colorClass = getOccupancyColor(percentage);
              const status = getOccupancyStatus(percentage);
              
              console.log(`🎯 Rendering ROI: ${roi}`, { current, capacity, percentage });
              
              return (
                <div 
                  key={roi} 
                  className="bg-slate-800/50 border border-slate-600 rounded-lg p-3 hover:bg-slate-800/80 transition-all"
                >
                  {/* ROI Name & Trend */}
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <h4 className="font-semibold text-slate-100 text-sm">{roi}</h4>
                      {getTrendIcon(roi)}
                    </div>
                    <span className={`px-2 py-1 rounded text-xs font-bold border ${colorClass}`}>
                      {status}
                    </span>
                  </div>

                  {/* Occupancy Bar */}
                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-slate-400 flex items-center gap-1">
                        <Users className="w-3 h-3" />
                        Occupancy
                      </span>
                      <span className="text-xs font-bold text-slate-300">
                        {current}/{capacity}
                      </span>
                    </div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-500 ${
                          percentage >= 90 ? 'bg-red-500' :
                          percentage >= 70 ? 'bg-orange-500' :
                          percentage >= 50 ? 'bg-yellow-500' :
                          'bg-emerald-500'
                        }`}
                        style={{ width: `${Math.min(percentage, 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Stats Row */}
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1 text-slate-400">
                      <Clock className="w-3 h-3" />
                      <span className="capitalize">Trend: {trend}</span>
                    </div>
                    <span className={`font-bold ${colorClass.split(' ')[0]}`}>
                      {percentage}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default ROIPanel;