import React, { useState, useEffect } from 'react';
import { Activity, Clock, MapPin, TrendingUp, Flame } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

const BehavioralHeatmapTimeline = ({ refreshInterval = 10000 }) => {
  const [heatmapData, setHeatmapData] = useState([]);
  const [selectedCell, setSelectedCell] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [maxIntensity, setMaxIntensity] = useState(1);

  // Define time slots (24 hours split into 2-hour blocks)
  const timeSlots = [
    '00-02', '02-04', '04-06', '06-08', '08-10', '10-12',
    '12-14', '14-16', '16-18', '18-20', '20-22', '22-24'
  ];

  // Define zones
  const zones = ['Eastbound', 'Westbound'];

  useEffect(() => {
    const fetchHeatmapData = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/analytics/dashboard/anomalies?limit=500`);
        const data = await response.json();
        
        if (data.success) {
          // Create heatmap matrix
          const matrix = {};
          let max = 0;
          
          // Initialize matrix
          zones.forEach(zone => {
            matrix[zone] = {};
            timeSlots.forEach(slot => {
              matrix[zone][slot] = {
                count: 0,
                incidents: []
              };
            });
          });
          
          // Populate matrix with incident data
          data.anomalies?.forEach(anomaly => {
            const timestamp = new Date(anomaly.timestamp);
            const hour = timestamp.getHours();
            const slotIndex = Math.floor(hour / 2);
            const slot = timeSlots[slotIndex];
            
            let zone = 'Unknown';
            if (anomaly.roi?.toLowerCase().includes('eastbound')) {
              zone = 'Eastbound';
            } else if (anomaly.roi?.toLowerCase().includes('westbound')) {
              zone = 'Westbound';
            }
            
            if (matrix[zone] && matrix[zone][slot]) {
              matrix[zone][slot].count++;
              matrix[zone][slot].incidents.push(anomaly);
              
              if (matrix[zone][slot].count > max) {
                max = matrix[zone][slot].count;
              }
            }
          });
          
          setMaxIntensity(max || 1);
          setHeatmapData(matrix);
        }
        
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to fetch heatmap data:', err);
        setIsLoading(false);
      }
    };

    fetchHeatmapData();
    const interval = setInterval(fetchHeatmapData, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const getHeatColor = (count, max) => {
    if (count === 0) return 'bg-slate-800';
    const intensity = (count / max) * 100;
    
    if (intensity >= 80) return 'bg-red-600';
    if (intensity >= 60) return 'bg-orange-500';
    if (intensity >= 40) return 'bg-yellow-500';
    if (intensity >= 20) return 'bg-emerald-500';
    return 'bg-blue-500';
  };

  const getHeatIntensity = (count, max) => {
    if (count === 0) return 'opacity-20';
    const intensity = (count / max) * 100;
    
    if (intensity >= 80) return 'opacity-100';
    if (intensity >= 60) return 'opacity-80';
    if (intensity >= 40) return 'opacity-60';
    if (intensity >= 20) return 'opacity-40';
    return 'opacity-30';
  };

  const getCurrentTimeSlot = () => {
    const hour = new Date().getHours();
    const slotIndex = Math.floor(hour / 2);
    return timeSlots[slotIndex];
  };

  const handleCellClick = (zone, slot, cellData) => {
    if (cellData.count > 0) {
      setSelectedCell({ zone, slot, ...cellData });
    }
  };

  const renderHeatmapCell = (zone, slot) => {
    const cellData = heatmapData[zone]?.[slot];
    if (!cellData) return null;
    
    const isCurrentSlot = slot === getCurrentTimeSlot();
    const isSelected = selectedCell?.zone === zone && selectedCell?.slot === slot;
    
    return (
      <div
        key={`${zone}-${slot}`}
        onClick={() => handleCellClick(zone, slot, cellData)}
        className={`relative h-16 rounded cursor-pointer transition-all border-2 ${
          isSelected ? 'border-cyan-400 ring-2 ring-cyan-400/50' : 'border-transparent'
        } ${isCurrentSlot ? 'ring-2 ring-blue-400' : ''}`}
        title={`${zone} ${slot}:00 - ${cellData.count} incidents`}
      >
        {/* Heat Background */}
        <div className={`absolute inset-0 rounded ${getHeatColor(cellData.count, maxIntensity)} ${getHeatIntensity(cellData.count, maxIntensity)}`} />
        
        {/* Count Badge */}
        {cellData.count > 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-white font-bold text-sm drop-shadow-lg">
              {cellData.count}
            </span>
          </div>
        )}
        
        {/* Current Time Indicator */}
        {isCurrentSlot && (
          <div className="absolute -top-1 -right-1">
            <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse" />
          </div>
        )}
        
        {/* Hot Spot Indicator */}
        {cellData.count >= maxIntensity * 0.8 && cellData.count > 0 && (
          <div className="absolute -top-1 -left-1">
            <Flame className="w-4 h-4 text-orange-400 animate-pulse" />
          </div>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-lg h-full flex items-center justify-center">
        <Activity className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg h-full flex flex-col">
      {/* Header */}
      <div className="bg-slate-800 border-b border-slate-700 px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-slate-100 flex items-center gap-2 text-sm">
            <Activity className="w-4 h-4 text-cyan-400" />
            Behavioral Heatmap Timeline
          </h3>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Peak: {maxIntensity} incidents</span>
          </div>
        </div>
        
        {/* Legend */}
        <div className="flex items-center gap-3 text-[10px]">
          <span className="text-slate-500">Intensity:</span>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-blue-500 opacity-30 rounded" />
            <span className="text-slate-500">Low</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-emerald-500 opacity-40 rounded" />
            <span className="text-slate-500">Mild</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-yellow-500 opacity-60 rounded" />
            <span className="text-slate-500">Medium</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-orange-500 opacity-80 rounded" />
            <span className="text-slate-500">High</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-red-600 opacity-100 rounded" />
            <span className="text-slate-500">Critical</span>
          </div>
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="flex-1 overflow-auto p-4">
        <div className="min-w-max">
          {/* Time Labels */}
          <div className="grid grid-cols-[100px_repeat(12,80px)] gap-2 mb-2">
            <div className="text-xs font-semibold text-slate-400">Zone / Time</div>
            {timeSlots.map(slot => (
              <div key={slot} className="text-xs text-slate-500 text-center">
                {slot}
              </div>
            ))}
          </div>
          
          {/* Heatmap Rows */}
          {zones.map(zone => (
            <div key={zone} className="grid grid-cols-[100px_repeat(12,80px)] gap-2 mb-2">
              <div className="flex items-center">
                <span className="text-xs font-semibold text-slate-300">{zone}</span>
              </div>
              {timeSlots.map(slot => renderHeatmapCell(zone, slot))}
            </div>
          ))}
        </div>
      </div>

      {/* Selected Cell Details */}
      {selectedCell && (
        <div className="border-t border-slate-700 bg-slate-800/50 p-4">
          <div className="flex items-start justify-between mb-3">
            <div>
              <h4 className="font-bold text-slate-100 flex items-center gap-2">
                <MapPin className="w-4 h-4 text-cyan-400" />
                {selectedCell.zone} - {selectedCell.slot}:00
              </h4>
              <p className="text-xs text-slate-500 mt-1">
                {selectedCell.count} incident{selectedCell.count !== 1 ? 's' : ''} in this time block
              </p>
            </div>
            <button
              onClick={() => setSelectedCell(null)}
              className="text-slate-500 hover:text-slate-300 text-xs"
            >
              ✕ Close
            </button>
          </div>

          {/* Incidents List */}
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {selectedCell.incidents.slice(0, 5).map((incident, idx) => (
              <div key={idx} className="bg-slate-900 rounded p-2 text-xs">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-slate-300 font-semibold">{incident.type}</span>
                  <span className="text-slate-600">
                    {new Date(incident.timestamp).toLocaleTimeString('en-US', { 
                      hour: '2-digit', 
                      minute: '2-digit' 
                    })}
                  </span>
                </div>
                {incident.uid && (
                  <div className="text-slate-500">UID: {incident.uid.substring(0, 12)}...</div>
                )}
              </div>
            ))}
            {selectedCell.count > 5 && (
              <p className="text-center text-slate-600 text-[10px]">
                +{selectedCell.count - 5} more incidents
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default BehavioralHeatmapTimeline;