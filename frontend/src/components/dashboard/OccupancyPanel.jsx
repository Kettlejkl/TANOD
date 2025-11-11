import React from 'react';
import { Users } from 'lucide-react';

const OccupancyPanel = ({ occupancy }) => {
  console.log('📊 OccupancyPanel received:', occupancy);
  console.log('📊 Occupancy type:', typeof occupancy);
  console.log('📊 Is array?:', Array.isArray(occupancy));
  
  // Safety check - ensure occupancy is an object
  const safeOccupancy = occupancy && typeof occupancy === 'object' && !Array.isArray(occupancy) 
    ? occupancy 
    : {};
  
  const zones = Object.entries(safeOccupancy);
  
  console.log('📊 Zones array:', zones);
  
  const totalCurrent = zones.reduce((sum, [, data]) => {
    const current = data?.current || 0;
    console.log('📊 Adding current:', current);
    return sum + current;
  }, 0);
  
  const totalCapacity = zones.reduce((sum, [, data]) => {
    const capacity = data?.capacity || 0;
    console.log('📊 Adding capacity:', capacity);
    return sum + capacity;
  }, 0);
  
  console.log('📊 Final totalCurrent:', totalCurrent);
  console.log('📊 Final totalCapacity:', totalCapacity);

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg h-full flex flex-col">
      <div className="bg-slate-800 border-b border-slate-600 px-4 py-3 flex items-center justify-between">
        <h3 className="font-semibold text-slate-100 flex items-center gap-2 text-sm">
          <Users className="w-4 h-4 text-blue-400" />
          Live Occupancy
        </h3>
        <div className="text-right">
          <div className="text-xl font-bold text-cyan-400">{totalCurrent}</div>
          <div className="text-xs text-slate-400">of {totalCapacity}</div>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-3">
        {zones.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-400 text-sm">No zones with active occupancy</p>
          </div>
        ) : (
          <div className="space-y-3">
            {zones.map(([zone, data]) => {
              console.log(`📊 Rendering zone: ${zone}`, data);
              
              const current = data?.current || 0;
              const capacity = data?.capacity || 50;
              const trend = data?.trend || 'stable';
              
              const percentage = capacity > 0 ? Math.round((current / capacity) * 100) : 0;
              
              const getStatusColor = () => {
                if (percentage >= 90) return { bg: 'from-red-500 to-red-600', text: 'text-red-400' };
                if (percentage >= 70) return { bg: 'from-orange-500 to-orange-600', text: 'text-orange-400' };
                if (percentage >= 50) return { bg: 'from-yellow-500 to-yellow-600', text: 'text-yellow-400' };
                return { bg: 'from-emerald-500 to-emerald-600', text: 'text-emerald-400' };
              };
              
              const colors = getStatusColor();

              return (
                <div key={zone} className="bg-slate-800/50 border border-slate-600 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-slate-100 text-sm truncate">{zone}</span>
                    <span className={`font-bold ${colors.text}`}>{current}/{capacity}</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2 mb-2 overflow-hidden">
                    <div 
                      className={`h-2 rounded-full transition-all duration-1000 bg-gradient-to-r ${colors.bg}`} 
                      style={{ width: `${percentage}%` }} 
                    />
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={`font-medium ${colors.text}`}>{percentage}%</span>
                    <span className="text-slate-400 capitalize">{trend}</span>
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

export default OccupancyPanel;