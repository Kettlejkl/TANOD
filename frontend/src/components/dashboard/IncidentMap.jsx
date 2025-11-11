import React from 'react';
import { Map } from 'lucide-react';

const IncidentMap = ({ incidents = [] }) => {
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      default: return 'bg-blue-500';
    }
  };

  const getIncidentPosition = (incident) => {
    try {
      const hash = (incident?.id || '').split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
      const x = 15 + ((hash * 7) % 70);
      const y = 15 + ((hash * 13) % 70);
      return { x, y };
    } catch (e) {
      return { x: 50, y: 50 };
    }
  };

  const activeIncidents = incidents.filter(i => i?.status !== 'resolved');

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 h-full flex flex-col">
      <h3 className="font-semibold text-slate-100 mb-3 flex items-center gap-2 text-sm">
        <Map className="w-4 h-4 text-emerald-400" />
        Incident Map
      </h3>
      
      <div className="bg-slate-950 border border-slate-700 rounded flex-1 relative overflow-hidden">
        {/* Grid background */}
        <div className="absolute inset-0 opacity-20">
          <div className="grid grid-cols-12 grid-rows-8 h-full w-full">
            {Array.from({ length: 96 }).map((_, i) => (
              <div key={i} className="border border-slate-700/30"></div>
            ))}
          </div>
        </div>

        {/* Incident markers */}
        <div className="absolute inset-0">
          {incidents.map((incident, idx) => {
            if (!incident) return null;
            
            const pos = getIncidentPosition(incident);
            const isActive = incident.status !== 'resolved';
            
            return (
              <div
                key={incident.id || idx}
                className="absolute"
                style={{ 
                  left: `${pos.x}%`, 
                  top: `${pos.y}%`,
                  transform: 'translate(-50%, -50%)'
                }}
              >
                <div 
                  className={`w-3 h-3 rounded-full ${getSeverityColor(incident.severity)} 
                    ${isActive ? 'animate-pulse' : 'opacity-30'}
                    shadow-lg`}
                />
              </div>
            );
          })}
        </div>

        {/* Bottom stats bar */}
        <div className="absolute bottom-3 left-3 right-3 bg-slate-900/80 border border-slate-700 rounded px-3 py-2 flex items-center justify-between">
          <p className="text-slate-300 text-xs font-medium">
            {activeIncidents.length} active incident{activeIncidents.length !== 1 ? 's' : ''}
          </p>
          <div className="flex items-center gap-2 text-[10px]">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
              <span className="text-slate-400">Critical</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-orange-500"></div>
              <span className="text-slate-400">High</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
              <span className="text-slate-400">Medium</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IncidentMap;