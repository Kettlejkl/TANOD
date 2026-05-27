import React from 'react';
import { AlertCircle, Flame, Wind, Users, PersonStanding, Siren, Timer, UserX } from 'lucide-react';

const getAnomalyIcon = (type) => {
  switch(type.toLowerCase()) {
    case 'fire': return <Flame className="w-4 h-4" />;
    case 'smoke': return <Wind className="w-4 h-4" />;
    case 'crowd': return <Users className="w-4 h-4" />;
    case 'violence': return <Siren className="w-4 h-4" />;
    case 'fallen': return <UserX className="w-4 h-4" />;
    case 'loitering': return <Timer className="w-4 h-4" />;
    case 'running': return <PersonStanding className="w-4 h-4" />;
    default: return <AlertCircle className="w-4 h-4" />;
  }
};

const getAnomalyDetails = (anomaly) => {
  const details = [];
  const type = anomaly.type?.toLowerCase() || '';
  
  switch(type) {
    case 'loitering':
      if (anomaly.duration) details.push(`${anomaly.duration.toFixed(1)}s stationary`);
      if (anomaly.movement_range) details.push(`Movement: ${anomaly.movement_range.toFixed(0)}px`);
      if (anomaly.avg_velocity) details.push(`Velocity: ${anomaly.avg_velocity.toFixed(1)} px/s`);
      if (anomaly.person_id !== undefined) details.push(`Person ID: ${anomaly.person_id}`);
      break;
      
    case 'running':
      if (anomaly.velocity) details.push(`Speed: ${anomaly.velocity.toFixed(1)} px/s`);
      if (anomaly.max_velocity) details.push(`Peak: ${anomaly.max_velocity.toFixed(1)} px/s`);
      if (anomaly.duration) details.push(`Duration: ${anomaly.duration.toFixed(1)}s`);
      if (anomaly.confirmation_ratio) details.push(`${(anomaly.confirmation_ratio * 100).toFixed(0)}% confirmed`);
      if (anomaly.person_id !== undefined) details.push(`Person ID: ${anomaly.person_id}`);
      break;
      
    case 'violence':
      if (anomaly.violent_person_ids?.length) {
        details.push(`${anomaly.violent_person_ids.length} person(s) involved`);
        details.push(`IDs: ${anomaly.violent_person_ids.join(', ')}`);
      } else if (anomaly.person_id !== undefined) {
        details.push(`Person ID: ${anomaly.person_id}`);
      }
      if (anomaly.max_arm_velocity) details.push(`Arm velocity: ${anomaly.max_arm_velocity.toFixed(0)} px/s`);
      if (anomaly.confirmation_ratio) details.push(`${(anomaly.confirmation_ratio * 100).toFixed(0)}% confirmed`);
      if (anomaly.duration) details.push(`Duration: ${anomaly.duration.toFixed(1)}s`);
      if (anomaly.face_blur_disabled) details.push('⚠️ Face blur disabled');
      if (anomaly.nearby_persons?.length) details.push(`${anomaly.nearby_persons.length} nearby`);
      break;
      
    case 'fallen':
      if (anomaly.time_horizontal) details.push(`Motionless: ${anomaly.time_horizontal.toFixed(1)}s`);
      if (anomaly.aspect_ratio) details.push(`Aspect ratio: ${anomaly.aspect_ratio.toFixed(2)}`);
      if (anomaly.y_change) details.push(`Fall distance: ${anomaly.y_change.toFixed(0)}px`);
      if (anomaly.avg_velocity !== undefined) details.push(`Velocity: ${anomaly.avg_velocity.toFixed(1)} px/s`);
      if (anomaly.person_id !== undefined) details.push(`Person ID: ${anomaly.person_id}`);
      break;
      
    case 'fire':
    case 'smoke':
      if (anomaly.area_ratio) details.push(`Coverage: ${anomaly.area_ratio.toFixed(1)}% of frame`);
      if (anomaly.duration) details.push(`Detected for: ${anomaly.duration.toFixed(1)}s`);
      if (anomaly.detection_frames) details.push(`${anomaly.detection_frames} frames`);
      if (anomaly.detection_method) {
        const method = anomaly.detection_method === 'yolo_object_detection' ? 'YOLO Detection' : 'Color Analysis';
        details.push(`Method: ${method}`);
      }
      if (anomaly.zone_id) details.push(`Zone: ${anomaly.zone_id}`);
      break;
      
    case 'crowd':
      if (anomaly.person_count) details.push(`${anomaly.person_count} people detected`);
      if (anomaly.avg_density) details.push(`Avg density: ${anomaly.avg_density.toFixed(1)}`);
      if (anomaly.zone_id) details.push(`Zone: ${anomaly.zone_id}`);
      break;
      
    default:
      if (anomaly.description) details.push(anomaly.description);
  }
  
  return details;
};

const AnomalyLogsPanel = ({ anomalies = [] }) => {
  const anomalyList = Array.isArray(anomalies) ? anomalies : [];
  
  // 🔍 DEBUG: Log the anomalies data
  React.useEffect(() => {
    if (anomalyList.length > 0) {
      console.log('📊 Anomalies Data:', anomalyList);
      console.log('📊 First Anomaly Full Data:', anomalyList[0]);
    }
  }, [anomalyList]);
  
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg h-full flex flex-col">
      <div className="bg-slate-800 border-b border-slate-600 px-4 py-3">
        <h3 className="font-semibold text-slate-100 flex items-center gap-2 text-sm">
          <AlertCircle className="w-4 h-4 text-orange-400" />
          Anomaly Logs
          <span className="ml-auto text-slate-400 font-normal">
            {anomalyList.length} alert{anomalyList.length !== 1 ? 's' : ''}
          </span>
        </h3>
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        {anomalyList.length === 0 ? (
          <div className="text-center text-slate-500 text-sm py-8">
            No anomalies detected
          </div>
        ) : (
          <div className="space-y-2">
            {anomalyList.map((anomaly, index) => {
              const details = getAnomalyDetails(anomaly);
              const anomalyType = anomaly.type || 'unknown';
              const severity = anomaly.severity || 'medium';
              
              return (
                <div 
                  key={anomaly.id || `anomaly-${index}`} 
                  className="bg-slate-800/50 border border-slate-600 rounded p-3 hover:bg-slate-800/70 transition-colors"
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2 flex-1">
                      <span className={`${
                        severity === 'critical' ? 'text-red-400' :
                        severity === 'high' ? 'text-orange-400' :
                        'text-yellow-400'
                      }`}>
                        {getAnomalyIcon(anomalyType)}
                      </span>
                      <span className="text-slate-200 text-sm font-medium capitalize">
                        {anomalyType.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium shrink-0 ${
                      severity === 'critical' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                      severity === 'high' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                      'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                    }`}>
                      {severity}
                    </span>
                  </div>
                  
                  {details.length > 0 && (
                    <div className="space-y-1 mb-2">
                      {details.map((detail, idx) => (
                        <div key={idx} className="text-xs text-slate-300 flex items-start gap-1.5">
                          <span className="text-slate-500 mt-0.5">•</span>
                          <span className="flex-1">{detail}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  <div className="flex justify-between items-center text-xs text-slate-400 pt-2 border-t border-slate-700/50">
                    <span className="truncate">
                      {anomaly.roi || anomaly.zone_id || 'Unknown ROI'}
                    </span>
                    <span className="shrink-0 ml-2">
                      {anomaly.timestamp 
                        ? new Date(anomaly.timestamp).toLocaleTimeString()
                        : 'N/A'
                      }
                    </span>
                    {anomaly.confidence !== undefined && (
                      <span className="text-slate-500 shrink-0 ml-2">
                        {(anomaly.confidence * 100).toFixed(0)}%
                      </span>
                    )}
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

export default AnomalyLogsPanel;