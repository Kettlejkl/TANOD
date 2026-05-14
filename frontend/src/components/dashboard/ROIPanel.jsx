import React, { useState, useEffect, useRef } from 'react';
import { Target, Zap, Clock, AlertTriangle, User, Shield, TrendingUp, Eye, Activity } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

const RiskConstellationViz = ({ refreshInterval = 5000 }) => {
  const [constellation, setConstellation] = useState({ persons: [], links: [], stats: {} });
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [viewMode, setViewMode] = useState('risk'); // risk, behavior, temporal
  const [isLoading, setIsLoading] = useState(true);
  const [rotation, setRotation] = useState(0);
  const [debugInfo, setDebugInfo] = useState('');
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    const fetchConstellationData = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/analytics/dashboard/anomalies?limit=200`);
        const data = await response.json();
        
        console.log('🔍 Raw API Response:', data);
        
        if (data.success && data.anomalies) {
          console.log('📊 Total Anomalies:', data.anomalies.length);
          console.log('📊 Sample Anomaly:', data.anomalies[0]);
          
          // Build person profiles with FLEXIBLE ID handling
          const personMap = {};
          const behaviorTypes = new Set();
          const zones = new Set();
          
          data.anomalies.forEach((anomaly, idx) => {
            // FLEXIBLE PERSON ID - try multiple fields
            let personId = anomaly.persistent_id || 
                          anomaly.person_id || 
                          anomaly.track_id ||
                          anomaly.id;
            
            // If still no ID, create synthetic one based on zone + behavior + time bucket
            if (!personId) {
              const zone = anomaly.roi || anomaly.fence_name || 'Unknown';
              const behavior = anomaly.type || 'Unknown';
              const timeBucket = Math.floor(new Date(anomaly.timestamp).getTime() / (1000 * 60 * 5)); // 5min buckets
              personId = `synthetic_${zone}_${behavior}_${timeBucket}`.replace(/\s+/g, '_');
            }
            
            const zone = anomaly.roi || anomaly.fence_name || anomaly.zone_id || 'Unknown';
            const behavior = anomaly.type || 'Unknown';
            const timestamp = new Date(anomaly.timestamp).getTime();
            
            behaviorTypes.add(behavior);
            zones.add(zone);
            
            if (!personMap[personId]) {
              personMap[personId] = {
                id: personId,
                isSynthetic: !anomaly.persistent_id && !anomaly.person_id,
                events: [],
                behaviors: {},
                zones: new Set(),
                riskScore: 0,
                firstSeen: timestamp,
                lastSeen: timestamp,
                velocity: 0
              };
            }
            
            const person = personMap[personId];
            person.events.push({ behavior, zone, timestamp, anomaly });
            person.behaviors[behavior] = (person.behaviors[behavior] || 0) + 1;
            person.zones.add(zone);
            person.lastSeen = Math.max(person.lastSeen, timestamp);
            person.firstSeen = Math.min(person.firstSeen, timestamp);
          });
          
          console.log('👥 Total Persons Mapped:', Object.keys(personMap).length);
          console.log('👥 Sample Person:', Object.values(personMap)[0]);
          
          // Calculate risk scores and positions
          const persons = Object.values(personMap).map((person, idx) => {
            // Risk calculation
            let risk = 0;
            risk += person.events.length * 10; // Base from event count
            risk += Object.keys(person.behaviors).length * 15; // Behavior diversity
            risk += person.zones.size * 8; // Multi-zone presence
            
            // Severity multipliers
            if (person.behaviors.violence) risk += 40;
            if (person.behaviors.fallen) risk += 30;
            if (person.behaviors.running) risk += 20;
            if (person.behaviors.loitering) risk += 15;
            if (person.behaviors.fire || person.behaviors.smoke) risk += 35;
            
            // Recency bonus
            const hoursSince = (Date.now() - person.lastSeen) / (1000 * 60 * 60);
            if (hoursSince < 1) risk += 25;
            else if (hoursSince < 6) risk += 15;
            
            person.riskScore = Math.min(risk, 100);
            person.riskLevel = risk >= 70 ? 'critical' : risk >= 40 ? 'high' : 'medium';
            
            // Calculate activity velocity (events per hour active)
            const activeHours = Math.max((person.lastSeen - person.firstSeen) / (1000 * 60 * 60), 0.1);
            person.velocity = person.events.length / activeHours;
            
            // Position in constellation (circular layout based on risk)
            const totalPersons = Object.keys(personMap).length;
            const angle = (idx / totalPersons) * Math.PI * 2;
            const radius = 60 + (person.riskScore / 100) * 80; // Higher risk = outer orbit
            
            return {
              ...person,
              x: 200 + Math.cos(angle) * radius,
              y: 180 + Math.sin(angle) * radius,
              angle,
              radius,
              zones: Array.from(person.zones)
            };
          });
          
          console.log('✅ Persons After Processing:', persons.length);
          
          // Build connections between persons with shared behaviors/zones
          const links = [];
          for (let i = 0; i < persons.length; i++) {
            for (let j = i + 1; j < persons.length; j++) {
              const p1 = persons[i];
              const p2 = persons[j];
              
              // Find common behaviors
              const commonBehaviors = Object.keys(p1.behaviors).filter(b => p2.behaviors[b]);
              
              // Find common zones
              const commonZones = p1.zones.filter(z => p2.zones.includes(z));
              
              if (commonBehaviors.length > 0 || commonZones.length > 0) {
                links.push({
                  from: p1.id,
                  to: p2.id,
                  strength: commonBehaviors.length * 2 + commonZones.length,
                  commonBehaviors,
                  commonZones,
                  type: commonBehaviors.length > 0 ? 'behavior' : 'zone'
                });
              }
            }
          }
          
          console.log('🔗 Links Created:', links.length);
          
          const stats = {
            totalPersons: persons.length,
            criticalRisk: persons.filter(p => p.riskLevel === 'critical').length,
            avgRisk: persons.length > 0 ? persons.reduce((sum, p) => sum + p.riskScore, 0) / persons.length : 0,
            totalConnections: links.length,
            syntheticIds: persons.filter(p => p.isSynthetic).length
          };
          
          setConstellation({
            persons: persons.sort((a, b) => b.riskScore - a.riskScore),
            links,
            stats
          });
          
          setDebugInfo(`Loaded: ${persons.length} entities, ${links.length} connections`);
        } else {
          console.error('❌ API returned no anomalies');
          setDebugInfo('No anomalies returned from API');
        }
        
        setIsLoading(false);
      } catch (err) {
        console.error('❌ Failed to fetch constellation data:', err);
        setDebugInfo(`Error: ${err.message}`);
        setIsLoading(false);
      }
    };

    fetchConstellationData();
    const interval = setInterval(fetchConstellationData, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  // Rotation animation
  useEffect(() => {
    const animate = () => {
      setRotation(prev => (prev + 0.3) % 360);
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Draw constellation
  useEffect(() => {
    if (!canvasRef.current || constellation.persons.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Draw connection links
    constellation.links.forEach(link => {
      const p1 = constellation.persons.find(p => p.id === link.from);
      const p2 = constellation.persons.find(p => p.id === link.to);
      
      if (!p1 || !p2) return;
      
      const isRelevant = !selectedPerson || 
                        selectedPerson.id === link.from || 
                        selectedPerson.id === link.to;
      
      if (!isRelevant && selectedPerson) return;
      
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      
      const alpha = isRelevant ? 0.4 : 0.1;
      const color = link.type === 'behavior' ? `rgba(139, 92, 246, ${alpha})` : `rgba(34, 211, 238, ${alpha})`;
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 1 + (link.strength * 0.3);
      ctx.stroke();
    });
    
    // Draw orbit rings
    [60, 100, 140].forEach((r, idx) => {
      ctx.beginPath();
      ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(100, 116, 139, ${0.15 - idx * 0.04})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
    });
    
    // Draw center indicator
    ctx.beginPath();
    ctx.arc(centerX, centerY, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(34, 211, 238, 0.3)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.6)';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw persons as nodes
    constellation.persons.forEach(person => {
      const isSelected = selectedPerson?.id === person.id;
      const isHovered = hoveredNode?.id === person.id;
      const isRelevant = !selectedPerson || isSelected || 
                        constellation.links.some(l => 
                          (l.from === person.id && l.to === selectedPerson.id) ||
                          (l.to === person.id && l.from === selectedPerson.id)
                        );
      
      const opacity = isRelevant ? 1 : 0.2;
      
      // Node size based on view mode
      let nodeSize = 8;
      if (viewMode === 'risk') {
        nodeSize = 6 + (person.riskScore / 100) * 10;
      } else if (viewMode === 'behavior') {
        nodeSize = 6 + Object.keys(person.behaviors).length * 2;
      } else {
        nodeSize = 6 + Math.min(person.velocity * 1.5, 10);
      }
      
      if (isSelected || isHovered) nodeSize *= 1.2;
      
      // Color based on risk level
      const colors = {
        critical: ['rgba(239, 68, 68, ', 'rgba(220, 38, 38, '],
        high: ['rgba(249, 115, 22, ', 'rgba(234, 88, 12, '],
        medium: ['rgba(234, 179, 8, ', 'rgba(202, 138, 4, ']
      };
      
      const [primaryColor, secondaryColor] = colors[person.riskLevel] || colors.medium;
      
      // Glow effect
      if (isSelected || isHovered) {
        const gradient = ctx.createRadialGradient(
          person.x, person.y, 0,
          person.x, person.y, nodeSize * 2.5
        );
        gradient.addColorStop(0, primaryColor + '0.6)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(person.x, person.y, nodeSize * 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      
      // Pulse ring for critical
      if (person.riskLevel === 'critical') {
        const pulseSize = nodeSize + Math.sin(rotation / 20) * 2;
        ctx.beginPath();
        ctx.arc(person.x, person.y, pulseSize, 0, Math.PI * 2);
        ctx.strokeStyle = primaryColor + '0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      
      // Node body
      ctx.beginPath();
      ctx.arc(person.x, person.y, nodeSize, 0, Math.PI * 2);
      ctx.fillStyle = primaryColor + opacity + ')';
      ctx.fill();
      ctx.strokeStyle = secondaryColor + opacity + ')';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Inner core
      ctx.beginPath();
      ctx.arc(person.x, person.y, nodeSize * 0.4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 255, 255, ' + (opacity * 0.8) + ')';
      ctx.fill();
      
      // Event count badge
      if (isRelevant && person.events.length > 1) {
        ctx.font = 'bold 8px monospace';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        const badgeWidth = 16;
        ctx.fillRect(person.x - badgeWidth/2, person.y + nodeSize + 1, badgeWidth, 10);
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(person.events.length, person.x, person.y + nodeSize + 6);
      }
      
      // Person ID label on hover/select
      if (isSelected || isHovered) {
        ctx.font = 'bold 9px monospace';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        const displayId = person.isSynthetic ? 'Event Group' : `#${person.id}`;
        const textWidth = ctx.measureText(displayId).width;
        ctx.fillRect(person.x - textWidth/2 - 3, person.y - nodeSize - 16, textWidth + 6, 12);
        ctx.fillStyle = primaryColor + '1)';
        ctx.textAlign = 'center';
        ctx.fillText(displayId, person.x, person.y - nodeSize - 10);
      }
    });
    
  }, [constellation, rotation, selectedPerson, hoveredNode, viewMode]);

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const clicked = constellation.persons.find(p => {
      const dx = p.x - x;
      const dy = p.y - y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const nodeSize = 6 + (p.riskScore / 100) * 10;
      return distance < nodeSize + 5;
    });
    
    setSelectedPerson(clicked === selectedPerson ? null : clicked);
  };

  const handleCanvasHover = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const hovered = constellation.persons.find(p => {
      const dx = p.x - x;
      const dy = p.y - y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const nodeSize = 6 + (p.riskScore / 100) * 10;
      return distance < nodeSize + 5;
    });
    
    setHoveredNode(hovered || null);
  };

  const getRiskColor = (level) => {
    return level === 'critical' ? 'text-red-400' : 
           level === 'high' ? 'text-orange-400' : 'text-yellow-400';
  };

  if (isLoading) {
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-lg h-full flex items-center justify-center">
        <Shield className="w-8 h-8 text-cyan-400 animate-pulse" />
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-slate-950 via-slate-900 to-violet-950 border border-violet-500/20 rounded-lg h-full flex flex-col overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="relative bg-gradient-to-r from-slate-800 via-violet-900/20 to-transparent border-b border-violet-500/30 px-4 py-3">
        <div className="absolute inset-0 bg-gradient-to-r from-violet-500/5 to-transparent pointer-events-none"></div>
        <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-red-400 via-orange-500 to-yellow-400"></div>
        
        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-red-400/20 rounded-lg blur-md animate-pulse"></div>
              <div className="relative bg-gradient-to-br from-red-500 via-orange-500 to-yellow-500 p-2 rounded-lg">
                <Target className="w-4 h-4 text-white" />
              </div>
            </div>
            <div>
              <h3 className="font-bold text-transparent bg-clip-text bg-gradient-to-r from-red-400 via-orange-400 to-yellow-400 text-sm tracking-wide">
                RISK CONSTELLATION
              </h3>
              <p className="text-xs text-slate-500 font-mono">Behavior correlation network</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded px-3 py-1 space-y-0.5">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-red-400 animate-pulse"></div>
                <span className="text-xs font-mono text-red-300">{constellation.stats.criticalRisk}</span>
                <span className="text-xs text-slate-500">critical</span>
              </div>
              <div className="flex items-center gap-2">
                <Activity className="w-3 h-3 text-orange-400" />
                <span className="text-xs font-mono text-orange-300">{constellation.stats.totalPersons}</span>
                <span className="text-xs text-slate-500">entities</span>
              </div>
            </div>
            
            <div className="flex gap-1 bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded p-1">
              {[
                { mode: 'risk', icon: Shield },
                { mode: 'behavior', icon: Zap },
                { mode: 'temporal', icon: Clock }
              ].map(({ mode, icon: Icon }) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`p-1.5 rounded transition-all ${
                    viewMode === mode
                      ? 'bg-gradient-to-r from-red-500 to-orange-600 text-white shadow-lg'
                      : 'text-slate-400 hover:text-red-400 hover:bg-slate-700/50'
                  }`}
                  title={mode}
                >
                  <Icon className="w-3.5 h-3.5" />
                </button>
              ))}
            </div>
          </div>
        </div>
        {debugInfo && (
          <div className="mt-1 text-[10px] text-cyan-400 font-mono">{debugInfo}</div>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1 relative">
        <canvas 
          ref={canvasRef}
          className="absolute inset-0 w-full h-full cursor-pointer"
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasHover}
          onMouseLeave={() => setHoveredNode(null)}
        />
        
        {constellation.persons.length === 0 && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center">
              <Target className="w-12 h-12 mx-auto mb-3 text-slate-600" />
              <p className="font-semibold text-slate-400">No Risk Data</p>
              <p className="text-slate-500 text-xs mt-1">Awaiting detections...</p>
              <p className="text-slate-600 text-xs mt-2 font-mono">{debugInfo}</p>
            </div>
          </div>
        )}
        
        {/* Hover tooltip */}
        {hoveredNode && !selectedPerson && (
          <div 
            className="absolute pointer-events-none z-50"
            style={{
              left: Math.min(hoveredNode.x + 20, 350),
              top: hoveredNode.y - 40
            }}
          >
            <div className="bg-slate-800/95 backdrop-blur-sm border border-red-500/50 rounded-lg p-2 shadow-2xl min-w-[140px]">
              <div className="text-xs font-bold text-red-400 mb-1">
                {hoveredNode.isSynthetic ? 'Event Group' : `Person #${hoveredNode.id}`}
              </div>
              <div className="space-y-0.5 text-xs text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-500">Risk:</span>
                  <span className={getRiskColor(hoveredNode.riskLevel)}>{hoveredNode.riskScore}/100</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Events:</span>
                  <span>{hoveredNode.events.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Zones:</span>
                  <span>{hoveredNode.zones.length}</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Legend */}
        <div className="absolute bottom-2 left-2 bg-slate-900/80 backdrop-blur-sm border border-slate-700/50 rounded px-2 py-1.5 text-xs">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-purple-500"></div>
              <span className="text-slate-400 text-[10px]">Behavior Link</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-cyan-500"></div>
              <span className="text-slate-400 text-[10px]">Zone Link</span>
            </div>
          </div>
        </div>
      </div>

      {/* Detail Panel */}
      {selectedPerson && (
        <div className="border-t border-violet-700/50 bg-gradient-to-br from-slate-900/95 to-violet-950/95 backdrop-blur-sm p-3">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center gap-2">
              <User className="w-4 h-4 text-red-400" />
              <span className="font-bold text-white text-sm">
                {selectedPerson.isSynthetic ? 'Event Group' : `Person #${selectedPerson.id}`}
              </span>
              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                selectedPerson.riskLevel === 'critical' ? 'bg-red-500/20 text-red-400 border border-red-500/50' :
                selectedPerson.riskLevel === 'high' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/50' :
                'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50'
              }`}>
                RISK: {selectedPerson.riskScore}/100
              </span>
            </div>
            <button 
              onClick={() => setSelectedPerson(null)}
              className="text-slate-500 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>
          
          <div className="grid grid-cols-3 gap-2 mb-2">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-xs text-slate-500">Events</div>
              <div className="text-lg font-bold text-white">{selectedPerson.events.length}</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-xs text-slate-500">Behaviors</div>
              <div className="text-lg font-bold text-purple-400">{Object.keys(selectedPerson.behaviors).length}</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-xs text-slate-500">Zones</div>
              <div className="text-lg font-bold text-cyan-400">{selectedPerson.zones.length}</div>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <div className="text-slate-500 mb-1">Top Behaviors</div>
              <div className="space-y-1">
                {Object.entries(selectedPerson.behaviors)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 3)
                  .map(([behavior, count]) => (
                    <div key={behavior} className="flex items-center justify-between bg-slate-800/30 rounded px-2 py-1">
                      <span className="text-slate-300 text-[10px]">{behavior}</span>
                      <span className="text-red-400 font-bold">{count}</span>
                    </div>
                  ))}
              </div>
            </div>
            
            <div>
              <div className="text-slate-500 mb-1">Zones Visited</div>
              <div className="flex flex-wrap gap-1">
                {selectedPerson.zones.slice(0, 3).map((zone, idx) => (
                  <span key={idx} className="px-2 py-0.5 bg-cyan-900/30 border border-cyan-500/50 rounded text-[10px] text-cyan-300">
                    {zone}
                  </span>
                ))}
              </div>
              <div className="text-slate-600 text-[10px] mt-1">
                Last: {new Date(selectedPerson.lastSeen).toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RiskConstellationViz;