import React, { useState, useEffect, useRef, useMemo } from 'react';
import { AlertTriangle, Bell, CheckCircle, MapPin, Clock, Filter, ChevronDown, AlertCircle, Activity, RefreshCw, Wifi, WifiOff } from 'lucide-react';

// Import your API service
const API_BASE_URL = 'http://127.0.0.1:5000/api';

const apiService = {
  getIncidents: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/incidents`);
      const data = await response.json();
      return data.success ? data.incidents : [];
    } catch (error) {
      console.error('Error fetching incidents:', error);
      return [];
    }
  },
  
  acknowledgeIncident: async (id) => {
    try {
      const response = await fetch(`${API_BASE_URL}/alerts/${id}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error acknowledging incident:', error);
      return { success: false };
    }
  }
};

const IncidentPanel = ({ refreshInterval = 5000 }) => {
  const [incidents, setIncidents] = useState([]);
  const [highlightedId, setHighlightedId] = useState(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [showFilters, setShowFilters] = useState(false);
  const [expandedIncident, setExpandedIncident] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [isConnected, setIsConnected] = useState(true);
  const [error, setError] = useState(null);
  
  const scrollRef = useRef(null);
  const prevIncidentsRef = useRef([]);
  const intervalRef = useRef(null);

  // Fetch incidents from API
  const fetchIncidents = async (showRefreshIndicator = false) => {
    try {
      if (showRefreshIndicator) setIsRefreshing(true);
      
      const data = await apiService.getIncidents();
      setIncidents(data);
      setLastUpdate(new Date());
      setIsConnected(true);
      setError(null);
      
      if (isLoading) setIsLoading(false);
    } catch (err) {
      console.error('Failed to fetch incidents:', err);
      setIsConnected(false);
      setError('Failed to connect to server');
    } finally {
      setIsRefreshing(false);
    }
  };

  // Initial fetch and setup polling
  useEffect(() => {
    fetchIncidents();

    // Set up polling interval
    intervalRef.current = setInterval(() => {
      fetchIncidents(false);
    }, refreshInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [refreshInterval]);

  // Filtered incidents
  const filteredIncidents = useMemo(() => {
    return incidents.filter(incident => {
      const statusMatch = filterStatus === 'all' || incident.status === filterStatus;
      const severityMatch = filterSeverity === 'all' || incident.severity === filterSeverity;
      const isActive = incident.status !== 'resolved';
      return isActive && statusMatch && severityMatch;
    });
  }, [incidents, filterStatus, filterSeverity]);

  const newIncidents = filteredIncidents.filter(i => i.status === 'reported');

  // Detect new incidents and highlight them
  useEffect(() => {
    const prevIds = prevIncidentsRef.current.map(i => i.id);
    const newIncident = filteredIncidents.find(i => !prevIds.includes(i.id));
    
    if (newIncident && prevIncidentsRef.current.length > 0) {
      setHighlightedId(newIncident.id);
      
      // Play notification sound effect
      try {
        const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBDGH0fPTgjMGHm7A7+OZSA0PVqzn77BdGAk+ltzy0H0pBSh+zPLaizsIGGS57OihUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWW67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8bllHAU2jdXyzn4sBSp7yvLajToIGWS67OmiUhELTKXh8Q==');
        audio.volume = 0.3;
        audio.play().catch(() => {});
      } catch (e) {}
      
      setTimeout(() => setHighlightedId(null), 3000);
      
      if (scrollRef.current) {
        scrollRef.current.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }
    
    prevIncidentsRef.current = filteredIncidents;
  }, [filteredIncidents]);

  const getSeverityColor = (severity) => {
    const colors = {
      critical: 'text-red-400 border-red-500 bg-red-500/10',
      high: 'text-orange-400 border-orange-500 bg-orange-500/10',
      medium: 'text-yellow-400 border-yellow-500 bg-yellow-500/10',
      low: 'text-blue-400 border-blue-500 bg-blue-500/10'
    };
    return colors[severity] || colors.low;
  };

  const getSeverityBadge = (severity) => {
    const badges = {
      critical: 'bg-red-500/20 text-red-400 border-red-500/50',
      high: 'bg-orange-500/20 text-orange-400 border-orange-500/50',
      medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
      low: 'bg-blue-500/20 text-blue-400 border-blue-500/50'
    };
    return badges[severity] || badges.low;
  };

  const getTimeSince = (timestamp) => {
    const seconds = Math.floor((Date.now() - new Date(timestamp)) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  const handleAcknowledge = async (id) => {
    try {
      const result = await apiService.acknowledgeIncident(id);
      if (result.success) {
        // Refresh incidents after acknowledgment
        await fetchIncidents(true);
      } else {
        console.error('Failed to acknowledge incident');
      }
    } catch (error) {
      console.error('Failed to acknowledge incident:', error);
    }
  };

  const handleManualRefresh = () => {
    fetchIncidents(true);
  };

  const toggleExpand = (id) => {
    setExpandedIncident(expandedIncident === id ? null : id);
  };

  const getSeverityCount = (severity) => {
    return filteredIncidents.filter(i => i.severity === severity).length;
  };

  const getLastUpdateText = () => {
    if (!lastUpdate) return 'Never';
    const seconds = Math.floor((Date.now() - lastUpdate) / 1000);
    if (seconds < 10) return 'Just now';
    if (seconds < 60) return `${seconds}s ago`;
    return `${Math.floor(seconds / 60)}m ago`;
  };

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg h-full flex flex-col">
      {/* Header */}
      <div className="bg-slate-800 border-b border-slate-600 px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-slate-100 flex items-center gap-2 text-sm">
            <Bell className={`w-4 h-4 ${newIncidents.length > 0 ? 'text-red-400 animate-pulse' : 'text-slate-400'}`} />
            Incident Reports
            {isConnected ? (
              <Wifi className="w-3 h-3 text-emerald-400" title="Connected" />
            ) : (
              <WifiOff className="w-3 h-3 text-red-400" title="Disconnected" />
            )}
          </h3>
          <div className="flex gap-2 items-center">
            {newIncidents.length > 0 && (
              <div className="bg-red-600 text-white px-2 py-1 rounded text-xs font-medium animate-pulse flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />
                {newIncidents.length} New
              </div>
            )}
            <span className={`px-2 py-1 rounded text-xs font-bold ${
              filteredIncidents.length > 0 
                ? 'bg-red-500/20 text-red-400' 
                : 'bg-emerald-500/20 text-emerald-400'
            }`}>
              {filteredIncidents.length} ACTIVE
            </span>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="p-1 hover:bg-slate-700 rounded transition-colors"
              title="Toggle filters"
            >
              <Filter className={`w-4 h-4 ${showFilters ? 'text-cyan-400' : 'text-slate-400'}`} />
            </button>
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

        {/* Last Update & Error */}
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Updated: {getLastUpdateText()}</span>
          {error && (
            <span className="text-red-400 flex items-center gap-1">
              <AlertCircle className="w-3 h-3" />
              {error}
            </span>
          )}
        </div>

        {/* Severity Summary */}
        <div className="flex gap-2 text-xs mt-2">
          {['critical', 'high', 'medium', 'low'].map(severity => {
            const count = getSeverityCount(severity);
            if (count === 0) return null;
            return (
              <div 
                key={severity}
                className={`px-2 py-1 rounded border ${getSeverityBadge(severity)}`}
              >
                {severity}: {count}
              </div>
            );
          })}
        </div>

        {/* Filters */}
        {showFilters && (
          <div className="mt-3 pt-3 border-t border-slate-700 space-y-2">
            <div className="flex gap-2 items-center">
              <label className="text-xs text-slate-400 w-16">Status:</label>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="flex-1 bg-slate-700 text-slate-200 text-xs rounded px-2 py-1 border border-slate-600 focus:outline-none focus:border-cyan-500"
              >
                <option value="all">All</option>
                <option value="reported">Reported</option>
                <option value="in-progress">In Progress</option>
              </select>
            </div>
            <div className="flex gap-2 items-center">
              <label className="text-xs text-slate-400 w-16">Severity:</label>
              <select
                value={filterSeverity}
                onChange={(e) => setFilterSeverity(e.target.value)}
                className="flex-1 bg-slate-700 text-slate-200 text-xs rounded px-2 py-1 border border-slate-600 focus:outline-none focus:border-cyan-500"
              >
                <option value="all">All</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
          </div>
        )}
      </div>
      
      {/* Incident List */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3">
        {isLoading ? (
          <div className="text-center py-8">
            <Activity className="w-12 h-12 mx-auto mb-3 text-cyan-400 animate-spin" />
            <p className="text-slate-400 text-sm">Loading incidents...</p>
          </div>
        ) : filteredIncidents.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle className="w-12 h-12 mx-auto mb-3 text-emerald-400" />
            <p className="font-semibold text-emerald-400 mb-1">All Clear</p>
            <p className="text-slate-400 text-sm">
              {filterStatus !== 'all' || filterSeverity !== 'all' 
                ? 'No incidents match current filters' 
                : 'No active incidents'}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredIncidents.map((incident) => {
              const isHighlighted = highlightedId === incident.id;
              const isNew = incident.status === 'reported';
              const isExpanded = expandedIncident === incident.id;
              
              return (
                <div 
                  key={incident.id} 
                  className={`bg-slate-800/50 border rounded-lg p-3 transition-all duration-300 ${
                    isHighlighted 
                      ? 'border-cyan-400 shadow-lg shadow-cyan-400/20 scale-[1.02]' 
                      : 'border-slate-600 hover:bg-slate-800/80'
                  } ${isNew ? 'ring-2 ring-red-500/50' : ''}`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className={`w-4 h-4 flex-shrink-0 ${
                          incident.severity === 'critical' ? 'text-red-400' :
                          incident.severity === 'high' ? 'text-orange-400' :
                          incident.severity === 'medium' ? 'text-yellow-400' :
                          'text-blue-400'
                        }`} />
                        <h4 className="font-semibold text-slate-100 text-sm truncate">
                          {incident.category || incident.title}
                        </h4>
                        <span className={`px-2 py-1 rounded text-xs font-bold uppercase flex-shrink-0 ${
                          incident.status === 'reported' ? 'bg-red-500/20 text-red-400' :
                          incident.status === 'in-progress' ? 'bg-orange-500/20 text-orange-400' :
                          'bg-emerald-500/20 text-emerald-400'
                        }`}>
                          {incident.status.replace('-', ' ')}
                        </span>
                      </div>
                      
                      <p className="text-slate-300 text-sm mb-2 flex items-center gap-1">
                        <MapPin className="w-3 h-3 text-cyan-400 flex-shrink-0" />
                        <span className="truncate">{incident.zone || incident.location}</span>
                      </p>
                      
                      <div className="flex items-center gap-3 text-xs text-slate-400 flex-wrap">
                        <span className={`px-2 py-1 rounded border ${getSeverityBadge(incident.severity)}`}>
                          {incident.severity}
                        </span>
                        <span className="bg-slate-700/50 px-2 py-1 rounded">
                          {incident.type?.replace('_', ' ') || 'incident'}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {getTimeSince(incident.timestamp)}
                        </span>
                      </div>

                      {/* Expanded Details */}
                      {isExpanded && incident.description && (
                        <div className="mt-3 pt-3 border-t border-slate-700">
                          <p className="text-slate-300 text-xs leading-relaxed">
                            {incident.description}
                          </p>
                        </div>
                      )}
                    </div>
                    
                    <div className="flex flex-col gap-1 ml-2">
                      <button
                        onClick={() => handleAcknowledge(incident.id)}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-xs font-medium flex-shrink-0 transition-colors"
                      >
                        Update
                      </button>
                      {incident.description && (
                        <button
                          onClick={() => toggleExpand(incident.id)}
                          className="bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-1 rounded text-xs font-medium flex-shrink-0 transition-colors flex items-center justify-center"
                        >
                          <ChevronDown className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                        </button>
                      )}
                    </div>
                  </div>
                  
                  {/* Progress indicator for in-progress items */}
                  {incident.status === 'in-progress' && (
                    <div className="mt-2 pt-2 border-t border-slate-700">
                      <div className="flex items-center gap-2 mb-1">
                        <Activity className="w-3 h-3 text-orange-400" />
                        <span className="text-xs text-slate-400">In Progress</span>
                      </div>
                      <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
                        <div className="h-full bg-orange-500 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default IncidentPanel;