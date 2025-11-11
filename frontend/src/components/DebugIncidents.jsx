import React, { useState } from 'react';
import { AlertCircle, CheckCircle, RefreshCw, Database, FileJson } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

const ApiDebugTool = () => {
  const [incidentsData, setIncidentsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchIncidents = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/incidents`);
      const data = await response.json();
      
      console.log('📊 Full API Response:', data);
      console.log('📋 Incidents Array:', data.incidents);
      console.log('🔢 Number of Incidents:', data.incidents?.length || 0);
      
      setIncidentsData(data);
    } catch (err) {
      console.error('❌ Error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getActiveIncidents = () => {
    if (!incidentsData?.incidents) return [];
    return incidentsData.incidents.filter(i => i.status !== 'resolved');
  };

  const getIncidentsByStatus = () => {
    if (!incidentsData?.incidents) return {};
    
    const statusCounts = {
      reported: 0,
      'in-progress': 0,
      resolved: 0,
      total: incidentsData.incidents.length
    };

    incidentsData.incidents.forEach(incident => {
      if (statusCounts.hasOwnProperty(incident.status)) {
        statusCounts[incident.status]++;
      }
    });

    return statusCounts;
  };

  const getIncidentsBySeverity = () => {
    if (!incidentsData?.incidents) return {};
    
    const severityCounts = {
      critical: 0,
      high: 0,
      medium: 0,
      low: 0
    };

    incidentsData.incidents.forEach(incident => {
      if (severityCounts.hasOwnProperty(incident.severity)) {
        severityCounts[incident.severity]++;
      }
    });

    return severityCounts;
  };

  const activeIncidents = getActiveIncidents();
  const statusCounts = getIncidentsByStatus();
  const severityCounts = getIncidentsBySeverity();

  return (
    <div className="min-h-screen bg-slate-950 p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-slate-900 border border-slate-700 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Database className="w-8 h-8 text-cyan-400" />
              <div>
                <h1 className="text-2xl font-bold text-slate-100">API Debug Tool</h1>
                <p className="text-sm text-slate-400">Incidents Endpoint Inspector</p>
              </div>
            </div>
            <button
              onClick={fetchIncidents}
              disabled={loading}
              className="bg-cyan-600 hover:bg-cyan-700 disabled:bg-slate-700 text-white px-6 py-3 rounded-lg font-medium flex items-center gap-2 transition-colors"
            >
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Fetching...' : 'Fetch Incidents'}
            </button>
          </div>

          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
            <div className="flex items-center gap-2 text-sm">
              <span className="text-slate-400">Endpoint:</span>
              <code className="text-cyan-400 bg-slate-950 px-3 py-1 rounded">
                GET {API_BASE_URL}/analytics/dashboard/incidents
              </code>
            </div>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-red-400 mb-1">Connection Error</h3>
                <p className="text-sm text-red-300">{error}</p>
                <p className="text-xs text-red-400 mt-2">
                  Make sure your Flask backend is running on http://127.0.0.1:5000
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Success State */}
        {incidentsData && !error && (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-slate-900 border border-slate-700 rounded-lg p-4">
                <div className="text-slate-400 text-sm mb-1">Total Incidents</div>
                <div className="text-3xl font-bold text-slate-100">{statusCounts.total}</div>
              </div>
              <div className="bg-slate-900 border border-emerald-700 rounded-lg p-4">
                <div className="text-slate-400 text-sm mb-1">Active</div>
                <div className="text-3xl font-bold text-emerald-400">{activeIncidents.length}</div>
              </div>
              <div className="bg-slate-900 border border-red-700 rounded-lg p-4">
                <div className="text-slate-400 text-sm mb-1">Reported</div>
                <div className="text-3xl font-bold text-red-400">{statusCounts.reported}</div>
              </div>
              <div className="bg-slate-900 border border-orange-700 rounded-lg p-4">
                <div className="text-slate-400 text-sm mb-1">In Progress</div>
                <div className="text-3xl font-bold text-orange-400">{statusCounts['in-progress']}</div>
              </div>
            </div>

            {/* Breakdown by Status & Severity */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Status Breakdown */}
              <div className="bg-slate-900 border border-slate-700 rounded-lg p-6">
                <h3 className="font-semibold text-slate-100 mb-4 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-cyan-400" />
                  Status Breakdown
                </h3>
                <div className="space-y-3">
                  {Object.entries(statusCounts).map(([status, count]) => (
                    <div key={status} className="flex items-center justify-between">
                      <span className="text-slate-300 capitalize">{status.replace('-', ' ')}</span>
                      <span className="font-bold text-slate-100 bg-slate-800 px-3 py-1 rounded">
                        {count}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Severity Breakdown */}
              <div className="bg-slate-900 border border-slate-700 rounded-lg p-6">
                <h3 className="font-semibold text-slate-100 mb-4 flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-orange-400" />
                  Severity Breakdown
                </h3>
                <div className="space-y-3">
                  {Object.entries(severityCounts).map(([severity, count]) => {
                    const colors = {
                      critical: 'text-red-400 bg-red-500/10',
                      high: 'text-orange-400 bg-orange-500/10',
                      medium: 'text-yellow-400 bg-yellow-500/10',
                      low: 'text-blue-400 bg-blue-500/10'
                    };
                    return (
                      <div key={severity} className="flex items-center justify-between">
                        <span className={`capitalize ${colors[severity].split(' ')[0]}`}>
                          {severity}
                        </span>
                        <span className={`font-bold px-3 py-1 rounded ${colors[severity]}`}>
                          {count}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Raw JSON Response */}
            <div className="bg-slate-900 border border-slate-700 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-slate-100 flex items-center gap-2">
                  <FileJson className="w-5 h-5 text-cyan-400" />
                  Raw API Response
                </h3>
                <span className={`px-3 py-1 rounded text-xs font-medium ${
                  incidentsData.success 
                    ? 'bg-emerald-500/20 text-emerald-400' 
                    : 'bg-red-500/20 text-red-400'
                }`}>
                  {incidentsData.success ? 'Success' : 'Failed'}
                </span>
              </div>
              <div className="bg-slate-950 rounded-lg p-4 overflow-x-auto">
                <pre className="text-xs text-slate-300">
                  {JSON.stringify(incidentsData, null, 2)}
                </pre>
              </div>
            </div>

            {/* Sample Incidents */}
            {activeIncidents.length > 0 && (
              <div className="bg-slate-900 border border-slate-700 rounded-lg p-6">
                <h3 className="font-semibold text-slate-100 mb-4">
                  Active Incidents Preview (First 3)
                </h3>
                <div className="space-y-3">
                  {activeIncidents.slice(0, 3).map((incident, idx) => (
                    <div key={idx} className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <span className={`px-2 py-1 rounded text-xs font-bold ${
                              incident.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                              incident.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                              incident.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                              'bg-blue-500/20 text-blue-400'
                            }`}>
                              {incident.severity}
                            </span>
                            <span className={`px-2 py-1 rounded text-xs font-bold ${
                              incident.status === 'reported' ? 'bg-red-500/20 text-red-400' :
                              'bg-orange-500/20 text-orange-400'
                            }`}>
                              {incident.status}
                            </span>
                          </div>
                          <div className="text-slate-300 text-sm">
                            <strong>Category:</strong> {incident.category || 'N/A'}
                          </div>
                          <div className="text-slate-300 text-sm">
                            <strong>Location:</strong> {incident.zone || incident.location || 'N/A'}
                          </div>
                          <div className="text-slate-400 text-xs mt-1">
                            <strong>ID:</strong> {incident.id}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* No Incidents Message */}
            {activeIncidents.length === 0 && statusCounts.total === 0 && (
              <div className="bg-slate-900 border border-slate-700 rounded-lg p-8 text-center">
                <CheckCircle className="w-16 h-16 mx-auto mb-4 text-slate-600" />
                <h3 className="text-xl font-semibold text-slate-400 mb-2">No Incidents Found</h3>
                <p className="text-slate-500">
                  Your API returned an empty incidents array. Your backend might not have any incidents yet.
                </p>
              </div>
            )}
          </>
        )}

        {/* Initial State */}
        {!incidentsData && !error && !loading && (
          <div className="bg-slate-900 border border-slate-700 rounded-lg p-12 text-center">
            <Database className="w-16 h-16 mx-auto mb-4 text-slate-600" />
            <h3 className="text-xl font-semibold text-slate-400 mb-2">Ready to Debug</h3>
            <p className="text-slate-500 mb-6">
              Click "Fetch Incidents" to see what your API is returning
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ApiDebugTool;