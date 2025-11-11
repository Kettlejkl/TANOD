import React, { useState, useEffect, useRef } from 'react';
import { Shield, RefreshCw, Wifi, User } from 'lucide-react';
// Import all components 
import LoginRegisterModal from './LoginRegisterModal';
import CCTVFeed from './CCTVFeed';
import IncidentPanel from './IncidentPanel';
import OccupancyPanel from './OccupancyPanel';
import AnalyticsDashboard from './AnalyticsDashboard';
import ROIPanel from './ROIPanel';
import AnomalyLogsPanel from './AnomalyLogsPanel';
import SystemHealthPanel from './SystemHealthPanel';
import IncidentMap from './IncidentMap';
// Import API service
import { apiService } from '../../services/apiService';

const Dashboard = () => {
  // State management
  const [incidents, setIncidents] = useState([]);
  const [occupancy, setOccupancy] = useState({});
  const [analytics, setAnalytics] = useState({});
  const [roiData, setRoiData] = useState({});
  const [anomalies, setAnomalies] = useState([]);
  const [systemHealth, setSystemHealth] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  
  // Authentication states
  const [user, setUser] = useState(null);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [pendingExportType, setPendingExportType] = useState(null);
  
  // Ref to trigger export after login
  const analyticsRef = useRef(null);

  // Check for existing auth on mount
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    const savedUser = localStorage.getItem('user');
    if (token && savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (e) {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user');
      }
    }
  }, []);

  // Data fetching
  const fetchData = async () => {
    try {
      const [incidentsData, occupancyData, analyticsData, roiDataRes, anomaliesData, healthData] = await Promise.all([
        apiService.getIncidents(),
        apiService.getOccupancy(),
        apiService.getAnalytics(),
        apiService.getROIData(),
        apiService.getAnomalyLogs(),
        apiService.getSystemHealth()
      ]);

      setIncidents(incidentsData);
      setOccupancy(occupancyData);
      setAnalytics(analyticsData);
      setRoiData(roiDataRes);
      setAnomalies(anomaliesData);
      setSystemHealth(healthData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 8000);
    return () => clearInterval(interval);
  }, []);

  // Event handlers
  const handleAcknowledge = async (incidentId) => {
    try {
      await apiService.acknowledgeIncident(incidentId);
      setIncidents(prev => prev.map(incident => 
        incident.id === incidentId 
          ? { ...incident, status: 'in-progress' }
          : incident
      ));
    } catch (error) {
      console.error('Failed to acknowledge incident:', error);
    }
  };

  const handleLoginSuccess = (userData) => {
    setUser(userData.user);
    setShowLoginModal(false);
    
    if (pendingExportType && analyticsRef.current) {
      console.log('✅ Login successful, triggering pending export:', pendingExportType);
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('trigger-export', { 
          detail: { type: pendingExportType } 
        }));
        setPendingExportType(null);
      }, 500);
    }
  };

  const handleExportClick = (exportType) => {
    if (!user) {
      console.log('🔒 Not authenticated, showing login modal. Export type:', exportType);
      setPendingExportType(exportType);
      setShowLoginModal(true);
    } else {
      console.log('✅ Already authenticated, export will proceed:', exportType);
    }
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
  };

  // Computed values
  const totalOccupancy = Object.values(occupancy).reduce((sum, data) => {
    return sum + (data?.current || 0);
  }, 0);
  
  const activeIncidents = incidents.filter(i => i.status !== 'resolved');
  const alertCameraIds = activeIncidents.map(i => i.camera_id);

  return (
    <div style={{ 
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      width: '100vw', 
      height: '100vh', 
      overflow: 'hidden', 
      margin: 0, 
      padding: 0,
      display: 'flex',
      flexDirection: 'column'
    }} className="bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b border-slate-700 flex-shrink-0" style={{ margin: 0, padding: '12px 24px' }}>
        <div className="flex items-center justify-between" style={{ width: '100%' }}>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600/20 rounded-lg border border-blue-500/30">
                <Shield className="w-6 h-6 text-blue-400" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-100">
                  TANOD Transportation Monitoring System
                </h1>
                <p className="text-slate-400 text-sm">Real-time Detection & Response</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Summary Cards */}
            <div className="flex items-center gap-3">
              <div className="bg-blue-500/20 border border-blue-400/30 rounded-lg px-4 py-2 text-center min-w-[80px]">
                <div className="text-cyan-400 text-lg font-bold">{totalOccupancy}</div>
                <div className="text-cyan-300 text-xs">Occupancy</div>
              </div>
              <div className="bg-red-500/20 border border-red-400/30 rounded-lg px-4 py-2 text-center min-w-[80px]">
                <div className="text-red-400 text-lg font-bold">{activeIncidents.length}</div>
                <div className="text-red-300 text-xs">Active</div>
              </div>
              <div className="bg-orange-500/20 border border-orange-400/30 rounded-lg px-4 py-2 text-center min-w-[80px]">
                <div className="text-orange-400 text-lg font-bold">{analytics.avgResponseTime || 0}m</div>
                <div className="text-orange-300 text-xs">Response</div>
              </div>
              <div className="bg-purple-500/20 border border-purple-400/30 rounded-lg px-4 py-2 text-center min-w-[80px]">
                <div className="text-purple-400 text-lg font-bold">{systemHealth.activeUIDs || 0}</div>
                <div className="text-purple-300 text-xs">UIDs</div>
              </div>
            </div>
            
            {/* User Status */}
            {user && (
              <div className="flex items-center gap-2 bg-emerald-500/20 border border-emerald-400/30 rounded-lg px-4 py-2">
                <User className="w-4 h-4 text-emerald-400" />
                <span className="text-emerald-300 text-sm font-medium">{user.username}</span>
                <button
                  onClick={handleLogout}
                  className="text-emerald-400 hover:text-emerald-300 text-xs ml-2 underline"
                >
                  Logout
                </button>
              </div>
            )}
            
            {/* Status and Refresh */}
            <div className="flex items-center gap-2 bg-slate-800/50 px-4 py-2 rounded-lg border border-slate-600">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <Wifi className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-slate-300 text-sm font-mono">
                {lastUpdate.toLocaleTimeString()}
              </div>
            </div>
            
            <button
              onClick={fetchData}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white rounded-lg font-semibold transition-all"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Main Content - FORCE FULL WIDTH */}
      <div style={{ 
        flex: 1,
        width: '100%', 
        height: '100%',
        overflow: 'auto', 
        padding: '20px', 
        margin: 0,
        boxSizing: 'border-box'
      }}>
        <div style={{ display: 'grid', gap: '20px', width: '100%', margin: 0, padding: 0 }}>
          {/* Top Section - CCTV + Incidents */}
          <div style={{ display: 'grid', gridTemplateColumns: '70% 30%', gap: '20px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', height: '480px' }}>
              <CCTVFeed 
                cameraId="CAM001" 
                title="Eastbound Platform"
                isActive={alertCameraIds.includes('CAM001')}
                trackedIds={incidents.find(i => i.camera_id === 'CAM001')?.tracked_ids || []}
              />
              <CCTVFeed 
                cameraId="CAM002" 
                title="Westbound Platform"
                isActive={alertCameraIds.includes('CAM002')}
                trackedIds={incidents.find(i => i.camera_id === 'CAM002')?.tracked_ids || []}
              />
            </div>
            <div style={{ height: '480px' }}>
              <IncidentPanel incidents={incidents} onAcknowledge={handleAcknowledge} />
            </div>
          </div>

          {/* Middle Section - Better Proportions */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '20px' }}>
            <div style={{ height: '340px' }}>
              <AnalyticsDashboard 
                ref={analyticsRef}
                onExportClick={handleExportClick} 
                user={user}
              />
            </div>
            <div style={{ height: '340px' }}>
              <OccupancyPanel occupancy={occupancy} />
            </div>
            <div style={{ height: '340px' }}>
              <AnomalyLogsPanel anomalies={anomalies} />
            </div>
            <div style={{ height: '340px' }}>
              <SystemHealthPanel health={systemHealth} />
            </div>
          </div>

          {/* Bottom Section - More Space */}
          <div style={{ display: 'grid', gridTemplateColumns: '60% 40%', gap: '20px' }}>
            <div style={{ height: '380px' }}>
              <IncidentMap incidents={incidents} />
            </div>
            <div style={{ height: '380px' }}>
              <ROIPanel roiData={roiData} />
            </div>
          </div>
        </div>
      </div>

      {/* Login/Register Modal */}
      {showLoginModal && (
        <LoginRegisterModal
          isOpen={showLoginModal}
          onClose={() => {
            setShowLoginModal(false);
            setPendingExportType(null);
          }}
          onSuccess={handleLoginSuccess}
        />
      )}
    </div>
  );
};

export default Dashboard;