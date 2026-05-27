const API_BASE_URL = 'http://127.0.0.1:5000/api';

export const apiService = {
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
  
  getOccupancy: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/occupancy`);
      const data = await response.json();
      return data.success ? data.occupancy : {};
    } catch (error) {
      console.error('Error fetching occupancy:', error);
      return {};
    }
  },

  getAnalytics: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/analytics`);
      const data = await response.json();
      return data.success ? data.analytics : {};
    } catch (error) {
      console.error('Error fetching analytics:', error);
      return {};
    }
  },

  getROIData: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/roi`);
      const data = await response.json();
      return data.success ? data.roiData : {};
    } catch (error) {
      console.error('Error fetching ROI data:', error);
      return {};
    }
  },

  getAnomalyLogs: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/anomalies`);
      const data = await response.json();
      console.log('🔥 Raw API Response:', data);
      return data.success ? data.anomalies : [];
    } catch (error) {
      console.error('Error fetching anomaly logs:', error);
      return [];
    }
  },

  getSystemHealth: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/dashboard/system-health`);
      const data = await response.json();
      return data.success ? data.health : {};
    } catch (error) {
      console.error('Error fetching system health:', error);
      return {};
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