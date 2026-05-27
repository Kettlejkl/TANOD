import { useState, useEffect } from 'react';

export function useAPI(endpoint, options = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { interval } = options;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`/api${endpoint}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Set up polling if interval is specified
    let intervalId;
    if (interval) {
      intervalId = setInterval(fetchData, interval);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [endpoint, interval]);

  return { data, loading, error };
}