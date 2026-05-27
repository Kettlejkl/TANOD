import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/dashboard/Dashboard';
import DebugIncidents from './components/DebugIncidents';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        {/* Your default dashboard route */}
        <Route path="/" element={<Dashboard />} />

        {/* Your new DebugIncidents route */}
        <Route path="/debug-incidents" element={<DebugIncidents />} />
      </Routes>
    </Router>
  );
}

export default App;
