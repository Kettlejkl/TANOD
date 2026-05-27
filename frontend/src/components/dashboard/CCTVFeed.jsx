import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { Camera, Edit3, Save, X, Plus, Trash2, List } from 'lucide-react';

const FENCE_COLORS = [
  { primary: 'rgba(0, 255, 255, 0.9)', fill: 'rgba(0, 255, 255, 0.15)' },
  { primary: 'rgba(255, 0, 255, 0.9)', fill: 'rgba(255, 0, 255, 0.15)' },
  { primary: 'rgba(0, 255, 0, 0.9)', fill: 'rgba(0, 255, 0, 0.15)' },
  { primary: 'rgba(255, 255, 0, 0.9)', fill: 'rgba(255, 255, 0, 0.15)' },
  { primary: 'rgba(255, 128, 0, 0.9)', fill: 'rgba(255, 128, 0, 0.15)' },
];

const CCTVFeed = ({ cameraId, title, isActive }) => {
  const [frame, setFrame] = useState(null);
  const [count, setCount] = useState(0);
  const [insideCount, setInsideCount] = useState(0);
  const [editMode, setEditMode] = useState(false);
  const [showList, setShowList] = useState(false);
  const [geoFence, setGeoFence] = useState(null);
  const [selectedFenceId, setSelectedFenceId] = useState(null);
  const [placingPoints, setPlacingPoints] = useState(false);
  const [tempPoints, setTempPoints] = useState([]);
  const [isPolygonClosed, setIsPolygonClosed] = useState(false);
  const [fenceName, setFenceName] = useState('');
  const [draggingIndex, setDraggingIndex] = useState(null);
  const [editingFenceName, setEditingFenceName] = useState(null);
  const [newFenceName, setNewFenceName] = useState('');
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const socketRef = useRef(null);

  useEffect(() => {
    socketRef.current = io('http://localhost:5000');
    socketRef.current.emit('join_camera', { camera_id: cameraId });
    socketRef.current.on('video_frame', (data) => {
      if (data.camera_id === cameraId) {
        setFrame(`data:image/jpeg;base64,${data.frame}`);
        setCount(data.count || 0);
        setInsideCount(data.inside_count || 0);
      }
    });
    loadGeoFences();
    return () => socketRef.current?.disconnect();
  }, [cameraId]);

  const loadGeoFences = async () => {
    try {
      const res = await fetch(`http://localhost:5000/api/video/geo-fences/${cameraId}`);
      const data = await res.json();
      if (data.success && data.fences && data.fences.length > 0) {
        setGeoFence(data.fences[0]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const saveCurrentFence = async () => {
    if (tempPoints.length < 3 || !isPolygonClosed) {
      alert('Complete the polygon first');
      return;
    }
    const name = fenceName.trim() || 'Zone 1';
    try {
      const res = await fetch(`http://localhost:5000/api/video/geo-fences/${cameraId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, points: tempPoints })
      });
      if (res.ok) {
        setPlacingPoints(false);
        setTempPoints([]);
        setFenceName('');
        setIsPolygonClosed(false);
        loadGeoFences();
        socketRef.current?.emit('update_geo_fences', { camera_id: cameraId });
      }
    } catch (err) {
      alert('Failed to save');
    }
  };

  const deleteFence = async () => {
    if (!geoFence || !confirm('Delete?')) return;
    try {
      const res = await fetch(`http://localhost:5000/api/video/geo-fences/${cameraId}/${geoFence.id}`, { method: 'DELETE' });
      if (res.ok) {
        setGeoFence(null);
        setSelectedFenceId(null);
        socketRef.current?.emit('update_geo_fences', { camera_id: cameraId });
      }
    } catch (err) {
      alert('Failed');
    }
  };

  const renameFence = async (name) => {
    if (!name.trim() || !geoFence) return;
    try {
      const res = await fetch(`http://localhost:5000/api/video/geo-fences/${cameraId}/${geoFence.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim() })
      });
      if (res.ok) {
        setEditingFenceName(null);
        setNewFenceName('');
        loadGeoFences();
        socketRef.current?.emit('update_geo_fences', { camera_id: cameraId });
      }
    } catch (err) {}
  };

  const updatePoints = async (fenceId, points) => {
    try {
      const res = await fetch(`http://localhost:5000/api/video/geo-fences/${cameraId}/${fenceId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points })
      });
      if (res.ok) socketRef.current?.emit('update_geo_fences', { camera_id: cameraId });
    } catch (err) {}
  };

  const drawPolygon = (ctx, points, scaleX, scaleY, color, fillColor, label, complete) => {
    if (points.length < 2) return;
    const pts = points.map(([x, y]) => [x * scaleX, y * scaleY]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    if (complete) {
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
    }
    ctx.stroke();
    pts.forEach(([x, y], i) => {
      const first = i === 0;
      const r = first ? 12 : 10;
      ctx.fillStyle = first ? 'rgba(255,215,0,0.9)' : color;
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = 'white';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.font = 'bold 11px monospace';
      ctx.fillStyle = first ? 'rgba(139,69,0,0.9)' : 'rgba(0,0,0,0.8)';
      ctx.fillRect(x - 12, y - 28, 24, 18);
      ctx.strokeStyle = first ? 'rgba(255,215,0,0.9)' : color;
      ctx.lineWidth = 1;
      ctx.strokeRect(x - 12, y - 28, 24, 18);
      ctx.fillStyle = 'white';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${i + 1}`, x, y - 19);
    });
    if (label && complete) {
      const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
      const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      ctx.font = 'bold 14px monospace';
      const w = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0,0,0,0.8)';
      ctx.fillRect(cx - w/2 - 8, cy - 12, w + 16, 24);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - w/2 - 8, cy - 12, w + 16, 24);
      ctx.fillStyle = 'white';
      ctx.fillText(label, cx, cy);
    }
  };

  useEffect(() => {
    if (!canvasRef.current || !imageRef.current || !imageSize.width) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = imageRef.current.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scaleX = canvas.width / imageSize.width;
    const scaleY = canvas.height / imageSize.height;

    if (editMode && geoFence) {
      const c = FENCE_COLORS[0];
      const sel = selectedFenceId === geoFence.id;
      const op = sel ? 1 : 0.3;
      const col = c.primary.replace('0.9)', `${0.9 * op})`);
      const fill = c.fill.replace('0.15)', `${0.15 * op})`);
      drawPolygon(ctx, geoFence.points, scaleX, scaleY, col, fill, geoFence.name, true);
    } else if (!editMode && geoFence && geoFence.enabled) {
      const c = FENCE_COLORS[0];
      drawPolygon(ctx, geoFence.points, scaleX, scaleY, c.primary, c.fill, geoFence.name, true);
    }

    if (placingPoints && tempPoints.length > 0) {
      const c = FENCE_COLORS[0];
      const pts = tempPoints.map(([x, y]) => [x * scaleX, y * scaleY]);
      if (tempPoints.length > 1) {
        ctx.strokeStyle = c.primary;
        ctx.lineWidth = 3;
        ctx.setLineDash([10, 5]);
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (tempPoints.length >= 3 && !isPolygonClosed) {
        ctx.strokeStyle = 'rgba(255,215,0,0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(pts[pts.length - 1][0], pts[pts.length - 1][1]);
        ctx.lineTo(pts[0][0], pts[0][1]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      pts.forEach(([x, y], i) => {
        const first = i === 0;
        const r = first ? 12 : 10;
        ctx.fillStyle = first ? 'rgba(255,215,0,0.9)' : c.primary;
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.font = 'bold 11px monospace';
        ctx.fillStyle = first ? 'rgba(139,69,0,0.9)' : 'rgba(0,0,0,0.8)';
        ctx.fillRect(x - 12, y - 28, 24, 18);
        ctx.strokeStyle = first ? 'rgba(255,215,0,0.9)' : c.primary;
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 12, y - 28, 24, 18);
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${i + 1}`, x, y - 19);
      });
    }
  }, [editMode, geoFence, tempPoints, imageSize, placingPoints, selectedFenceId, isPolygonClosed]);

  const getMousePos = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || !imageSize.width) return [0, 0];
    const rect = canvas.getBoundingClientRect();
    const sx = imageSize.width / canvas.width;
    const sy = imageSize.height / canvas.height;
    return [Math.round((e.clientX - rect.left) * sx), Math.round((e.clientY - rect.top) * sy)];
  };

  const handleClick = (e) => {
    if (!editMode || !placingPoints || isPolygonClosed) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const sx = canvas.width / imageSize.width;
    const sy = canvas.height / imageSize.height;
    if (tempPoints.length >= 3) {
      const [fx, fy] = tempPoints[0];
      const dx = cx - fx * sx;
      const dy = cy - fy * sy;
      if (Math.sqrt(dx * dx + dy * dy) < 30) {
        setIsPolygonClosed(true);
        return;
      }
    }
    setTempPoints([...tempPoints, getMousePos(e)]);
  };

  const handleMouseDown = (e) => {
    if (!editMode || placingPoints || !selectedFenceId || !geoFence) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const sx = canvas.width / imageSize.width;
    const sy = canvas.height / imageSize.height;
    const idx = geoFence.points.findIndex(([px, py]) => {
      const d = Math.sqrt((px * sx - mx) ** 2 + (py * sy - my) ** 2);
      return d < 25;
    });
    if (idx !== -1) setDraggingIndex(idx);
  };

  const handleMouseMove = (e) => {
    if (draggingIndex === null || !selectedFenceId || !geoFence) return;
    const [x, y] = getMousePos(e);
    const cx = Math.max(0, Math.min(x, imageSize.width));
    const cy = Math.max(0, Math.min(y, imageSize.height));
    setGeoFence({ ...geoFence, points: geoFence.points.map((p, i) => i === draggingIndex ? [cx, cy] : p) });
  };

  const handleMouseUp = () => {
    if (draggingIndex !== null && selectedFenceId && geoFence) {
      updatePoints(geoFence.id, geoFence.points);
    }
    setDraggingIndex(null);
  };

  return (
    <div className={`bg-slate-900 border ${isActive ? 'border-red-500' : 'border-slate-700'} rounded-lg overflow-hidden h-full flex flex-col`}>
      <div className={`px-3 py-2 flex items-center justify-between ${isActive ? 'bg-red-900/30' : 'bg-slate-800'}`}>
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-red-400 animate-pulse' : 'bg-emerald-400'}`} />
          <span className="font-mono text-slate-100 text-sm font-semibold">{cameraId}</span>
          <span className="text-slate-300 text-xs truncate">{title}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {editMode ? (
            <>
              {!placingPoints && !geoFence && <button onClick={() => { setPlacingPoints(true); setTempPoints([]); setSelectedFenceId(null); setFenceName(''); setIsPolygonClosed(false); }} className="p-1.5 bg-emerald-500 hover:bg-emerald-600 text-white rounded transition-colors"><Plus className="w-4 h-4" /></button>}
              {placingPoints && (
                <>
                  <button onClick={saveCurrentFence} disabled={!isPolygonClosed} className={`p-1.5 text-white rounded transition-colors ${isPolygonClosed ? 'bg-emerald-500 hover:bg-emerald-600' : 'bg-slate-600 opacity-50'}`}><Save className="w-4 h-4" /></button>
                  <button onClick={() => { setPlacingPoints(false); setTempPoints([]); setFenceName(''); setIsPolygonClosed(false); }} className="p-1.5 bg-yellow-500 hover:bg-yellow-600 text-white rounded transition-colors"><X className="w-4 h-4" /></button>
                </>
              )}
              <button onClick={() => { setEditMode(false); setPlacingPoints(false); setTempPoints([]); setSelectedFenceId(null); loadGeoFences(); }} className="p-1.5 bg-red-500 hover:bg-red-600 text-white rounded transition-colors"><X className="w-4 h-4" /></button>
            </>
          ) : (
            <button onClick={() => setEditMode(true)} className="p-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded transition-colors"><Edit3 className="w-4 h-4" /></button>
          )}
          {insideCount > 0 && <div className="bg-cyan-500/20 text-cyan-300 px-2 py-1 rounded text-xs font-mono">{insideCount}</div>}
          {isActive && <div className="bg-red-600 text-white px-2 py-1 rounded text-xs font-semibold animate-pulse">ALERT</div>}
        </div>
      </div>

      <div className="flex-1 bg-slate-950 relative overflow-hidden">
        {frame ? (
          <>
            <img ref={imageRef} src={frame} alt={`Camera ${cameraId}`} className="absolute inset-0 w-full h-full object-cover" onLoad={() => imageRef.current && setImageSize({ width: imageRef.current.naturalWidth, height: imageRef.current.naturalHeight })} />
            <canvas ref={canvasRef} className={`absolute inset-0 w-full h-full ${!editMode ? 'pointer-events-none' : placingPoints ? 'cursor-crosshair' : draggingIndex !== null ? 'cursor-grabbing' : 'cursor-default'}`} style={{ zIndex: 10 }} onClick={handleClick} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp} />
          </>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center"><Camera className="w-16 h-16 text-slate-600 animate-pulse" /></div>
        )}

        {editMode && (
          <div className="absolute top-2 left-2 space-y-2" style={{ zIndex: 20 }}>
            <div className="bg-blue-600 text-white text-xs px-3 py-1.5 rounded font-semibold shadow-lg flex items-center gap-2">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
              {!placingPoints ? (geoFence ? (selectedFenceId ? 'DRAG CORNERS' : 'CLICK ZONE TO EDIT') : 'CLICK + TO ADD ZONE') : isPolygonClosed ? `CLOSED (${tempPoints.length}) - SAVE` : tempPoints.length >= 3 ? `CLICK #1 (${tempPoints.length})` : `PLACE (${tempPoints.length})`}
            </div>
            {placingPoints && <input type="text" placeholder="Zone name" value={fenceName} onChange={(e) => setFenceName(e.target.value)} className="w-full px-2 py-1 bg-slate-800 text-white rounded text-xs border border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500" maxLength={50} />}
          </div>
        )}

        {editMode && !placingPoints && geoFence && (
          <div className="absolute top-2 right-2 bg-slate-900/95 text-white text-xs rounded shadow-lg p-2 max-w-xs backdrop-blur-sm" style={{ zIndex: 20 }}>
            <div className="font-semibold mb-2">Zone Info</div>
            <div className="space-y-1">
              {editingFenceName ? (
                <div className="flex items-center gap-1">
                  <input type="text" value={newFenceName} onChange={(e) => setNewFenceName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && renameFence(newFenceName)} className="flex-1 px-1 py-0.5 bg-slate-700 text-white rounded text-xs border border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-500" autoFocus maxLength={50} />
                  <button onClick={() => renameFence(newFenceName)} className="p-0.5 hover:text-emerald-400 transition-colors"><Save className="w-3 h-3" /></button>
                  <button onClick={() => { setEditingFenceName(null); setNewFenceName(''); }} className="p-0.5 hover:text-red-400 transition-colors"><X className="w-3 h-3" /></button>
                </div>
              ) : (
                <div className={`flex items-center gap-2 p-1 rounded transition-colors ${selectedFenceId === geoFence.id ? 'bg-blue-600' : 'bg-slate-800 hover:bg-slate-700'}`}>
                  <div onClick={() => setSelectedFenceId(geoFence.id)} className="flex items-center gap-2 flex-1 cursor-pointer">
                    <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: FENCE_COLORS[0].primary.replace('0.9)', '1)') }} />
                    <span className="truncate">{geoFence.name}</span>
                    <span className="text-slate-400 text-[10px]">({geoFence.points.length})</span>
                  </div>
                  <button onClick={() => { setEditingFenceName(geoFence.id); setNewFenceName(geoFence.name); }} className="p-1 hover:text-blue-400 transition-colors"><Edit3 className="w-3 h-3" /></button>
                  <button onClick={deleteFence} className="p-1 hover:text-red-400 transition-colors"><Trash2 className="w-3 h-3" /></button>
                </div>
              )}
            </div>
          </div>
        )}

        {!editMode && geoFence && geoFence.enabled && (
          <div className="absolute top-2 left-2 space-y-2">
            <div className="bg-slate-900/90 text-cyan-300 text-xs px-2 py-1 rounded font-mono flex items-center gap-2 backdrop-blur-sm">
              <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse"></div>
              {geoFence.name}
            </div>
            <div className="bg-slate-900/90 text-slate-300 text-xs px-2 py-1 rounded font-mono backdrop-blur-sm">Total: {count}</div>
          </div>
        )}

        <div className="absolute bottom-2 right-2 space-y-1">
          <div className="bg-slate-900/90 text-cyan-300 text-xs px-2 py-1 rounded font-mono">{new Date().toLocaleTimeString()}</div>
          <div className="bg-slate-900/90 text-emerald-400 text-xs px-2 py-1 rounded font-semibold flex items-center gap-1">
            <div className="w-1 h-1 bg-emerald-400 rounded-full animate-pulse"></div>
            LIVE
          </div>
        </div>
      </div>
    </div>
  );
};

export default CCTVFeed;