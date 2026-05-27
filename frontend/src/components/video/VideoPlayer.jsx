import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

const VideoPlayer = ({ cameraId, className = '' }) => {
  const imgRef = useRef(null);
  const socketRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [frameCount, setFrameCount] = useState(0);

  useEffect(() => {
    // Initialize socket connection
    socketRef.current = io('http://localhost:5000');
    
    socketRef.current.on('connect', () => {
      setIsConnected(true);
      // Join the specific camera room
      socketRef.current.emit('join_camera', { camera_id: cameraId });
    });
    
    socketRef.current.on('disconnect', () => {
      setIsConnected(false);
    });
    
    socketRef.current.on('video_frame', (data) => {
      if (data.camera_id === cameraId && imgRef.current) {
        imgRef.current.src = `data:image/jpeg;base64,${data.frame}`;
        setFrameCount(prev => prev + 1);
      }
    });
    
    // Start the camera stream
    fetch(`/api/video/start/${cameraId}`);
    
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [cameraId]);

  return (
    <div className={`video-player ${className}`}>
      <div className="relative">
        <img
          ref={imgRef}
          alt={`Camera ${cameraId}`}
          className="w-full h-auto rounded-lg shadow-lg"
        />
        <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-sm">
          {isConnected ? (
            <span className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
              Live • {frameCount} frames
            </span>
          ) : (
            <span className="flex items-center">
              <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
              Disconnected
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;