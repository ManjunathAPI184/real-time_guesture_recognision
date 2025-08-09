import streamlit as st
import streamlit.components.v1 as components
import base64
import cv2
import numpy as np
from PIL import Image
import io

def camera_input_live():
    """Custom camera component with live feed"""
    
    camera_html = """
    <div id="camera-container">
        <video id="video" width="640" height="480" autoplay muted></video>
        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        <div id="controls">
            <button onclick="startCamera()">Start Camera</button>
            <button onclick="captureFrame()">Capture Frame</button>
            <button onclick="toggleProcessing()">Toggle Live Processing</button>
        </div>
        <div id="status">Camera Status: Ready</div>
    </div>

    <script>
    let video = document.getElementById('video');
    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    let stream = null;
    let isProcessing = false;
    let processingInterval = null;

    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: 640, 
                    height: 480,
                    frameRate: { ideal: 30, max: 30 }
                }
            });
            video.srcObject = stream;
            document.getElementById('status').innerText = 'Camera Status: Active';
            
            // Send ready signal to Streamlit
            window.parent.postMessage({
                type: 'camera_ready'
            }, '*');
            
        } catch (err) {
            console.error('Camera error:', err);
            document.getElementById('status').innerText = 'Camera Status: Error - ' + err.message;
        }
    }

    function captureFrame() {
        if (video.videoWidth === 0) return;
        
        ctx.drawImage(video, 0, 0, 640, 480);
        let imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send frame to Streamlit
        window.parent.postMessage({
            type: 'frame_captured',
            data: imageData
        }, '*');
    }

    function toggleProcessing() {
        isProcessing = !isProcessing;
        
        if (isProcessing) {
            // Capture frames every 100ms for real-time processing
            processingInterval = setInterval(captureFrame, 100);
            document.getElementById('status').innerText = 'Camera Status: Live Processing';
        } else {
            clearInterval(processingInterval);
            document.getElementById('status').innerText = 'Camera Status: Active';
        }
    }

    // Auto-start camera when component loads
    window.addEventListener('load', startCamera);
    </script>

    <style>
    #camera-container {
        text-align: center;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        background: #f9f9f9;
    }
    #video {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    #controls {
        margin: 10px 0;
    }
    #controls button {
        margin: 5px;
        padding: 10px 20px;
        background: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    #controls button:hover {
        background: #45a049;
    }
    #status {
        margin-top: 10px;
        font-weight: bold;
        color: #333;
    }
    </style>
    """
    
    # Render the component
    component_value = components.html(camera_html, height=600)
    return component_value
