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
    <video id="video" width="800" height="600" autoplay muted playsinline></video>
    <canvas id="overlay" width="800" height="600" style="position: absolute; border: 3px solid #00ff00;"></canvas>
    <div id="controls" style="margin-top: 15px;">
        <button onclick="startCamera()" style="background: #4CAF50; color: white; border: none; padding: 12px 24px; margin: 8px; border-radius: 8px; font-size: 16px;">Start Camera</button>
        <button onclick="toggleProcessing()" style="background: #2196F3; color: white; border: none; padding: 12px 24px; margin: 8px; border-radius: 8px; font-size: 16px;">Start Live Processing</button>
    </div>
    <div id="status" style="margin-top: 15px; font-weight: bold; font-size: 18px;">Camera Status: Ready</div>
    <div id="prediction" style="margin-top: 10px; font-size: 24px; color: #00ff00; font-weight: bold; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px;">Prediction: None</div>
</div>

<script>
let video = document.getElementById('video');
let overlay = document.getElementById('overlay');
let ctx = overlay.getContext('2d');
let stream = null;
let isProcessing = false;
let processingInterval = null;

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { min: 640, ideal: 1280, max: 1920 },
                height: { min: 480, ideal: 720, max: 1080 },
                frameRate: { ideal: 30 },
                facingMode: 'user'
            }
        });
        video.srcObject = stream;
        document.getElementById('status').innerText = 'Camera Status: Active - High Resolution';
    } catch (err) {
        console.error('Camera error:', err);
        document.getElementById('status').innerText = 'Camera Status: Error - ' + err.message;
    }
}

function captureAndProcess() {
    if (video.videoWidth === 0) return;
    
    // Create larger canvas for better processing
    let canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 600;
    let tempCtx = canvas.getContext('2d');
    
    // Draw current video frame at larger size
    tempCtx.drawImage(video, 0, 0, 800, 600);
    let imageData = canvas.toDataURL('image/jpeg', 0.9); // Higher quality
    
    // Show processing indicator
    document.getElementById('status').innerText = 'Camera Status: Processing Frame...';
    
    // Here you would send to your backend for processing
    // For now, simulate processing
    setTimeout(() => {
        document.getElementById('status').innerText = 'Camera Status: Live Processing Active';
        // You'll integrate with your gesture recognition here
    }, 100);
}

function toggleProcessing() {
    isProcessing = !isProcessing;
    
    if (isProcessing) {
        processingInterval = setInterval(captureAndProcess, 500); // Every 500ms for better processing
        document.getElementById('status').innerText = 'Camera Status: Live Processing Active';
    } else {
        clearInterval(processingInterval);
        document.getElementById('status').innerText = 'Camera Status: Active';
    }
}

// Auto-start camera
window.addEventListener('load', function() {
    setTimeout(startCamera, 1000);
});
</script>

<style>
#camera-container {
    position: relative;
    display: inline-block;
    text-align: center;
    max-width: 100%;
    background: #f0f0f0;
    border-radius: 15px;
    padding: 20px;
}
#video {
    border: 3px solid #00ff00;
    border-radius: 10px;
    background: #000;
    max-width: 100%;
    height: auto;
}
#overlay {
    position: absolute;
    top: 20px;
    left: 20px;
    pointer-events: none;
    border-radius: 10px;
}
#prediction {
    background: rgba(0, 255, 0, 0.1) !important;
    border: 2px solid #00ff00;
}
</style>
"""

    
    # Render the component
    component_value = components.html(camera_html, height=600)
    return component_value
