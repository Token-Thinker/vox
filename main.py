#!/usr/bin/env python3
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import io
import base64
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import librosa
import cv2
import os
import glob
import threading
import time

# Camera fallback if libcamera/Picamera2 not available
try:
    from picamera2 import Picamera2
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False

# Global variables for video handling
current_video_source = None
picamera_obj = None
opencv_capture_obj = None
video_lock = threading.Lock()
MEDIA_DIR = "media"  # Directory for pre-recorded videos
os.makedirs(MEDIA_DIR, exist_ok=True)

# Initialize app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
app.title = "Vox Executive Anomaly Dashboard"

# Initialize sound/image detectors (fallback to dummy if models missing)
class _DummyDetector:
    def __init__(self):
        self.sample_rate = 44100
    def predict(self, *args, **kwargs):
        return 0.0

# Dummy default detectors
sound_detector = _DummyDetector()
image_detector = _DummyDetector()

# Create error image
def create_error_image(message):
    img = Image.new('RGB', (640, 480), color='red')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), message, fill='white')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

error_image_src = create_error_image("Video source unavailable")

# Get available video sources
def get_video_sources():
    sources = []
    
    # Pi Camera option
    if PI_CAMERA_AVAILABLE:
        sources.append({'label': 'Pi Camera', 'value': 'pi'})
    
    # USB Cameras (try indices 0-3)
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            sources.append({'label': f'USB Camera {i}', 'value': f'usb_{i}'})
        cap.release()
    
    print("scanning for recorded media")
    # Pre-recorded videos
    video_files = glob.glob(os.path.join(MEDIA_DIR, "*.mp4")) 
    for video_file in video_files:
        filename = os.path.basename(video_file)
        print("New Media Found" + filename)
        sources.append({'label': f'Video: {filename}', 'value': f'file_{video_file}'})
    
    return sources

VIDEO_SOURCES = get_video_sources()
DEFAULT_SOURCE = VIDEO_SOURCES[0]['value'] if VIDEO_SOURCES else None

# --- Dummy data examples for static visualizations ---
# Waveform
t = np.linspace(0, 1, 500)
wave = np.sin(2 * np.pi * 5 * t)
wave_line_fig = go.Figure(go.Scatter(x=t, y=wave, mode='lines', name='Waveform (Line)'))
wave_line_fig.update_layout(
    title='Waveform (Line)',
    xaxis_title='Time [s]',
    yaxis_title='Amplitude',
    template="plotly_dark"
)

# Spectrogram
S = np.abs(librosa.stft(wave, n_fft=256))
spec_heat_fig = go.Figure(go.Heatmap(
    z=20 * np.log10(S + 1e-6), 
    x=np.arange(S.shape[1]), 
    y=np.arange(S.shape[0]),
    colorscale='Viridis'
))
spec_heat_fig.update_layout(
    title='Spectrogram (Heatmap)',
    xaxis_title='Frame',
    yaxis_title='Frequency Bin',
    template="plotly_dark"
)

# Video source switching
def switch_video_source(new_source):
    global current_video_source, picamera_obj, opencv_capture_obj
    
    # Release current resources
    if current_video_source == 'pi' and picamera_obj is not None:
        picamera_obj.stop()
        picamera_obj = None
    elif opencv_capture_obj is not None:
        opencv_capture_obj.release()
        opencv_capture_obj = None
    
    # Open new source
    try:
        if new_source == 'pi':
            if PI_CAMERA_AVAILABLE:
                picamera_obj = Picamera2()
                picamera_obj.start()
            else:
                raise Exception("Pi camera not available")
        elif new_source.startswith('usb_'):
            index = int(new_source.split('_')[1])
            opencv_capture_obj = cv2.VideoCapture(index)
            if not opencv_capture_obj.isOpened():
                raise Exception(f"Could not open USB camera {index}")
        elif new_source.startswith('file_'):
            filepath = new_source.split('file_', 1)[1]
            opencv_capture_obj = cv2.VideoCapture(filepath)
            if not opencv_capture_obj.isOpened():
                raise Exception(f"Could not open video file: {filepath}")
        current_video_source = new_source
    except Exception as e:
        print(f"Error switching video source: {e}")
        current_video_source = None

# Cleanup function
def cleanup_resources():
    if picamera_obj is not None:
        picamera_obj.stop()
    if opencv_capture_obj is not None:
        opencv_capture_obj.release()

# ================================================
# Improved Layout with Camera Source Selection
# ================================================
app.layout = dbc.Container(fluid=True, children=[
    # Header
    dbc.Row([
        dbc.Col(html.H1("Vox Executive Anomaly Dashboard", 
                        className="text-center my-4"),
                width=12)
    ], className="bg-dark"),
    
    # Real-time monitoring
    dbc.Row([
        # Camera Card
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.Span("Camera Stream", className="me-auto"),
                dbc.Button("Refresh Sources", id="refresh-sources", size="sm", className="me-2"),
                dcc.Dropdown(
                    id='video-source-dropdown',
                    options=VIDEO_SOURCES,
                    value=DEFAULT_SOURCE,
                    clearable=False,
                    style={'width': '300px'}
                )
            ], className="bg-primary text-white d-flex justify-content-between align-items-center"),
            dbc.CardBody([
                html.Div(
                    html.Img(
                        id="video-frame", 
                        style={"border": "1px solid #444", "height": "400px"}
                    ),
                    className="mb-3"
                ),
                dcc.Graph(
                    id="video-anom-gauge", 
                    config={"displayModeBar": False},
                    style={'height': '150px'}
                )
            ])
        ], className="shadow"), width=8, className="mb-4"),
        
        # Audio Card
        dbc.Col(dbc.Card([
            dbc.CardHeader("Audio Monitoring", className="bg-success text-white"),
            dbc.CardBody([
                dcc.Graph(
                    id="audio-anom-gauge", 
                    config={"displayModeBar": False},
                    style={'height': '250px'}
                ),
                dbc.Progress(id="audio-level", value=0, striped=True, animated=True, 
                            className="my-3", style={"height": "20px"}),
                html.Div("Real-time audio anomaly detection", className="text-center mt-3")
            ])
        ], className="shadow"), width=4, className="mb-4")
    ]),
    
    # Visualization examples
    dbc.Row([
        dbc.Col(html.H2("Analytical Visualizations", className="my-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Waveform (Line)"),
            dbc.CardBody(dcc.Graph(figure=wave_line_fig))
        ], className="h-100 shadow"), md=6, lg=4, className="mb-4"),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("Spectrogram (Heatmap)"),
            dbc.CardBody(dcc.Graph(figure=spec_heat_fig))
        ], className="h-100 shadow"), md=6, lg=4, className="mb-4"),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("System Status"),
            dbc.CardBody([
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Div("Camera Status", className="fw-bold"),
                        html.Div(id="camera-status", children="Not initialized")
                    ]),
                    dbc.ListGroupItem([
                        html.Div("Audio Status", className="fw-bold"),
                        html.Div("Ready")
                    ]),
                    dbc.ListGroupItem([
                        html.Div("Anomaly Detection", className="fw-bold"),
                        html.Div("Active")
                    ]),
                ], flush=True)
            ])
        ], className="h-100 shadow"), md=6, lg=4, className="mb-4"),
    ]),
    
    # Hidden components
    dcc.Interval(id="interval-video", interval=1000, n_intervals=0),
    dcc.Interval(id="interval-audio", interval=1000, n_intervals=0),
    dcc.Store(id='current-video-source', data=DEFAULT_SOURCE),
    
    # Status footer
    dbc.Row([
        dbc.Col(html.Div([
            html.Small(f"Media Directory: {MEDIA_DIR}"),
            html.Br(),
            html.Small(id="timestamp", className="text-muted")
        ], className="text-center mt-4 p-2"), width=12)
    ], className="bg-dark text-light")
], style={'backgroundColor': '#1a1a1a'})

# ================================================
# Callbacks
# ================================================
@app.callback(
    Output("video-frame", "src"),
    Output("video-anom-gauge", "figure"),
    Output("camera-status", "children"),
    Input("interval-video", "n_intervals"),
    Input("video-source-dropdown", "value"),
    State('current-video-source', 'data')
)
def update_video(n, new_source, current_source):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle source switching
    if trigger_id == "video-source-dropdown" and new_source != current_source:
        with video_lock:
            switch_video_source(new_source)
        return dash.no_update, dash.no_update, f"Switching to {new_source}"
    
    # Capture frame
    frame = None
    status = "Active"
    
    try:
        if current_video_source == 'pi' and picamera_obj is not None:
            frame = picamera_obj.capture_array()
        elif opencv_capture_obj is not None:
            ret, frame = opencv_capture_obj.read()
            if not ret:
                # Reset video if we've reached the end
                opencv_capture_obj.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = opencv_capture_obj.read()
                if not ret:
                    status = "Error: End of video"
    except Exception as e:
        status = f"Error: {str(e)}"
        frame = None
    
    # Process frame
    if frame is not None:
        # Convert to PIL Image
        if current_video_source != 'pi':  # OpenCV frames are BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        
        # Convert to base64
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        src = f"data:image/jpeg;base64,{img_b64}"
        
        # Dummy anomaly score (replace with actual model prediction)
        score = 0.2 if np.random.random() > 0.8 else np.random.random()/5
        
        # Create gauge figure
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.6], 'color': "green"},
                    {'range': [0.6, 0.8], 'color': "orange"},
                    {'range': [0.8, 1], 'color': "red"}
                ],
            }
        ))
        gauge_fig.update_layout(
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            font={'color': "white", 'family': "Arial"},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return src, gauge_fig, status
    
    # Return error state
    error_gauge = go.Figure()
    error_gauge.update_layout(
        title="Camera Error",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return error_image_src, error_gauge, status

@app.callback(
    Output("audio-anom-gauge", "figure"),
    Output("audio-level", "value"),
    Input("interval-audio", "n_intervals")
)
def update_audio(n):
    # Dummy implementation - replace with actual audio processing
    duration = 1.0
    fs = 44100
    recording = np.random.randn(int(duration * fs)) * 0.1
    
    # Create random audio level
    audio_level = min(100, max(0, int(50 + np.random.randn() * 20)))
    
    # Create random anomaly score
    score = 0.3 if np.random.random() > 0.9 else np.random.random()/3
    
    # Create gauge figure
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 0.6], 'color': "green"},
                {'range': [0.6, 0.8], 'color': "orange"},
                {'range': [0.8, 1], 'color': "red"}
            ],
        }
    ))
    gauge_fig.update_layout(
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        font={'color': "white", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return gauge_fig, audio_level

@app.callback(
    Output("timestamp", "children"),
    Input("interval-video", "n_intervals")
)
def update_timestamp(n):
    return f"Last Update: {time.strftime('%Y-%m-%d %H:%M:%S')}"

@app.callback(
    Output("video-source-dropdown", "options"),
    Input("refresh-sources", "n_clicks")
)
def refresh_sources(n_clicks):
    if n_clicks is None:
        return dash.no_update
    return get_video_sources()

# Main execution
if __name__ == "__main__":
    # Cleanup on exit
    import atexit
    atexit.register(cleanup_resources)
    
    # Initialize default video source
    if DEFAULT_SOURCE:
        switch_video_source(DEFAULT_SOURCE)
    
    app.run(host="0.0.0.0", port=8050, debug=False)