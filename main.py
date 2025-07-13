#!/usr/bin/env python3
import dash
from dash import html, dcc
from dash.dependencies import Input, Output

import io
import base64
import numpy as np
import sounddevice as sd
from PIL import Image
import plotly.graph_objects as go
import librosa
import networkx as nx

# Camera fallback if libcamera/Picamera2 not available
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

app = dash.Dash(__name__)
server = app.server
app.title = "Vox Executive Anomaly Dashboard"

# Initialize sound/image detectors (fallback to dummy if models missing)
class _DummyDetector:
    def __init__(self):
        self.sample_rate = 44100
    def predict(self, *args, **kwargs):
        return 0.0

# Dummy default detectors and camera (overridden under __main__)
sound_detector = _DummyDetector()
image_detector = _DummyDetector()
camera = None

# Initialize camera if available (will be restarted under __main__)
if CAMERA_AVAILABLE:
    camera = None  # placeholder; real camera init happens under __main__

# --- Dummy data examples for static visualizations ---
# Waveform
t = np.linspace(0, 1, 500)
wave = np.sin(2 * np.pi * 5 * t)
wave_line_fig = go.Figure(go.Scatter(x=t, y=wave, mode='lines', name='Waveform (Line)'))
wave_line_fig.update_layout(title='Waveform (Line)', xaxis_title='Time [s]', yaxis_title='Amplitude')
wave_scatter_fig = go.Figure(go.Scatter(x=t, y=wave, mode='markers', name='Waveform (Scatter)'))
wave_scatter_fig.update_layout(title='Waveform (Scatter)', xaxis_title='Time [s]', yaxis_title='Amplitude')

# Spectrogram
S = np.abs(librosa.stft(wave, n_fft=256))
spec_heat_fig = go.Figure(go.Heatmap(z=20 * np.log10(S + 1e-6), x=np.arange(S.shape[1]), y=np.arange(S.shape[0])))
spec_heat_fig.update_layout(title='Spectrogram (Heatmap)', xaxis_title='Frame', yaxis_title='Frequency Bin')
spec_surf_fig = go.Figure(go.Surface(z=20 * np.log10(S + 1e-6), x=np.arange(S.shape[1]), y=np.arange(S.shape[0])))
spec_surf_fig.update_layout(title='Spectrogram (Surface)', scene=dict(xaxis_title='Frame', yaxis_title='Freq Bin', zaxis_title='dB'))

# Object trajectories
num_points = 100
x = np.cumsum(np.random.randn(num_points))
y = np.cumsum(np.random.randn(num_points))
traj2d_fig = go.Figure(go.Scatter(x=x, y=y, mode='lines+markers', name='Trajectory 2D'))
traj2d_fig.update_layout(title='Object Trajectory (2D)', xaxis_title='X', yaxis_title='Y')
z = np.cumsum(np.random.randn(num_points))
traj3d_fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', name='Trajectory 3D'))
traj3d_fig.update_layout(title='Object Trajectory (3D)', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Object interactions
adj = np.random.randint(0, 10, size=(10, 10))
inter_heat_fig = go.Figure(go.Heatmap(z=adj, x=list(range(adj.shape[1])), y=list(range(adj.shape[0]))))
inter_heat_fig.update_layout(title='Object Interaction Heatmap', xaxis_title='Object', yaxis_title='Object')
G = nx.erdos_renyi_graph(n=10, p=0.3)
pos = nx.spring_layout(G)
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray'), hoverinfo='none')
node_x = [pos[n][0] for n in G.nodes()]
node_y = [pos[n][1] for n in G.nodes()]
node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='blue'), text=[str(n) for n in G.nodes()], hoverinfo='text')
inter_net_fig = go.Figure(data=[edge_trace, node_trace])
inter_net_fig.update_layout(title='Object Interaction Network', xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))

# Reconstruction error
errors = np.random.rand(100)
recon_line_fig = go.Figure(go.Scatter(x=np.arange(len(errors)), y=errors, mode='lines', name='Reconstruction Error'))
recon_line_fig.update_layout(title='Reconstruction Error (Line)', xaxis_title='Sample', yaxis_title='Error')
recon_scatter_fig = go.Figure(go.Scatter(x=np.arange(len(errors)), y=errors, mode='markers', name='Reconstruction Error'))
recon_scatter_fig.update_layout(title='Reconstruction Error (Scatter)', xaxis_title='Sample', yaxis_title='Error')

# Feature space visualization
features = np.random.randn(100, 2)
feat2_fig = go.Figure(go.Scatter(x=features[:, 0], y=features[:, 1], mode='markers', name='Features 2D'))
feat2_fig.update_layout(title='Feature Space (2D)', xaxis_title='Feature 1', yaxis_title='Feature 2')
features3 = np.random.randn(100, 3)
feat3_fig = go.Figure(go.Scatter3d(x=features3[:, 0], y=features3[:, 1], z=features3[:, 2], mode='markers', name='Features 3D'))
feat3_fig.update_layout(title='Feature Space (3D)', scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='F3'))

app.layout = html.Div([
    html.H1("Vox Executive Anomaly Dashboard"),
    html.Div([
        (html.Div([
            html.H2("Camera Stream"),
            html.Img(id="video-frame", style={"width": "640px", "height": "480px", "border": "1px solid #ccc"}),
            dcc.Graph(id="video-anom-gauge", config={"displayModeBar": False}),
            dcc.Interval(id="interval-video", interval=1000, n_intervals=0),
        ], style={"display": "inline-block", "marginRight": "20px", "verticalAlign": "top"})
         if CAMERA_AVAILABLE else html.Div([
            html.H2("Camera Stream"), html.Div("Camera unavailable", style={"color": "red"})
        ], style={"display": "inline-block", "marginRight": "20px", "verticalAlign": "top"})),
        html.Div([
            html.H2("Audio Anomaly Score"),
            dcc.Graph(id="audio-anom-gauge", config={"displayModeBar": False}),
            dcc.Interval(id="interval-audio", interval=10000, n_intervals=0),
        ], style={"display": "inline-block", "verticalAlign": "top"}),
    ]),
    html.H2("Visualization Examples"),
    html.Div([
        dcc.Graph(figure=wave_line_fig),
        dcc.Graph(figure=wave_scatter_fig),
        dcc.Graph(figure=spec_heat_fig),
        dcc.Graph(figure=spec_surf_fig),
        dcc.Graph(figure=traj2d_fig),
        dcc.Graph(figure=traj3d_fig),
        dcc.Graph(figure=inter_heat_fig),
        dcc.Graph(figure=inter_net_fig),
        dcc.Graph(figure=recon_line_fig),
        dcc.Graph(figure=recon_scatter_fig),
        dcc.Graph(figure=feat2_fig),
        dcc.Graph(figure=feat3_fig),
    ])
])

@app.callback(
    Output("video-frame", "src"),
    Output("video-anom-gauge", "figure"),
    Input("interval-video", "n_intervals")
)
def update_video(n):
    if not CAMERA_AVAILABLE:
        empty_fig = go.Figure()
        empty_fig.update_layout(title='Camera unavailable', margin={"l":20, "r":20, "t":20, "b":20})
        return "", empty_fig
    frame = camera.capture_array()
    score = image_detector.predict(frame)
    pil = Image.fromarray(frame)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    src = f"data:image/jpeg;base64,{img_b64}"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={"axis": {"range": [0, 1]}},
    ))
    fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
    return src, fig

@app.callback(
    Output("audio-anom-gauge", "figure"),
    Input("interval-audio", "n_intervals")
)
def update_audio(n):
    duration = 10.0
    fs = sound_detector.sample_rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    score = sound_detector.predict(recording)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={"axis": {"range": [0, 1]}},
    ))
    fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})
    return fig

if __name__ == "__main__":
    # Real-model initialization in script mode
    if CAMERA_AVAILABLE:
        camera = Picamera2()
        camera.start()

    try:
        from lib.sound_detector.predictor import SoundAnomalyDetector
        sound_detector = SoundAnomalyDetector("lib/sound_detector")
    except Exception:
        sound_detector = _DummyDetector()

    try:
        from lib.image_anomaly.predictor import ImageAnomalyDetector
        image_detector = ImageAnomalyDetector("lib/image_anomaly/models")
    except Exception:
        image_detector = _DummyDetector()

    app.run_server(host="0.0.0.0", port=8050, debug=False)
