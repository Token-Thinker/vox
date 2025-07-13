# Vox Dashboard

A simple web-based dashboard for streaming camera and audio data using Dash.

## Dependencies

All Python dependencies are listed in `requirements.txt`. To install them, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: The Picamera2 library is installed from the official Raspberry Pi GitHub repository at version `0_3_12` as specified in `requirements.txt`.

## Usage

### Start in background

```bash
chmod +x start.sh
./start.sh
```

The script sets `FLASK_APP=main:server` and runs `flask run` in the background,
redirecting logs to `vox.log` and writing the process ID to `vox.pid`.

### Run in foreground

```bash
export FLASK_APP=main:server
flask run --host=0.0.0.0 --port=8050
```

Then open your web browser to http://127.0.0.1:8050 to view the executive anomaly dashboard.

## Model files

Place your TFLite models and metadata under the following paths for offline inference:

- `lib/sound_detector/`: `encoder.tflite`, `classifier.model`, `_min.npy`, `_max.npy`
- `lib/image_anomaly/models/`: `encoder.tflite` (and optional classifier/model files)

## Project Structure

- `main.py`: Dash application entry point (camera & audio anomaly UI).
- `start.sh`: Helper script to start the dashboard in the background.
- `requirements.txt`: Python dependencies.
- `lib/sound_detector/`: Audio anomaly detection logic (TFLite encoder + classifier).
- `lib/image_anomaly/`: Image/scene anomaly detection logic (TFLite encoder + detector).
- Picamera2 library is installed from the official Raspberry Pi GitHub repository (version `0_3_12`) via `requirements.txt`.

## Visualization Examples

Below are minimal Plotly code snippets illustrating common visualization types:

### Waveform

**Line Plot**
```python
import numpy as np
import plotly.graph_objects as go

# Simulate a simple waveform
t = np.linspace(0, 1, 500)
wave = np.sin(2 * np.pi * 5 * t)
fig = go.Figure(go.Scatter(x=t, y=wave, mode='lines', name='Waveform (Line)'))
fig.update_layout(title='Waveform (Line)', xaxis_title='Time [s]', yaxis_title='Amplitude')
fig.show()
```

**Scatter Plot**
```python
fig = go.Figure(go.Scatter(x=t, y=wave, mode='markers', name='Waveform (Scatter)'))
fig.update_layout(title='Waveform (Scatter)', xaxis_title='Time [s]', yaxis_title='Amplitude')
fig.show()
```

### Spectrogram

**Heatmap**
```python
import librosa

# Compute a magnitude spectrogram
S = np.abs(librosa.stft(wave, n_fft=256))
fig = go.Figure(go.Heatmap(z=20 * np.log10(S + 1e-6), x=np.arange(S.shape[1]), y=np.arange(S.shape[0])))
fig.update_layout(title='Spectrogram (Heatmap)', xaxis_title='Frame', yaxis_title='Frequency Bin')
fig.show()
```

**Surface Plot**
```python
fig = go.Figure(go.Surface(z=20 * np.log10(S + 1e-6), x=np.arange(S.shape[1]), y=np.arange(S.shape[0])))
fig.update_layout(title='Spectrogram (Surface)', scene=dict(xaxis_title='Frame', yaxis_title='Freq Bin', zaxis_title='dB'))
fig.show()
```

### Object Trajectories

**2D Trajectory (Scatter)**
```python
num_points = 100
x = np.cumsum(np.random.randn(num_points))
y = np.cumsum(np.random.randn(num_points))
fig = go.Figure(go.Scatter(x=x, y=y, mode='lines+markers', name='Trajectory 2D'))
fig.update_layout(title='Object Trajectory (2D)', xaxis_title='X', yaxis_title='Y')
fig.show()
```

**3D Trajectory (Scatter3d)**
```python
z = np.cumsum(np.random.randn(num_points))
fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', name='Trajectory 3D'))
fig.update_layout(title='Object Trajectory (3D)', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig.show()
```

### Object Interactions

**Heatmap**
```python
adj = np.random.randint(0, 10, size=(10, 10))
fig = go.Figure(go.Heatmap(z=adj, x=list(range(adj.shape[1])), y=list(range(adj.shape[0]))))
fig.update_layout(title='Object Interaction Heatmap', xaxis_title='Object', yaxis_title='Object')
fig.show()
```

**Network Graph**
```python
import networkx as nx

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

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(title='Object Interaction Network', xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))
fig.show()
```

### Reconstruction Error

**Line Plot**
```python
errors = np.random.rand(100)
fig = go.Figure(go.Scatter(x=np.arange(len(errors)), y=errors, mode='lines', name='Reconstruction Error'))
fig.update_layout(title='Reconstruction Error (Line)', xaxis_title='Sample', yaxis_title='Error')
fig.show()
```

**Scatter Plot**
```python
fig = go.Figure(go.Scatter(x=np.arange(len(errors)), y=errors, mode='markers', name='Reconstruction Error'))
fig.update_layout(title='Reconstruction Error (Scatter)', xaxis_title='Sample', yaxis_title='Error')
fig.show()
```

### Feature Space Visualization

**2D Feature Space (Scatter)**
```python
features = np.random.randn(100, 2)
fig = go.Figure(go.Scatter(x=features[:, 0], y=features[:, 1], mode='markers', name='Features 2D'))
fig.update_layout(title='Feature Space (2D)', xaxis_title='Feature 1', yaxis_title='Feature 2')
fig.show()
```

**3D Feature Space (Scatter3d)**
```python
features3 = np.random.randn(100, 3)
fig = go.Figure(go.Scatter3d(x=features3[:, 0], y=features3[:, 1], z=features3[:, 2], mode='markers', name='Features 3D'))
fig.update_layout(title='Feature Space (3D)', scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='F3'))
fig.show()
```

## Running Tests

There is a basic smoke-test script that runs over 100 scenarios to verify the visualizations and callbacks:

```bash
python3 tests/run_tests.py
```
