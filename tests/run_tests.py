#!/usr/bin/env python3
"""
Basic smoke tests for the Vox dashboard example visualizations.
Runs >100 simple scenarios to verify static figures and callbacks execute without error.
"""
import os
import sys
# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import main

# Monkey-patch audio recording and detectors for testing
main.sd.rec = lambda count, samplerate, channels, dtype: np.zeros((count, channels), dtype=dtype)
main.sd.wait = lambda: None
main.sound_detector.predict = lambda x: 0.5

# Build scenarios: 12 static figures * 9 iterations = 108, + 1 video fallback, + 10 audio updates
static_figs = [
    main.wave_line_fig, main.wave_scatter_fig,
    main.spec_heat_fig, main.spec_surf_fig,
    main.traj2d_fig, main.traj3d_fig,
    main.inter_heat_fig, main.inter_net_fig,
    main.recon_line_fig, main.recon_scatter_fig,
    main.feat2_fig, main.feat3_fig,
]
scenarios = []
for fig in static_figs * 9:
    scenarios.append(('static_fig', fig))
scenarios.append(('update_video', 0))
for n in range(10):
    scenarios.append(('update_audio', n))

failures = 0
for idx, (kind, val) in enumerate(scenarios, start=1):
    try:
        if kind == 'static_fig':
            fig = val
            assert hasattr(fig, 'data') and len(fig.data) > 0
        elif kind == 'update_video':
            src, fig = main.update_video(val)
            assert src == ''
            title = getattr(fig.layout.title, 'text', '')
            assert 'Camera unavailable' in title
        elif kind == 'update_audio':
            fig = main.update_audio(val)
            assert hasattr(fig, 'data') and fig.data[0].type == 'indicator'
        else:
            raise AssertionError(f'Unknown scenario: {kind}')
    except AssertionError as e:
        print(f'Test {idx} ({kind}, {val}) FAILED: {e}')
        failures += 1

if failures:
    print(f"{failures}/{len(scenarios)} tests failed.")
    sys.exit(1)
else:
    print(f"All {len(scenarios)} tests passed.")
    sys.exit(0)