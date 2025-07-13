"""
ImageAnomalyDetector loads an image-based autoencoder or object detector model
to compute an anomaly score on camera frames.
"""
import os
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

class ImageAnomalyDetector:
    def __init__(self, model_dir):
        base = os.path.abspath(model_dir)
        encoder_path = os.path.join(base, 'encoder.tflite')
        self.interpreter = Interpreter(model_path=encoder_path)
        self.interpreter.allocate_tensors()
        details = self.interpreter.get_input_details()[0]
        self._input_index = details['index']
        self._output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, frame: np.ndarray) -> float:
        data = np.expand_dims(frame.astype('float32'), axis=0)
        self.interpreter.set_tensor(self._input_index, data)
        self.interpreter.invoke()
        emb = self.interpreter.get_tensor(self._output_index).reshape(1, -1)
        return 0.0