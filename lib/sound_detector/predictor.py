"""
SoundAnomalyDetector loads a TFLite encoder and a classifier to detect
anomalies in 10s audio clips using a Mel-spectrogram autoencoder approach.
"""
import os
import pickle
import numpy as np
import librosa
import tensorflow as tf

class SoundAnomalyDetector:
    def __init__(self, model_dir, sample_rate=44100):
        self.sample_rate = sample_rate
        base = os.path.abspath(model_dir)
        self._min = np.load(os.path.join(base, '_min.npy'))
        self._max = np.load(os.path.join(base, '_max.npy'))
        self.encoder = tf.lite.Interpreter(model_path=os.path.join(base, 'encoder.tflite'))
        self.encoder.allocate_tensors()
        details = self.encoder.get_input_details()[0]
        self._input_index = details['index']
        self._output_index = self.encoder.get_output_details()[0]['index']
        with open(os.path.join(base, 'classifier.model'), 'rb') as f:
            self.detector = pickle.load(f)

    def predict(self, signal: np.ndarray) -> float:
        mel = librosa.feature.melspectrogram(
            y=signal.flatten(), sr=self.sample_rate,
            n_fft=1024, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        data = mel_db.reshape(1, *mel_db.shape, 1).astype('float32')
        data = np.pad(data, ((0, 0), (0, 0), (0, 7), (0, 0)), mode='constant')
        data = (data - self._min) / (self._max - self._min)
        self.encoder.set_tensor(self._input_index, data)
        self.encoder.invoke()
        emb = self.encoder.get_tensor(self._output_index).reshape(1, -1)
        y = self.detector.predict(emb)
        return float(y[0])

if __name__ == '__main__':
    import sys
    import soundfile as sf

    if len(sys.argv) != 2:
        print('Usage: predictor.py <wavfile>')
        sys.exit(1)
    wavfile = sys.argv[1]
    signal, sr = sf.read(wavfile)
    det = SoundAnomalyDetector(os.path.dirname(__file__))
    result = det.predict(signal)
    print('normal' if result == 1.0 else 'abnormal')
