"""
Spectral cleaner for removing frequency-domain watermarks
"""

import logging
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from typing import Dict, List, Any
import librosa

logger = logging.getLogger(__name__)


class SpectralCleaner:
    """
    Removes watermarks and anomalies from the frequency domain
    """

    def __init__(self, paranoid_mode: bool = False):
        self.paranoid_mode = paranoid_mode
        # Keep analysis bounded so untrusted/large inputs cannot trigger
        # unreasonably long verification or filter loops.
        self.max_notch_filters = 20 if paranoid_mode else 12
        self.max_verification_samples = 131072
        self.max_autocorr_samples = 16384
        self.watermark_freq_bands = [
            (18000, 18500),  # Known AI watermark ranges
            (19000, 19500),
            (20000, 20500),
            (21000, 21500),
        ]
        self.suspicious_patterns = [
            "periodic_peaks",
            "constant_frequencies",
            "unnatural_harmonics",
            "synchronization_tones",
        ]

    def clean_watermarks(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Remove spectral watermarks from audio

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate

        Returns:
            Dict containing cleaning results
        """
        result = {
            "cleaned_audio": audio_data.copy(),
            "watermarks_found": 0,
            "watermarks_removed": 0,
            "methods_used": [],
            "details": [],
        }

        # Ensure stereo handling
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=1)

        cleaned_channels = []

        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]
            cleaned_channel = channel_data.copy()

            # Method 1: High-frequency watermark removal
            watermark_result = self._remove_high_frequency_watermarks(
                cleaned_channel, sample_rate
            )
            cleaned_channel = watermark_result["cleaned_data"]
            result["watermarks_found"] += watermark_result["found"]
            result["watermarks_removed"] += watermark_result["removed"]
            result["methods_used"].append("high_freq_filter")

            # Method 2: Periodic pattern disruption
            pattern_result = self._disrupt_periodic_patterns(
                cleaned_channel, sample_rate
            )
            cleaned_channel = pattern_result["cleaned_data"]
            result["methods_used"].append("pattern_disruption")

            # Method 3: Spectral smoothing
            if self.paranoid_mode:
                smooth_result = self._spectral_smoothing(cleaned_channel, sample_rate)
                cleaned_channel = smooth_result["cleaned_data"]
                result["methods_used"].append("spectral_smoothing")

            # Method 4: Adaptive noise shaping
            if self.paranoid_mode:
                noise_result = self._adaptive_noise_shaping(
                    cleaned_channel, sample_rate
                )
                cleaned_channel = noise_result["cleaned_data"]
                result["methods_used"].append("noise_shaping")

            cleaned_channels.append(cleaned_channel)

        # Reconstruct multi-channel audio
        if len(cleaned_channels) == 1:
            result["cleaned_audio"] = cleaned_channels[0]
        else:
            result["cleaned_audio"] = np.column_stack(cleaned_channels)

        # Add verification details
        result["details"] = self._verify_cleaning(
            audio_data, result["cleaned_audio"], sample_rate
        )

        return result

    def _remove_high_frequency_watermarks(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Remove watermarks in high frequency ranges"""
        result = {
            "cleaned_data": audio_data.copy(),
            "found": 0,
            "removed": 0,
            "frequencies_cleaned": [],
        }

        # Perform FFT
        fft_data = fft(audio_data)
        freqs = fftfreq(len(audio_data), 1 / sample_rate)

        # Work on a copy to satisfy static analyzers
        fft_mod = fft_data.copy()

        # Scan for suspicious high frequency content
        for freq_range in self.watermark_freq_bands:
            freq_min, freq_max = freq_range

            if freq_max > sample_rate / 2:
                continue  # Skip frequencies above Nyquist

            # Find frequencies in range
            freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) < freq_max)
            freq_power = np.abs(fft_mod[freq_mask])

            if len(freq_power) > 0:
                avg_power = np.mean(freq_power)
                noise_floor = np.median(np.abs(fft_mod))

                # Detect watermark if power is significantly above noise floor
                if avg_power > noise_floor * 5:
                    result["found"] += 1

                    # Remove or attenuate suspicious frequencies
                    attenuation_factor = 0.1  # Reduce by 90%
                    fft_mod[freq_mask] = fft_mod[freq_mask] * attenuation_factor

                    result["removed"] += 1
                    result["frequencies_cleaned"].append(freq_range)

        # Convert back to time domain
        result["cleaned_data"] = np.real(ifft(fft_mod))

        return result

    def _disrupt_periodic_patterns(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Disrupt periodic spectral patterns that indicate watermarks"""
        result = {"cleaned_data": audio_data.copy(), "patterns_disrupted": []}

        # Compute spectrogram
        nperseg = min(2048, len(audio_data) // 8)
        if nperseg < 256:
            nperseg = 256
        f, t, Sxx = signal.spectrogram(audio_data, fs=sample_rate, nperseg=nperseg)

        # Look for periodic patterns in frequency domain
        spectral_mean = np.mean(Sxx, axis=1)

        # Find peaks in spectral content
        peaks, properties = signal.find_peaks(
            spectral_mean, height=np.max(spectral_mean) * 0.1
        )

        # Check for suspicious periodicity in peaks
        if len(peaks) > 2:
            peak_spacing = np.diff(peaks)
            spacing_consistency = 1.0 - (
                np.std(peak_spacing) / (np.mean(peak_spacing) + 1e-10)
            )

            if spacing_consistency > 0.8:  # Suspiciously consistent
                result["patterns_disrupted"].append(
                    {
                        "type": "spectral_periodicity",
                        "consistency": spacing_consistency,
                        "peak_count": len(peaks),
                    }
                )

                # Apply subtle randomization to disrupt pattern
                noise_level = 1e-5
                phase_noise = np.random.normal(0, noise_level, len(audio_data))
                result["cleaned_data"] += phase_noise

        # Bound notch filter passes to avoid pathological runtime on dense spectra
        if len(peaks) > self.max_notch_filters:
            peak_heights = properties.get("peak_heights")
            if peak_heights is not None and len(peak_heights) == len(peaks):
                strongest = np.argpartition(
                    peak_heights, -self.max_notch_filters
                )[-self.max_notch_filters :]
                peaks = np.sort(peaks[strongest])
            else:
                peaks = peaks[: self.max_notch_filters]

        # Apply notch filters to suspicious frequencies
        for peak_idx in peaks:
            freq = f[peak_idx]

            # Skip fundamental frequencies (< 100 Hz) and very high frequencies
            if 100 <= freq <= sample_rate / 2 - 100:
                # Apply narrow notch filter
                result["cleaned_data"] = self._apply_notch_filter(
                    result["cleaned_data"], freq, sample_rate, q=30
                )

        return result

    def _apply_notch_filter(
        self, data: np.ndarray, freq: float, sample_rate: int, q: float = 30
    ) -> np.ndarray:
        """Apply notch filter to remove specific frequency"""
        from scipy.signal import iirnotch, filtfilt

        # Design notch filter
        nyquist = sample_rate / 2
        normalized_freq = freq / nyquist

        # Ensure frequency is within valid range
        if normalized_freq >= 0.99:
            return data

        try:
            b, a = iirnotch(normalized_freq, q)
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        except Exception as e:
            logger.warning("Notch filter at %.0f Hz failed: %s", freq, e)
            return data

    def _spectral_smoothing(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Apply spectral smoothing to hide watermark patterns"""
        from scipy.ndimage import gaussian_filter

        result = {"cleaned_data": audio_data.copy(), "smoothing_applied": True}

        # Short-time Fourier Transform
        nperseg = min(2048, len(audio_data) // 8)
        if nperseg < 256:
            nperseg = 256
        stft = librosa.stft(audio_data, n_fft=nperseg)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Gaussian smoothing (sigma=1.0 matches the original per-element
        # Gaussian weight kernel with window_size=5)
        smoothed_magnitude = gaussian_filter(magnitude, sigma=1.0)

        # Reconstruct signal
        smoothed_stft = smoothed_magnitude * np.exp(1j * phase)
        result["cleaned_data"] = librosa.istft(smoothed_stft, length=len(audio_data))

        return result

    def _adaptive_noise_shaping(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Apply adaptive noise shaping to mask watermarks"""
        result = {"cleaned_data": audio_data.copy(), "noise_shaping_applied": True}

        # Analyze spectral content
        fft_data = fft(audio_data)
        freqs = fftfreq(len(audio_data), 1 / sample_rate)

        # Calculate spectral envelope
        magnitude = np.abs(fft_data)
        log_magnitude = np.log10(magnitude + 1e-10)

        # Smooth spectral envelope to detect anomalies
        envelope = signal.savgol_filter(log_magnitude, window_length=51, polyorder=3)

        # Identify spectral anomalies
        residuals = log_magnitude - envelope
        anomaly_threshold = np.std(residuals) * 2
        anomalies = np.abs(residuals) > anomaly_threshold

        # Add adaptive noise to mask anomalies in frequency domain
        noise_level = 1e-6
        noise_fft = np.random.normal(0, noise_level, len(fft_data)) + 1j * np.random.normal(0, noise_level, len(fft_data))

        # Increase noise at anomaly frequencies
        anomaly_freq_indices = np.where(anomalies)[0]
        if len(anomaly_freq_indices) > 0:
            noise_fft[anomaly_freq_indices] *= 3

        modified_fft = fft_data + noise_fft

        # Convert back to time domain
        result["cleaned_data"] = np.real(ifft(modified_fft))

        return result

    def _extract_verification_window(
        self, data: np.ndarray, max_samples: int
    ) -> np.ndarray:
        """Return a centered window so verification cost stays bounded."""
        if data.shape[0] <= max_samples:
            return data

        start = (data.shape[0] - max_samples) // 2
        return data[start : start + max_samples]

    def _to_mono_series(self, data: np.ndarray) -> np.ndarray:
        """Collapse channels to a single representative series."""
        if data.ndim == 1:
            return data
        return np.mean(data, axis=1)

    def _verify_cleaning(
        self, original: np.ndarray, cleaned: np.ndarray, sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Verify that cleaning was effective"""
        verification = []

        if original.size == 0 or cleaned.size == 0:
            return verification

        original_eval = self._extract_verification_window(
            original, self.max_verification_samples
        )
        cleaned_eval = self._extract_verification_window(
            cleaned, self.max_verification_samples
        )

        # Compute spectral comparison
        orig_fft = np.abs(np.fft.rfft(original_eval, axis=0))
        clean_fft = np.abs(np.fft.rfft(cleaned_eval, axis=0))

        # Check high frequency reduction
        freqs = np.fft.rfftfreq(len(original_eval), 1 / sample_rate)
        high_freq_mask = freqs > 15000

        if np.any(high_freq_mask):
            orig_hf_power = np.mean(orig_fft[high_freq_mask])
            clean_hf_power = np.mean(clean_fft[high_freq_mask])
            if orig_hf_power > 0 and clean_hf_power < orig_hf_power * 0.5:
                verification.append(
                    {
                        "metric": "high_frequency_reduction",
                        "original_power": float(orig_hf_power),
                        "cleaned_power": float(clean_hf_power),
                        "reduction_percentage": float(
                            (1 - clean_hf_power / orig_hf_power) * 100
                        ),
                    }
                )

        # Check for pattern disruption
        orig_series = self._extract_verification_window(
            self._to_mono_series(original_eval), self.max_autocorr_samples
        )
        clean_series = self._extract_verification_window(
            self._to_mono_series(cleaned_eval), self.max_autocorr_samples
        )

        orig_series = (orig_series - np.mean(orig_series)).astype(np.float32, copy=False)
        clean_series = (clean_series - np.mean(clean_series)).astype(np.float32, copy=False)

        orig_autocorr = signal.correlate(
            orig_series, orig_series, mode="same", method="fft"
        )
        clean_autocorr = signal.correlate(
            clean_series, clean_series, mode="same", method="fft"
        )

        orig_height = np.max(orig_autocorr) * 0.8 if orig_autocorr.size else 0.0
        clean_height = np.max(clean_autocorr) * 0.8 if clean_autocorr.size else 0.0

        orig_peaks = len(signal.find_peaks(orig_autocorr, height=orig_height)[0])
        clean_peaks = len(
            signal.find_peaks(clean_autocorr, height=clean_height)[0]
        )

        if clean_peaks < orig_peaks:
            verification.append(
                {
                    "metric": "pattern_disruption",
                    "original_peaks": orig_peaks,
                    "cleaned_peaks": clean_peaks,
                    "reduction": orig_peaks - clean_peaks,
                }
            )

        return verification

    def remove_synchronization_tones(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Remove synchronization tones often used in watermarking"""
        cleaned_data = audio_data.copy()

        # Common synchronization tone frequencies
        sync_freqs = [1000, 2000, 3000, 4000, 5000, 10000, 15000]

        for sync_freq in sync_freqs:
            if sync_freq < sample_rate / 2:
                # Apply very narrow notch filter
                cleaned_data = self._apply_notch_filter(
                    cleaned_data, sync_freq, sample_rate, q=50
                )

        return cleaned_data

    def spread_spectrum_watermark_removal(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Advanced spread spectrum watermark removal"""
        cleaned_data = audio_data.copy()
        original_length = len(cleaned_data)

        # Multiple window sizes for comprehensive analysis
        window_sizes = [512, 1024, 2048, 4096]

        for window_size in window_sizes:
            # Compute STFT
            f, t, Zxx = signal.stft(cleaned_data, fs=sample_rate, nperseg=window_size)
            magnitude = np.abs(Zxx)

            # Detect and suppress suspicious patterns
            for time_idx in range(magnitude.shape[1]):
                spectrum = magnitude[:, time_idx]

                # Find unusual peaks that could be watermark carriers
                median_level = np.median(spectrum)
                peak_threshold = median_level * 5

                peaks = signal.find_peaks(spectrum, height=peak_threshold)[0]

                for peak_idx in peaks:
                    # Attenuate suspicious peaks
                    attenuation = 0.2
                    Zxx[peak_idx, time_idx] *= attenuation

            # Reconstruct signal and preserve original length
            _, cleaned_data = signal.istft(Zxx, fs=sample_rate, nperseg=window_size)
            cleaned_data = cleaned_data[:original_length]

        return cleaned_data
