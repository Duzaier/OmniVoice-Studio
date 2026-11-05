"""
experiments/v2_20/sanity_checker.py

Isolated Reference Audio Sanity Check Component for OmniVoice V2.20.
Implements multi-criteria audio quality filtering without modifying core baseline.
"""

import math
import torch
import torchaudio


class ReferenceAudioSanityChecker:
    def __init__(
        self,
        min_duration_sec: float = 2.5,
        max_duration_sec: float = 15.0,
        min_rms: float = 0.015,
        max_rms: float = 0.35,
        max_clipping_ratio: float = 0.002,  # 0.2%
        min_snr_db: float = 14.0,
        max_silence_ratio: float = 0.30,  # 30%
        target_sample_rate: int = 24000,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_rms = min_rms
        self.max_rms = max_rms
        self.max_clipping_ratio = max_clipping_ratio
        self.min_snr_db = min_snr_db
        self.max_silence_ratio = max_silence_ratio
        self.target_sample_rate = target_sample_rate

    def evaluate(self, waveform: torch.Tensor, sample_rate: int) -> dict:
        """
        Evaluate a single-channel or multi-channel waveform tensor.
        waveform: [channels, samples] or [samples]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert to mono if multi-channel
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        num_samples = waveform.size(1)
        duration_sec = num_samples / float(sample_rate)
        
        # 1. Duration Check
        if duration_sec < self.min_duration_sec:
            return {
                "accepted": False,
                "reason": f"Duration too short ({duration_sec:.2f}s < {self.min_duration_sec}s)",
                "duration_sec": duration_sec,
                "metrics": {}
            }
        if duration_sec > self.max_duration_sec:
            return {
                "accepted": False,
                "reason": f"Duration too long ({duration_sec:.2f}s > {self.max_duration_sec}s)",
                "duration_sec": duration_sec,
                "metrics": {}
            }

        # 2. RMS / Energy Level
        rms = torch.sqrt(torch.mean(waveform ** 2)).item()
        if rms < self.min_rms:
            return {
                "accepted": False,
                "reason": f"RMS energy too low ({rms:.4f} < {self.min_rms})",
                "duration_sec": duration_sec,
                "metrics": {"rms": rms}
            }
        if rms > self.max_rms:
            return {
                "accepted": False,
                "reason": f"RMS energy too high / distorted ({rms:.4f} > {self.max_rms})",
                "duration_sec": duration_sec,
                "metrics": {"rms": rms}
            }

        # 3. Clipping Detection
        clipping_count = (waveform.abs() >= 0.999).sum().item()
        clipping_ratio = clipping_count / float(num_samples)
        if clipping_ratio > self.max_clipping_ratio:
            return {
                "accepted": False,
                "reason": f"Clipping ratio exceeded ({clipping_ratio*100:.2f}% > {self.max_clipping_ratio*100:.2f}%)",
                "duration_sec": duration_sec,
                "metrics": {"rms": rms, "clipping_ratio": clipping_ratio}
            }

        # 4. Silence Ratio & Estimated SNR via frame-level energy
        frame_len = int(sample_rate * 0.025)  # 25ms
        hop_len = int(sample_rate * 0.010)    # 10ms
        if num_samples >= frame_len:
            frames = waveform.unfold(1, frame_len, hop_len)
            frame_energy = (frames ** 2).mean(dim=-1).squeeze(0)
            threshold = (rms ** 2) * 0.01  # Energy threshold for speech vs silence
            silence_frames = (frame_energy < threshold).sum().item()
            total_frames = frame_energy.size(0)
            silence_ratio = silence_frames / float(total_frames) if total_frames > 0 else 0.0
            
            speech_energy = frame_energy[frame_energy >= threshold].mean().item() if (frame_energy >= threshold).any() else 1e-6
            noise_energy = frame_energy[frame_energy < threshold].mean().item() if (frame_energy < threshold).any() else 1e-6
            snr_db = 10.0 * math.log10(max(speech_energy / max(noise_energy, 1e-9), 1.0))
        else:
            silence_ratio = 0.0
            snr_db = 20.0

        if silence_ratio > self.max_silence_ratio:
            return {
                "accepted": False,
                "reason": f"Silence ratio too high ({silence_ratio*100:.1f}% > {self.max_silence_ratio*100:.1f}%)",
                "duration_sec": duration_sec,
                "metrics": {"rms": rms, "clipping_ratio": clipping_ratio, "silence_ratio": silence_ratio, "snr_db": snr_db}
            }

        if snr_db < self.min_snr_db:
            return {
                "accepted": False,
                "reason": f"Estimated SNR too low ({snr_db:.1f} dB < {self.min_snr_db} dB)",
                "duration_sec": duration_sec,
                "metrics": {"rms": rms, "clipping_ratio": clipping_ratio, "silence_ratio": silence_ratio, "snr_db": snr_db}
            }

        return {
            "accepted": True,
            "reason": "Passed all quality gates",
            "duration_sec": duration_sec,
            "metrics": {
                "rms": round(rms, 4),
                "clipping_ratio": round(clipping_ratio, 5),
                "silence_ratio": round(silence_ratio, 3),
                "snr_db": round(snr_db, 1)
            }
        }
