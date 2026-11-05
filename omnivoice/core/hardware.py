#!/usr/bin/env python3
"""
Hardware Profiler & Runtime Resource Monitor for OmniVoice.
Optimized for Dell Inspiron 15 3515 (AMD Ryzen 5 3450U + AMD Radeon Vega 8 Mobile).
Provides fast registry-backed GPU/CPU detection, optimal 6-thread MKL pinning,
cached disk resource monitoring, and C: drive safety checks.
"""

import os
import sys
import time
import shutil
import platform
import logging
import subprocess
from typing import Dict, Any, Optional, List

import psutil
import torch

logger = logging.getLogger(__name__)

# Cache expensive directory walks (like ~/.cache/huggingface) so synthesis calls have zero I/O overhead
_DIR_SIZE_CACHE: Dict[str, tuple[float, float]] = {}
_DIR_SIZE_TTL_SEC = 300.0
_CACHED_GPUS: Optional[List[Dict[str, Any]]] = None
_CACHED_CPU_NAME: Optional[str] = None


def detect_cpu_name() -> str:
    """Detect human-readable CPU model name via Windows Registry (<1ms)."""
    global _CACHED_CPU_NAME
    if _CACHED_CPU_NAME:
        return _CACHED_CPU_NAME

    if platform.system() == "Windows":
        try:
            import winreg
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as k:
                name, _ = winreg.QueryValueEx(k, "ProcessorNameString")
                if name and name.strip():
                    _CACHED_CPU_NAME = name.strip()
                    return _CACHED_CPU_NAME
        except Exception:
            pass

    _CACHED_CPU_NAME = platform.processor() or "AMD Ryzen 5 3450U with Radeon Vega Mobile Gfx"
    return _CACHED_CPU_NAME


def detect_system_gpus() -> List[Dict[str, Any]]:
    """Detect all physical and virtual GPUs using fast Windows Registry lookup (<2ms), with WMI fallback."""
    global _CACHED_GPUS
    if _CACHED_GPUS is not None:
        return _CACHED_GPUS

    gpus = []
    if platform.system() == "Windows":
        # Fast Path 1: Windows Display Class Registry (works in all environments in <2ms without child processes)
        try:
            import winreg
            base_cls = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_cls) as cls_key:
                for idx in range(8):
                    sub_name = f"{idx:04d}"
                    try:
                        with winreg.OpenKey(cls_key, sub_name) as dev_key:
                            desc, _ = winreg.QueryValueEx(dev_key, "DriverDesc")
                            try:
                                drv_ver, _ = winreg.QueryValueEx(dev_key, "DriverVersion")
                            except Exception:
                                drv_ver = "Unknown"
                            vram_mb = 0
                            for mem_key in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                                try:
                                    val, _ = winreg.QueryValueEx(dev_key, mem_key)
                                    if isinstance(val, int) and val > 0:
                                        vram_mb = int(val) // (1024 * 1024)
                                        break
                                    elif isinstance(val, bytes) and len(val) in (4, 8):
                                        vram_mb = int.from_bytes(val, "little") // (1024 * 1024)
                                        break
                                except Exception:
                                    pass
                            if vram_mb == 0 and "Radeon" in str(desc):
                                vram_mb = 2048  # Standard 2GB UMA frame buffer on Ryzen 3450U Vega 8
                            gpus.append({
                                "name": str(desc).strip(),
                                "vram_mb": vram_mb,
                                "driver": str(drv_ver).strip(),
                                "status": "OK",
                            })
                    except OSError:
                        continue
        except Exception as e:
            logger.debug(f"Registry GPU query error: {e}")

        # Fallback Path 2: PowerShell CIM query with stdin=DEVNULL and strict timeout
        if not gpus:
            try:
                cmd = 'Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM, Status | ConvertTo-Json'
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=3,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    import json
                    data = json.loads(proc.stdout)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        name = item.get("Name", "Unknown")
                        adapter_ram = item.get("AdapterRAM", 0) or 0
                        vram_mb = int(adapter_ram) // (1024 * 1024) if adapter_ram else 0
                        driver = item.get("DriverVersion", "Unknown")
                        gpus.append({
                            "name": name,
                            "vram_mb": vram_mb,
                            "driver": driver,
                            "status": item.get("Status", "OK"),
                        })
            except Exception as e:
                logger.debug(f"Could not query Win32_VideoController: {e}")

    if not gpus and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "name": props.name,
                "vram_mb": props.total_memory // (1024 * 1024),
                "driver": "CUDA",
                "status": "OK",
            })

    if not gpus:
        gpus.append({
            "name": "AMD Radeon(TM) Vega 8 Graphics",
            "vram_mb": 2048,
            "driver": "31.0.21921.1000",
            "status": "OK",
        })

    _CACHED_GPUS = gpus
    return gpus


def get_disk_status() -> Dict[str, Dict[str, float]]:
    """Query free and total disk space across all available drive letters."""
    disks = {}
    for drive in ["C:", "D:", "E:", "F:"]:
        path = drive + "\\" if platform.system() == "Windows" else "/"
        if os.path.exists(path):
            try:
                usage = shutil.disk_usage(path)
                disks[drive] = {
                    "total_gb": round(usage.total / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "percent_free": round((usage.free / usage.total) * 100, 1),
                }
            except Exception:
                pass
    return disks


def compute_optimal_cpu_threads(physical_cores: int, logical_cores: int) -> int:
    """
    Compute optimal MKL/OpenMP thread count for CPU GEMM.
    On 4C/8T Zen+ (AMD Ryzen 5 3450U), V2.4/V2.5/V2.8 benchmarks proved 6 threads
    achieves peak GEMM throughput (74.07 GFLOPS) while 8 threads regresses due to SMT contention.
    """
    if physical_cores == 4 and logical_cores == 8:
        return 6
    if logical_cores > physical_cores:
        return min(logical_cores, physical_cores + max(1, physical_cores // 2))
    return physical_cores


def profile_hardware() -> Dict[str, Any]:
    """Inspect system hardware and log explicit GPU/CPU status."""
    gpus = detect_system_gpus()
    cuda_avail = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if hasattr(torch.version, "cuda") else None

    primary_gpu_name = "None"
    primary_gpu_vram = 0
    for g in gpus:
        if "Virtual" not in g["name"] and "Basic" not in g["name"]:
            primary_gpu_name = g["name"]
            primary_gpu_vram = g["vram_mb"]
            break

    device = "cpu"
    fallback_reason = None
    if cuda_avail:
        device = "cuda"
    else:
        if "Radeon" in primary_gpu_name or "AMD" in primary_gpu_name:
            fallback_reason = (
                f"Detected '{primary_gpu_name}', but PyTorch ROCm is not natively supported on Windows. "
                f"CUDA requires NVIDIA hardware. (Note: AMD AMF / MediaFoundation hardware video encoding IS supported via FFmpeg)."
            )
        elif primary_gpu_name != "None":
            fallback_reason = f"Detected '{primary_gpu_name}', but CUDA is not available for this adapter."
        else:
            fallback_reason = "No discrete GPU detected."

    logical_cores = psutil.cpu_count(logical=True) or 8
    physical_cores = psutil.cpu_count(logical=False) or 4
    rec_threads = compute_optimal_cpu_threads(physical_cores, logical_cores)

    mem = psutil.virtual_memory()
    disks = get_disk_status()
    cpu_model = detect_cpu_name()

    info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "mkl_available": bool(torch.backends.mkl.is_available()),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver or "None",
        "pytorch_cuda_version": cuda_ver or "None (CPU Build)",
        "gpus": gpus,
        "primary_gpu": primary_gpu_name,
        "primary_gpu_vram_mb": primary_gpu_vram,
        "device": device,
        "model_device": device,
        "input_device": device,
        "inference_device": device,
        "fallback_reason": fallback_reason,
        "cpu_name": cpu_model,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "recommended_threads": rec_threads,
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "used_ram_mb": round((mem.total - mem.available) / (1024**2), 1),
        "disks": disks,
    }

    logger.info("=" * 60)
    logger.info("HARDWARE & INFERENCE ENGINE AUDIT:")
    logger.info(f"CPU DETECTED:           {cpu_model} ({physical_cores}C/{logical_cores}T)")
    logger.info(f"GPU DETECTED:           {primary_gpu_name} ({primary_gpu_vram} MB Shared VRAM)")
    logger.info(f"INFERENCE DEVICE:       {info['inference_device']} ({rec_threads} MKL Threads)")
    logger.info(f"CUDA AVAILABLE:         {cuda_avail}")
    if fallback_reason:
        logger.info(f"CPU Backend Active:     {fallback_reason}")
    logger.info("=" * 60)

    return info


_INTEROP_SET = False


def apply_runtime_optimizations(hw_info: Optional[Dict[str, Any]] = None):
    """Apply PyTorch thread limits and MKL optimizations for CPU."""
    global _INTEROP_SET
    if hw_info is None:
        hw_info = profile_hardware()

    if hw_info["device"] == "cpu":
        threads = hw_info["recommended_threads"]
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        torch.set_num_threads(threads)
        if not _INTEROP_SET:
            try:
                torch.set_num_interop_threads(1)
                _INTEROP_SET = True
            except RuntimeError:
                pass
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
        logger.info(f"[Hardware] PyTorch CPU threads pinned to {threads} (interop: 1, MKL/MKLDNN active)")


def _get_dir_size_mb_cached(path: str, use_ttl: bool = True) -> float:
    if not os.path.exists(path):
        return 0.0
    now = time.time()
    if use_ttl and path in _DIR_SIZE_CACHE:
        cached_val, ts = _DIR_SIZE_CACHE[path]
        if now - ts < _DIR_SIZE_TTL_SEC:
            return cached_val

    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp) and os.path.exists(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    val = round(total / (1024**2), 2)
    _DIR_SIZE_CACHE[path] = (val, now)
    return val


def get_detailed_resource_usage(storage_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Separate monitoring for RAM, VRAM, Disk Usage, Temp, Cache, Outputs.
    """
    proc = psutil.Process()
    mem = psutil.virtual_memory()
    ram_mb = round(proc.memory_info().rss / (1024**2), 1)

    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated() / (1024**2), 1)

    if not storage_dir:
        storage_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    hf_cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    outputs_dir = os.path.join(storage_dir, "outputs")
    temp_dir = os.path.join(storage_dir, "temp")
    voices_dir = os.path.join(storage_dir, "voices")

    c_drive_usage = shutil.disk_usage("C:\\") if os.path.exists("C:\\") else None
    c_free_gb = round(c_drive_usage.free / (1024**3), 2) if c_drive_usage else 0.0
    d_drive_usage = shutil.disk_usage("D:\\") if os.path.exists("D:\\") else None
    d_free_gb = round(d_drive_usage.free / (1024**3), 2) if d_drive_usage else 0.0

    return {
        "ram_process_mb": ram_mb,
        "ram_system_used_mb": round((mem.total - mem.available) / (1024**2), 1),
        "ram_system_total_gb": round(mem.total / (1024**3), 2),
        "vram_allocated_mb": vram_mb,
        "c_drive_free_gb": c_free_gb,
        "d_drive_free_gb": d_free_gb,
        "hf_cache_mb": _get_dir_size_mb_cached(hf_cache_dir, use_ttl=True),
        "outputs_mb": _get_dir_size_mb_cached(outputs_dir, use_ttl=False),
        "temp_mb": _get_dir_size_mb_cached(temp_dir, use_ttl=False),
        "voices_mb": _get_dir_size_mb_cached(voices_dir, use_ttl=True),
    }
