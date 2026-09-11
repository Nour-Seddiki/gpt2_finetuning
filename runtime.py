"""
Opt-in, machine-specific process setup shared by the training and eval scripts. Every knob is an
env var that defaults to off, so runs elsewhere (e.g. the rented Linux A100) are unaffected.
Written for the local 8 GB RTX 5050 Windows laptop - see CLAUDE.md for the measurements.

  CUDA_MEM_FRACTION=0.8  cap PyTorch's CUDA caching allocator at this fraction of VRAM. The Windows
                         (WDDM) driver never fails a cudaMalloc when VRAM is full - it silently falls
                         back to shared system RAM - so the allocator never gets the OOM that makes
                         it free its cache, and its reserved memory grows past the card.
  PIN_CPUS=0x0FFF        (Windows) pin the process to these logical CPUs at AboveNormal priority.
                         Started from a background shell, Windows can schedule the run on the
                         efficiency cores, ~5x slower for these kernel-launch-bound loops.
                         0x0FFF = the i7-13620H's P-core threads.
  KEEP_AWAKE=1           (Windows) keep the machine out of sleep / Modern Standby while the process
                         runs - standby suspends desktop apps, freezing an unattended run at the
                         idle timeout. Released automatically when the process exits.
"""
import ctypes
import os
import sys

import torch


def _kernel32():
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # the declared types matter: untyped, GetCurrentProcess()'s 64-bit pseudo-handle is passed as
    # a 32-bit int, the handle is invalid, and the calls fail silently
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    k32.SetThreadExecutionState.restype = wintypes.DWORD
    return k32


def configure_runtime(device):
    """Applies the env-var knobs above. Call once, before the first CUDA allocation, from the
    thread that runs the job - KEEP_AWAKE holds only while that thread is alive."""
    fraction = os.environ.get("CUDA_MEM_FRACTION")
    if fraction and str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(float(fraction))
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        print(f"CUDA allocator capped at {float(fraction)} x {total:.2f} GiB")

    pin, keep_awake = os.environ.get("PIN_CPUS"), os.environ.get("KEEP_AWAKE") == "1"
    if not (pin or keep_awake):
        return
    if sys.platform != "win32":
        print("PIN_CPUS / KEEP_AWAKE are Windows-only - ignored")
        return
    k32 = _kernel32()
    if pin:
        proc = k32.GetCurrentProcess()
        ok = k32.SetProcessAffinityMask(proc, int(pin, 0)) and k32.SetPriorityClass(proc, 0x8000)  # 0x8000 = ABOVE_NORMAL
        print(f"pinned to CPUs {pin} at AboveNormal priority" if ok
              else f"PIN_CPUS={pin} failed (Windows error {ctypes.get_last_error()})")
    if keep_awake:
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED - on Modern Standby machines the
        # display timing out is what triggers standby, so the display flag is the one that matters
        ok = k32.SetThreadExecutionState(0x80000000 | 0x1 | 0x2) != 0
        print("keep-awake on until exit" if ok else f"KEEP_AWAKE failed (Windows error {ctypes.get_last_error()})")
