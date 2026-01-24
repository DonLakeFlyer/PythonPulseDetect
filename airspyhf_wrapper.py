"""
Python ctypes wrapper for libairspyhf
Provides interface to Airspy HF+ device using the native C library
"""

import ctypes
import sys
import platform
from ctypes import c_int, c_uint, c_uint32, c_uint64, c_void_p, c_char_p, POINTER, Structure, CFUNCTYPE

# Determine library name based on platform
if platform.system() == 'Darwin':  # macOS
    LIB_NAME = 'libairspyhf.dylib'
elif platform.system() == 'Linux':
    LIB_NAME = 'libairspyhf.so'
else:
    raise RuntimeError(f"Unsupported platform: {platform.system()}")

# Try to load the library
try:
    libairspyhf = ctypes.CDLL(LIB_NAME)
except OSError as e:
    print(f"Error: Failed to load {LIB_NAME}")
    print(f"Make sure libairspyhf is installed:")
    print(f"  macOS: brew install airspyhf")
    print(f"  Linux: sudo apt-get install libairspyhf-dev")
    sys.exit(1)

# Return codes
AIRSPYHF_SUCCESS = 0
AIRSPYHF_TRUE = 1
AIRSPYHF_ERROR_INVALID_PARAM = -2
AIRSPYHF_ERROR_NOT_FOUND = -5
AIRSPYHF_ERROR_BUSY = -6
AIRSPYHF_ERROR_NO_MEM = -11
AIRSPYHF_ERROR_LIBUSB = -1000
AIRSPYHF_ERROR_THREAD = -1001
AIRSPYHF_ERROR_STREAMING_THREAD_ERR = -1002
AIRSPYHF_ERROR_STREAMING_STOPPED = -1003
AIRSPYHF_ERROR_OTHER = -9999

# Device handle type
airspyhf_device_t = c_void_p

# Transfer structure
class airspyhf_transfer_t(Structure):
    _fields_ = [
        ('device', airspyhf_device_t),
        ('ctx', c_void_p),
        ('samples', POINTER(ctypes.c_float)),
        ('sample_count', c_int),
        ('dropped_samples', c_uint64)
    ]

# Sample callback type
airspyhf_sample_block_cb_fn = CFUNCTYPE(c_int, POINTER(airspyhf_transfer_t))


# Function prototypes
libairspyhf.airspyhf_open.argtypes = [POINTER(airspyhf_device_t)]
libairspyhf.airspyhf_open.restype = c_int

libairspyhf.airspyhf_close.argtypes = [airspyhf_device_t]
libairspyhf.airspyhf_close.restype = c_int

libairspyhf.airspyhf_start.argtypes = [airspyhf_device_t, airspyhf_sample_block_cb_fn, c_void_p]
libairspyhf.airspyhf_start.restype = c_int

libairspyhf.airspyhf_stop.argtypes = [airspyhf_device_t]
libairspyhf.airspyhf_stop.restype = c_int

libairspyhf.airspyhf_is_streaming.argtypes = [airspyhf_device_t]
libairspyhf.airspyhf_is_streaming.restype = c_int

libairspyhf.airspyhf_set_freq.argtypes = [airspyhf_device_t, c_uint32]
libairspyhf.airspyhf_set_freq.restype = c_int

libairspyhf.airspyhf_set_samplerate.argtypes = [airspyhf_device_t, c_uint32]
libairspyhf.airspyhf_set_samplerate.restype = c_int

libairspyhf.airspyhf_set_hf_agc.argtypes = [airspyhf_device_t, c_uint]
libairspyhf.airspyhf_set_hf_agc.restype = c_int

libairspyhf.airspyhf_set_hf_lna.argtypes = [airspyhf_device_t, c_uint]
libairspyhf.airspyhf_set_hf_lna.restype = c_int

libairspyhf.airspyhf_set_hf_att.argtypes = [airspyhf_device_t, c_uint]
libairspyhf.airspyhf_set_hf_att.restype = c_int

# Error name lookup (some versions may not have airspyhf_error_name function)
ERROR_NAMES = {
    AIRSPYHF_SUCCESS: "AIRSPYHF_SUCCESS",
    AIRSPYHF_TRUE: "AIRSPYHF_TRUE",
    AIRSPYHF_ERROR_INVALID_PARAM: "AIRSPYHF_ERROR_INVALID_PARAM",
    AIRSPYHF_ERROR_NOT_FOUND: "AIRSPYHF_ERROR_NOT_FOUND",
    AIRSPYHF_ERROR_BUSY: "AIRSPYHF_ERROR_BUSY",
    AIRSPYHF_ERROR_NO_MEM: "AIRSPYHF_ERROR_NO_MEM",
    AIRSPYHF_ERROR_LIBUSB: "AIRSPYHF_ERROR_LIBUSB",
    AIRSPYHF_ERROR_THREAD: "AIRSPYHF_ERROR_THREAD",
    AIRSPYHF_ERROR_STREAMING_THREAD_ERR: "AIRSPYHF_ERROR_STREAMING_THREAD_ERR",
    AIRSPYHF_ERROR_STREAMING_STOPPED: "AIRSPYHF_ERROR_STREAMING_STOPPED",
    AIRSPYHF_ERROR_OTHER: "AIRSPYHF_ERROR_OTHER"
}


# Wrapper functions
def open():
    """Open Airspy HF+ device."""
    device = airspyhf_device_t()
    result = libairspyhf.airspyhf_open(ctypes.byref(device))
    return result, device


def close(device):
    """Close Airspy HF+ device."""
    return libairspyhf.airspyhf_close(device)


def start(device, callback):
    """
    Start receiving samples.

    Args:
        device: Device handle
        callback: Python callback function that takes (transfer) and returns int
    """
    # Create ctypes callback wrapper
    c_callback = airspyhf_sample_block_cb_fn(callback)
    # Keep reference to prevent garbage collection
    if not hasattr(device, '_callback_refs'):
        device._callback_refs = []
    device._callback_refs.append(c_callback)

    result = libairspyhf.airspyhf_start(device, c_callback, None)
    return result


def stop(device):
    """Stop receiving samples."""
    return libairspyhf.airspyhf_stop(device)


def is_streaming(device):
    """Check if device is currently streaming."""
    return libairspyhf.airspyhf_is_streaming(device) == AIRSPYHF_TRUE


def set_freq(device, freq_hz):
    """
    Set center frequency.

    Args:
        device: Device handle
        freq_hz: Frequency in Hz
    """
    return libairspyhf.airspyhf_set_freq(device, c_uint32(freq_hz))


def set_samplerate(device, samplerate):
    """
    Set sample rate.

    Args:
        device: Device handle
        samplerate: Sample rate in Hz
    """
    return libairspyhf.airspyhf_set_samplerate(device, c_uint32(samplerate))


def set_hf_agc(device, enable):
    """
    Enable/disable AGC.

    Args:
        device: Device handle
        enable: 1 to enable, 0 to disable
    """
    return libairspyhf.airspyhf_set_hf_agc(device, c_uint(enable))


def set_hf_lna(device, enable):
    """
    Enable/disable LNA.

    Args:
        device: Device handle
        enable: 1 to enable, 0 to disable
    """
    return libairspyhf.airspyhf_set_hf_lna(device, c_uint(enable))


def set_hf_att(device, att_step):
    """
    Set HF attenuation.

    Args:
        device: Device handle
        att_step: Attenuation step (0-8 for 0-48 dB in 6 dB steps)
    """
    return libairspyhf.airspyhf_set_hf_att(device, c_uint(att_step))


def error_name(error_code):
    """Get error name string from error code."""
    return ERROR_NAMES.get(error_code, f"Unknown error: {error_code}")
