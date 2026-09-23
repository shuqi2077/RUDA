"""RUDA execution streams/events. No host wait is hidden in query or wait_event.

The underlying C++ DeviceGuard also supports torch.Stream/torch.Event. These
wrappers provide torch.ruda's context-manager conveniences on the same IDs.
"""
from contextlib import contextmanager
import struct
from . import _C

class Stream:
    def __init__(self, *, priority=0, _id=None):
        if priority != 0:
            raise ValueError("RUDA supports only priority=0")
        self.stream_id = _C.stream_command(1) if _id is None else int(_id)
    def query(self):
        return bool(_C.stream_command(3,self.stream_id))
    def synchronize(self):
        _C.stream_command(4,self.stream_id)
    def wait_event(self, event):
        _C.stream_command(6,self.stream_id,event._id)
    def wait_stream(self, other):
        event=Event(); event.record(other); self.wait_event(event)
    def record_event(self, event=None):
        event=Event() if event is None else event
        event.record(self); return event

class Event:
    def __init__(self, *, enable_timing=False):
        self._id=0; self._timing=bool(enable_timing)
    def record(self, value=None):
        value=current_stream() if value is None else value
        self._id=_C.stream_command(5,value.stream_id,self._id,int(self._timing))
        return self
    def wait(self, value=None):
        (current_stream() if value is None else value).wait_event(self)
    def query(self):
        return bool(_C.stream_command(7,0,self._id))
    def synchronize(self):
        _C.stream_command(8,0,self._id)
    def elapsed_time(self, other):
        if not self._timing or not other._timing:
            raise ValueError("elapsed_time requires timing-enabled events")
        bits=_C.stream_command(10,self._id,other._id)
        return struct.unpack("<f",struct.pack("<I",bits))[0]
    def close(self):
        if self._id:
            _C.stream_command(9,0,self._id); self._id=0
    def __del__(self):
        # Keep finalizers non-throwing, but do not silence runtime errors from
        # explicit close(), record(), query(), wait() or synchronize().
        try:
            self.close()
        except Exception:
            pass

def current_stream(device=None):
    _device(device)
    return Stream(_id=_C.stream_command(0))
def default_stream(device=None):
    _device(device)
    return Stream(_id=0)
def _device(value):
    if value is not None and str(value) not in ("0","ruda","ruda:0"):
        raise ValueError("RUDA currently exposes only ruda:0")
@contextmanager
def stream(value):
    if not isinstance(value,Stream):
        raise TypeError("expected a RUDA Stream")
    previous=_C.stream_command(2,value.stream_id)
    try:
        yield value
    finally:
        _C.stream_command(2,previous)
def record_stream(tensor, value):
    """Keep allocation alive until already submitted work on value completes."""
    if not isinstance(value,Stream):
        raise TypeError("expected a RUDA Stream")
    _C.record_stream(tensor,value.stream_id)
