"""Minimal compiler device contexts; this does not register an Inductor backend."""
from contextlib import contextmanager

import torch


def _device_index(device):
    if device is None:
        return 0
    if type(device) is int:
        index = device
    else:
        value = torch.device(device)
        if value.type != 'ruda':
            raise ValueError('expected a ruda device')
        index = 0 if value.index is None else value.index
    if index != 0:
        raise ValueError('RUDA currently exposes only ruda:0')
    return index


def register_device_interface():
    from torch._dynamo.device_interface import DeviceInterface, register_interface_for_device, device_interfaces
    existing = device_interfaces.get('ruda')
    if existing is not None:
        if not getattr(existing, '_ruda_model_compiler', False):
            raise RuntimeError('another compiler interface is already registered for ruda')
        return

    class RudaInterface(DeviceInterface):
        _ruda_model_compiler = True
        # Generic stream/event classes delegate to the existing C++ DeviceGuard.
        Stream = torch.Stream
        Event = torch.Event

        @staticmethod
        @contextmanager
        def device(device):
            _device_index(device)
            yield

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def set_device(device):
            _device_index(device)

        @staticmethod
        def exchange_device(device):
            return _device_index(device)

        maybe_exchange_device = exchange_device

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def is_available():
            from . import is_available
            return is_available()

        @staticmethod
        def synchronize(device=None):
            _device_index(device)
            from . import synchronize
            synchronize()

        @staticmethod
        def is_bf16_supported(including_emulation=False):
            return True

    register_interface_for_device('ruda', RudaInterface)
    register_interface_for_device('ruda:0', RudaInterface)
