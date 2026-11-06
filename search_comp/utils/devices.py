"""Explicit one-model-per-GPU worker placement."""

def worker_devices(num_workers, visible):
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if (len(devices) < num_workers or len(set(devices)) != len(devices)
            or any(device == "-1" for device in devices)):
        raise ValueError("set CUDA_VISIBLE_DEVICES to distinct GPUs, at least one per worker")
    return devices[:num_workers]
