"""Backend registry for op_benchmark runtime targets."""


BACKEND_REGISTRY = {}


def register_backend(name):
    """Register a backend instance by name."""

    def decorator(cls):
        BACKEND_REGISTRY[name] = cls()
        return cls

    return decorator


class Backend:
    """Base backend interface."""

    def get_device(self):
        raise NotImplementedError

    def get_hardware_name(self):
        raise NotImplementedError

    def compile(self, generated_code, op, dataset_name=None):
        raise NotImplementedError

    def correctness_execution(self, ref_src):
        raise NotImplementedError

    def time_execution(self, eval_target="ModelNew"):
        raise NotImplementedError

    def cleanup(self):
        pass
