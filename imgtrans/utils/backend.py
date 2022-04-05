import os

def get_backend():
    backend = os.environ.get('IMG_TRANS_BACKEND')
    if backend == "pytorch" \
        or backend == "torch":
        return "pytorch"
    elif backend == "tensorflow" \
        or backend == "tf":
        return "tensorflow"
    else:
        return "numpy"