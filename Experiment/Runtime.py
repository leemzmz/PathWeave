def getDevice(preferred=None):
    if preferred is not None:
        return preferred

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
