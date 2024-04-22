import torch

if torch.cuda.is_available():
    # CUDA is available
    device = torch.device("cuda")
    print("CUDA is available.")
    print("Device name:", torch.cuda.get_device_name(0))  # Print the name of the GPU
    print("CUDA version:", torch.version.cuda)  # Print the CUDA version
    print("CUDNN version:", torch.backends.cudnn.version())  # Print the cuDNN version
else:
    # CUDA is not available
    print("CUDA is not available. Running on CPU.")
    device = torch.device("cpu")