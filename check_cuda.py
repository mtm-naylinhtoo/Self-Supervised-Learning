import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# Print CUDA version
if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")

    # Print GPU name
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Print number of GPUs
    print(f"Number of GPUs: {torch.cuda.device_count()}")


# #FOR Sillicon Macs
# import torch

# # Check if GPU is available (Metal API)
# cuda_available = torch.cuda.is_available()
# print(f"CUDA (Metal) available: {cuda_available}")

# # Print device properties if GPU available
# if cuda_available:
#     print(f"GPU name: {torch.cuda.get_device_name(0)}")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")

