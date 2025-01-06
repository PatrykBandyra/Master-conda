import torch

if __name__ == '__main__':
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Select the desired GPU (e.g., GPU 0)
        device = torch.device('cuda:0')
        print(f'Using GPU: {torch.cuda.get_device_name(device)}')

        # Create a tensor on the GPU
        x = torch.randn(10, device=device)

        # Perform an operation on the GPU
        y = x * 2

        # Check if the result is on the correct device
        print(f'Result is on: {y.device}')
    else:
        print('CUDA is not available. Using CPU.')
        device = torch.device('cpu')
