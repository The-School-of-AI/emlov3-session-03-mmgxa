import os
import json
import torch
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from model_file import Net

        
def test_epoch(model, device, data_loader):
    # Test the model on the test dataset and calculate metrics
    # ...
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    out = {'Test loss': test_loss, 'Accuracy': accuracy}

    # Print the evaluation results
    print(out)

    # Return the evaluation results
    return out


def main():
    # Initialize arguments
    # ...
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    # Set device (CPU or GPU)
    # ...
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set data loaders
    # ...
    torch.manual_seed(args.seed)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST('./mount/data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST('./mount/data', train=False,
                       transform=transform)
    kwargs = {'batch_size': args.test_batch_size,
              'shuffle': False}
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    # Initialize the model
    # ...
    model = Net().to(device)

    # Load the saved model checkpoint
    saved_ckpt = Path(".") / "mount" /"model" / "mnist_cnn.pt"
    model.load_state_dict(torch.load(saved_ckpt))
    # ...

    # Evaluate the model on the test dataset
    eval_results = test_epoch(model, device, test_loader)
    # ...

    # Save the evaluation results to a JSON file
    with (Path(".") / "mount" / "model" / "eval_results.json").open("w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
