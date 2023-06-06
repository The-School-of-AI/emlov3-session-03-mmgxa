import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from model_file import Net

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # Training logic for a single epoch
    # ...
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:
                break


def train(args, model, device, dataset, dataloader_kwargs):
    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)



def main():
    # Initialize arguments
    # ...
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 2)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training')
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
    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}

    # Initialize the model
    # ...
    model = Net().to(device)

    # Check if a saved checkpoint exists
    saved_ckpt = Path(".") / "mount" / "model" / "mnist_cnn.pt"

    if os.path.isfile(saved_ckpt):
    # ...
        print('Checkpoint exists. Resuming Training...')
        # Load the model from the checkpoint
        model.load_state_dict(torch.load(saved_ckpt))
    else:
         print('Checkpoint does not exist. Training from scratch...')
        # ...

    # Training loop
    train(args, model, device, train_dataset, kwargs)
    
    # Save the final model checkpoint
    if not os.path.exists("./mount/model"):
        print('creating directory for model')
        os.makedirs("./mount/model")
    torch.save(model.state_dict(), "./mount/model/mnist_cnn.pt")
    


if __name__ == "__main__":
    main()
