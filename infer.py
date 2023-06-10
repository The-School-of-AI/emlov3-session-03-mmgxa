import json
from pathlib import Path
from torchvision import datasets, transforms
from io import BytesIO
import base64
from model_file import Net
import torch
import os

def eval():
    model = Net()
    model.load_state_dict(torch.load("./mount/model/mnist_cnn.pt"))
    model.eval()
    dataset = datasets.MNIST('./mount/data', train=False, download=True)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    for i in range(5):
        ind = torch.randint(len(dataset), size=(1,)).item()
        # ind = random.randint(0, len(dataset))
        image,label = dataset[ind]
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')

        with torch.no_grad():
            image= transform(image).unsqueeze(0)
            prediction = model(image)
            predicted_class = torch.argmax(prediction, dim=1).item()
        filename = f'mount/results/predicted_{predicted_class}_label_{label}_index_{ind}.png'
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(img_str))


def main():
    with (Path(".") / "mount" / "model" / "eval_results.json").open("r") as f:
        eval_results = json.load(f)

    if eval_results['Accuracy'] > 95:
        print('Model well trained!')
        if not os.path.exists("./mount/results"):
            print('creating directory for results')
            os.makedirs("./mount/results")
        eval()
    else:
        print('Model poorly trained. Please re-train and re-evaluate!')

if __name__ == "__main__":
    main()