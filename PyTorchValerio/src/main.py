import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# 1- download dataset
# 2- create data loader
# 3- build model
# 4- train
# 5- trained model

BATCH_SIZE = 128
EPOCHS = 10
LERNING_RATE = 0.001


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # flatten is arbitary name
        self.dense_layers = nn.Sequential(
            nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for (
        inputs,
        targets,
    ) in data_loader:
        inputs, targets = input.to(device), targets.to(device)
        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-----------------------")
    print("training is done")


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data", download=True, train=True, transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data", download=True, train=True, transform=ToTensor()
    )
    return train_data, validation_data


if __name__ == "__main__":
    # downlad MNIST dataset
    train_data = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # create a data loader for the train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)

    # instantiate loss function
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LERNING_RATE)

    # train_model
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored at feedfowardnet.pth")
