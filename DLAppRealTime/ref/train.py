import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from ref.urbansound.urbansounddataset import UrbanSoundSataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "./data/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "./data/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        # input, target = input.cuda(), target.cuda()

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-----------------------")
    print("training is done")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )
    usd = UrbanSoundSataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    # create a data loader for the train set
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)

    # build model
    cnn = CNNNetwork().to(device)
    print(cnn)

    # instantiate loss function
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train_model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trained and stored at cnn.pth")
