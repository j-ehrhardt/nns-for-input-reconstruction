import torch
import torchvision
import matplotlib.pyplot as plt

def import_data():
    data = torch.utils.data.DataLoader(torchvision.datasets.MNIST("./data", transform=torchvision.transforms.ToTensor(), download=True), batch_size=128, shuffle=True)
    return data


def get_sample(data):
    sample = next(iter(data))
    x = sample[0][6]
    x = x.unsqueeze(1)
    return x.to("cpu")


def mask_sample(sample):
    upper, middle, lower = sample[:, :, :10, :].to("cpu"), sample[:, :, 10:18, :].to("cpu"), sample[:, :, 18:28, :].to("cpu")
    middle = torch.zeros(middle.shape, dtype=torch.float32, device="cpu")
    masked_sample = torch.cat((upper, middle, lower), dim=2).to("cpu")
    return masked_sample


def save_model(path, model):
    torch.save(model.state_dict(), path + "model.pth")
    return


def load_model(path, model):
    model = model.load_state_dict(torch.load(path + "model.pth"))
    return model


def plot_sample(sample, title=""):
    plt.imshow(sample.squeeze().cpu().numpy())
    plt.title(title)
    plt.show()
    plt.close()