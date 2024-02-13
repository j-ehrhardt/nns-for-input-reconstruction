from tqdm import tqdm
from utils import *

def reconstruct(model, sample, epochs=50, device="cpu"):
    sample = sample.to(device)
    model = model.to(device)
    upper, middle, lower = sample[:, :, :10, :].to(device), sample[:, :, 10:18, :].to(device), sample[:, :, 18:28, :].to(device)

    reconstructions = {}

    middle_autodiff = torch.zeros(middle.shape, requires_grad=True, dtype=torch.float32, device=device)
    opt = torch.optim.SGD([middle_autodiff], lr=0.1, momentum=0)

    # no model eval because of dropout
    with tqdm(range(epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            opt.zero_grad()

            upper_hat, middle_hat, lower_hat = model(upper, middle_autodiff, lower)
            sample_hat = torch.cat((upper, middle_hat, lower), dim=2).to(device)
            loss = ((sample - sample_hat) ** 2).sum()
            loss.backward()
            opt.step()

            reconstructions[epoch] = sample_hat.cpu().detach()
            # plot intermediate optimization steps
            #if epoch % 5 == 0:
            #    plot_sample(sample_hat)
    return reconstructions[min(reconstructions.keys())], min(reconstructions.keys())