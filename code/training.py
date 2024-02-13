import random
import numpy as np

from models import *
from reconstruction import *

def train(model, data, epochs=20, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())

    with tqdm(range(epochs), unit="epoch") as tepoch:
        for e in tepoch:
            for x, y in data:
                x = x.to(device)
                opt.zero_grad()

                upper, middle, lower = x[:, :, :10, :], x[:, :, 10:18, :], x[:, :, 18:28, :]

                upper_hat, middle_hat, lower_hat = model(upper, middle, lower)
                x_hat = torch.cat((upper_hat, middle_hat, lower_hat), dim=2)
                loss = ((x - x_hat) ** 2).sum()
                loss.backward()
                opt.step()

            tepoch.set_postfix(train_loss=loss.item())
    return model, loss.item()


if __name__ == "__main__":
    # init experiment
    path = "../eval/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # init model and data
    mnist = import_data()
    ae = Autoencoder(latent_dims=16)

    """  """
    # train + save model
    trained_ae, final_loss = train(ae, mnist, epochs=20, device=device)
    save_model(path, trained_ae)


    # select sample for reconstruction
    sample = get_sample(data=mnist)
    masked_sample = mask_sample(sample)

    # plot sample and masked sample
    plot_sample(sample)
    plot_sample(masked_sample)

    # reconstruct
    sampling_steps, res_dict = 10, {}
    for step in range(sampling_steps):
        random.seed(step)
        np.random.seed(step)
        torch.manual_seed(step)
        trained_ae = load_model(path, ae)
        final_reconstruction, loss_sampling = reconstruct(ae, masked_sample, epochs=5, device=device)
        res_dict[step] = final_reconstruction
        plot_sample(final_reconstruction)

    # plot best result:
    plot_sample(res_dict[min(res_dict.keys())], title="Best Result")


