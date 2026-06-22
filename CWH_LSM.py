import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_spikes,
)
from bindsnet.datasets import MNIST, SpokenMNIST
from bindsnet.encoding import PoissonEncoder
from Dataset.NMNIST import NMNIST

from bindsnet.network.monitors import Monitor
from Network.liquid_state_network import LiquidStateNetwork, RandnWeight, GammaWeightRandnMask, GammaWeightRandMask, ParetoWeightRandnMask, CauchyWeightRandnMask


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=784)
parser.add_argument("--examples", type=int, default=10000)
parser.add_argument("--n_workers", type=int, default=1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=80)

parser.add_argument("--no-train", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--output_neuron", type=int, default=900)

parser.add_argument("--cuda_name", type=str, default='cuda:1')
parser.add_argument("--mode", type=str, default='MNIST')
parser.add_argument("--weight_mode", type=str, default='Gaussian')
parser.add_argument("--hybrid", dest="hybrid", action="store_true")

parser.set_defaults(plot=True, gpu=True, hybrid=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
output_neuron = args.output_neuron
cuda_name = args.cuda_name
examples = args.examples
n_workers = args.n_workers
time = args.time
dt = args.dt
intensity = args.intensity
train = args.train
plot = args.plot
gpu = args.gpu

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

if output_neuron == 400:
    CWH = [50, 100, 250]
elif output_neuron == 784:
    CWH = [100, 300, 384]
elif output_neuron == 900:
    CWH = [100, 300, 500]
elif output_neuron == 1600:
    CWH = [300, 500, 800]
elif output_neuron == 2500:
    CWH = [500, 800, 1200]

else:
    raise ValueError(f"Unsupported neuron configuration: {output_neuron}")


if args.hybrid:
    save_name = f'CWH_{args.mode}_Hybird_train_multi_{sum(CWH)}_{len(CWH)}'
else:
    save_name = f'CWH_{args.mode}_{args.weight_mode}_train_multi_{sum(CWH)}_{len(CWH)}'

# Sets up Gpu use
device = torch.device(f"{cuda_name}" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  #
else:
    n_workers = torch.cuda.is_available() * 4 * torch.cuda.device_count()

# Create simple Torch NN
lsm = LiquidStateNetwork(dt=args.dt)
# Add input layers
lsm.add_input_layer(784, shape=(1, 28, 28), name="I")


def single_LSM(CWH, connect_name, weight_mode):
    if weight_mode == "Gamma":
        lsm.create_liquid_layer(CWH, connect_name)
        lsm.connect_layers("I",
                           target_name=connect_name,
                           weight_initializer=GammaWeightRandMask(),
                           connection_type="feedforward")
        lsm.connect_layers(source_name=connect_name,
                           target_name=connect_name,
                           weight_initializer=GammaWeightRandnMask(),
                           connection_type="feedforward")

    elif weight_mode == "Gaussian":
        lsm.create_liquid_layer(CWH, connect_name)
        lsm.connect_layers("I",
                           target_name=connect_name,
                           weight_initializer=RandnWeight(),
                           connection_type="feedforward")
        lsm.connect_layers(source_name=connect_name, target_name=connect_name,
                           weight_initializer=RandnWeight(),
                           connection_type="feedforward")

    elif weight_mode == "Pareto":
        lsm.create_liquid_layer(CWH, connect_name)
        lsm.connect_layers("I",
                           target_name=connect_name,
                           weight_initializer=ParetoWeightRandnMask(),
                           connection_type="feedforward")
        lsm.connect_layers(source_name=connect_name, target_name=connect_name,
                           weight_initializer=ParetoWeightRandnMask(),
                           connection_type="feedforward")
    elif weight_mode == "Cauchy":
        lsm.create_liquid_layer(CWH, connect_name)
        lsm.connect_layers("I",
                           target_name=connect_name,
                           weight_initializer=CauchyWeightRandnMask(),
                           connection_type="feedforward")
        lsm.connect_layers(source_name=connect_name, target_name=connect_name,
                           weight_initializer=CauchyWeightRandnMask(),
                           connection_type="feedforward")

    else:
        raise ValueError(f"Unsupported connection mode: {weight_mode}")


if args.hybrid:
    single_LSM(CWH[0], "O_1", "Gamma")
    single_LSM(CWH[1], "O_2", "Gaussian")
    single_LSM(CWH[2], "O_3", "Pareto")
    print(f"\nUsing hybrid Distribution...\n")

else:
    single_LSM(CWH[0], "O_1", args.weight_mode)
    single_LSM(CWH[1], "O_2", args.weight_mode)
    single_LSM(CWH[2], "O_3", args.weight_mode)
    print(f"\nUsing {args.weight_mode} Distribution...\n")


network = lsm.get_network().to(device)
print(network)

# Initialize monitors
spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=args.time, device=device)
    network.add_monitor(spikes[l], name=f"{l}_spikes")


# Load MNIST data.
if args.mode == "MNIST":
    dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join("/home/zhangyoudong", "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
        train=train,
    )

elif args.mode == "FSDD":
    dataset = SpokenMNIST(path='/home/zhangyoudong/data/FSDD', download=False, split=0.8, train=True,
                                  shuffle=True, audio_encoder=PoissonEncoder(time=time, dt=dt), mfcc_dimensions=28)

elif args.mode == "N-MNIST":
    dataset = NMNIST("/home/zhangyoudong/data/NMNIST", trian=True)

else:
    raise ValueError(f"Unsupported dataset: {args.mode}")

dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )


inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

n_iters = len(dataloader)
training_pairs = []

pbar = tqdm(enumerate(dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break

    if args.mode == "MNIST":
        datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
        label = dataPoint["label"]
        image = dataPoint["image"]
        pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
    elif args.mode == "FSDD":
        datum = dataPoint["encoded_audio"].view(int(time / dt), 1, 1, 28, 28).to(device)
        label = dataPoint["label"]
        audio = dataPoint["audio"]
        pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
    elif args.mode == "N-MNIST":
        datum = dataPoint["image"][:, :, 0, :, :].view(50, 1, 1, 28, 28).to(device)
        label = dataPoint["label"]
        pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
    else:
        raise ValueError(f"Unsupported dataset: {args.mode}")

    network.run(inputs={"I": datum}, time=args.time)

    # 分别从三个不同尺度的 LSM 输出神经元提取spike张量
    spike_1 = spikes["O_1"].get("s")
    spike_2 = spikes["O_2"].get("s")
    spike_3 = spikes["O_3"].get("s")

    concatenated_spikes = torch.concat((spike_1, spike_2, spike_3), dim=2)

    training_pairs.append([concatenated_spikes, label])

    # Plot spiking activity using monitors
    if plot:
        # inpt_axes, inpt_ims = plot_input(
        #     dataPoint["image"].view(28, 28),
        #     datum.view(int(time / dt), 784).sum(0).view(28, 28),
        #     label=label,
        #     axes=inpt_axes,
        #     ims=inpt_ims,
        # )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        # voltage_ims, voltage_axes = plot_voltages(
        #     {layer: voltages[layer].get("v").view(time, -1) for layer in voltages},
        #     ims=voltage_ims,
        #     axes=voltage_axes,
        # )
        # weights_im = plot_weights(
        #     get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
        # )
        # weights_im2 = plot_weights(C2_1.w, im=weights_im2, wmin=-2, wmax=2)

        plt.pause(1e-8)
    network.reset_state_variables()

training_pairs_dicts = [
    {"encoded_image": spikes, "label": label}
    for spikes, label in training_pairs
]

torch.save(training_pairs_dicts, open('./LSM_OUT/{}.pt'.format(save_name), "wb"))
