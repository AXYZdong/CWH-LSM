import argparse
import os
from time import time as t
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import csv
import glob

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)

from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import IncreasingInhibitionNetwork
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights


# ===================== Parser =====================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_workers", type=int, default=1)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--update_inhibation_weights", type=int, default=500)
parser.add_argument("--plot_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--no-plot", dest="plot", action="store_false")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--no-gpu", dest="gpu", action="store_false")
parser.add_argument("--som_neurons", type=int, default=900)
parser.add_argument("--cuda_name", type=str, default='cuda:0')
parser.add_argument("--dataset_name", type=str, required=True, choices=["MNIST", "FSDD", "NMNIST"])
parser.set_defaults(plot=True, gpu=True)

args = parser.parse_args()

# ===================== Dataset fixed hyperparams =====================
dataset_hp = {
    "MNIST": {"input_neuron":900, "n_epochs":12, "time":250},
    "FSDD": {"input_neuron":900, "n_epochs":300, "time":250},
    "NMNIST": {"input_neuron":900, "n_epochs":1, "time":50},
}
hp = dataset_hp[args.dataset_name]
input_neuron = hp["input_neuron"]
n_epochs = hp["n_epochs"]
time = hp["time"]

seed = args.seed
som_neurons = args.som_neurons
cuda_name = args.cuda_name
n_workers = args.n_workers
theta_plus = args.theta_plus
dt = 1.0
progress_interval = 10
plot_interval = args.plot_interval
update_interval = args.update_interval
plot = args.plot
gpu = args.gpu
update_inhibation_weights = args.update_inhibation_weights

# ===================== Scan all LSM encoded pt files, only keep filename contains "all" =====================
lsm_data_dir = os.path.join("./LSM_OUT", args.dataset_name)
pt_file_list = glob.glob(os.path.join(lsm_data_dir, "*.pt"))
# filter: only pt with "all" in filename
pt_file_list = [p for p in pt_file_list if "_all" in os.path.basename(p)]
pt_file_list = sorted(pt_file_list)
print(f"\nFound {len(pt_file_list)} pt files (only filename contains 'all') in {lsm_data_dir}")

summary_records = []

# ===================== Global folder prepare =====================
os.makedirs("Train_Out", exist_ok=True)
os.makedirs("acc", exist_ok=True)
os.makedirs("model_weights/model", exist_ok=True)
os.makedirs("model_weights/weight", exist_ok=True)

# ===================== Loop each pt =====================
for pt_path in pt_file_list:
    pt_basename = os.path.basename(pt_path)
    plot_name = pt_basename.replace(".pt", "")
    print(f"\n========== Start training for: {plot_name} ==========")
    print(f"Loading LSM encoded data: {pt_path}")

    # Set device & seed
    device = torch.device("{}".format(cuda_name) if torch.cuda.is_available() else "cpu")
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
        device = "cpu"
        if gpu:
            gpu = False
    torch.set_num_threads(os.cpu_count() - 1)
    print("Running on Device = ", device)

    n_sqrt = int(np.ceil(np.sqrt(som_neurons)))
    plt_sqrt = int(np.ceil(np.sqrt(input_neuron)))
    start_intensity = 64

    # Build network (REINIT per pt)
    network = IncreasingInhibitionNetwork(
        n_input=input_neuron,
        n_neurons=som_neurons,
        start_inhib=10,
        max_inhib=-40.0,
        theta_plus=0.05,
        tc_theta_decay=1e7,
        inpt_shape=(1, plt_sqrt, plt_sqrt),
        nu=(1e-4, 1e-2),
    )
    network.to(device)

    # Record spikes during the simulation.
    spike_record = torch.zeros((update_interval, int(time / dt), som_neurons), device=device)

    # Neuron assignments and spike proportions.
    n_classes = 10
    assignments = -torch.ones(som_neurons, device=device)
    proportions = torch.zeros((som_neurons, n_classes), device=device)
    rates = torch.zeros((som_neurons, n_classes), device=device)

    # Sequence of accuracy estimates.
    accuracy = {"all": [], "proportion": []}

    # Voltage recording for excitatory and inhibitory layers.
    som_voltage_monitor = Monitor(
        network.layers["Y"], ["v"], time=int(time / dt), device=device
    )
    network.add_monitor(som_voltage_monitor, name="som_voltage")

    # Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    voltages = {}
    for layer in set(network.layers) - {"X"}:
        voltages[layer] = Monitor(
            network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
        )
        network.add_monitor(voltages[layer], name="%s_voltages" % layer)

    # Plot handles
    inpt_ims, inpt_axes = None, None
    spike_ims, spike_axes = None, None
    weights_im = None
    assigns_im = None
    perf_ax = None
    voltage_axes, voltage_ims = None, None

    # ========== CHANGED: exp_root nested with dataset name ==========
    exp_root = os.path.join("Train_Out", args.dataset_name, plot_name)
    dirs_exp = [
        exp_root,
        os.path.join(exp_root, "weights"),
        os.path.join(exp_root, "performance"),
        os.path.join(exp_root, "assaiments")
    ]
    for d in dirs_exp:
        os.makedirs(d, exist_ok=True)

    save_weights_fn = os.path.join(exp_root, "weights", "weights.png")
    save_performance_fn = os.path.join(exp_root, "performance", "performance.png")
    save_assaiments_fn = os.path.join(exp_root, "assaiments", "assaiments.png")

    # diagonal weights for increassing the inhibitiosn
    weights_mask = (1 - torch.diag(torch.ones(som_neurons))).to(device)

    # Load LSM pre-encoded dataset
    dataloader = torch.load(open(pt_path, "rb"), map_location=device)

    print("\nBegin training.\n")
    start = t()
    best_acc = 0

    for epoch in range(n_epochs):
        labels = []
        if epoch % progress_interval == 0:
            print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
            start = t()

        n_train = len(dataloader)
        pbar = tqdm(total=n_train)
        for step, batch in enumerate(dataloader):
            if step > n_train:
                break
            # Get next input sample.
            inputs = {
                "X": batch['encoded_image'].view(int(time / dt), 1, 1, plt_sqrt, plt_sqrt).to(device)
            }

            if step > 0:
                if step % update_inhibation_weights == 0:
                    if step % (update_inhibation_weights * 10) == 0:
                        network.Y_to_Y.w -= weights_mask * 50
                    else:
                        network.Y_to_Y.w -= weights_mask * 0.5

                if step % update_interval == 0:
                    # Convert the array of labels into a tensor
                    label_tensor = torch.tensor(labels, device=device)

                    # Get network predictions.
                    all_activity_pred = all_activity(
                        spikes=spike_record, assignments=assignments, n_labels=n_classes
                    )
                    proportion_pred = proportion_weighting(
                        spikes=spike_record,
                        assignments=assignments,
                        proportions=proportions,
                        n_labels=n_classes,
                    )

                    # Compute network accuracy according to available classification strategies.
                    acc_all = 100 * torch.sum(torch.tensor(label_tensor.long() == all_activity_pred)).item() / len(label_tensor)
                    acc_prop = 100 * torch.sum(torch.tensor(label_tensor.long() == proportion_pred)).item() / len(label_tensor)
                    accuracy["all"].append(acc_all)
                    accuracy["proportion"].append(acc_prop)

                    tqdm.write(
                        "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                        % (
                            accuracy["all"][-1],
                            np.mean(accuracy["all"]),
                            np.max(accuracy["all"]),
                        )
                    )
                    tqdm.write(
                        "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                        " (best)\n"
                        % (
                            accuracy["proportion"][-1],
                            np.mean(accuracy["proportion"]),
                            np.max(accuracy["proportion"]),
                        )
                    )

                    # Assign labels to excitatory layer neurons.
                    assignments, proportions, rates = assign_labels(
                        spikes=spike_record,
                        labels=label_tensor,
                        n_labels=n_classes,
                        rates=rates,
                    )

                    # save the best model
                    if accuracy["all"][-1] > best_acc:
                        best_acc = accuracy["all"][-1]
                        print("Saving model")
                        network.save(os.path.join('model_weights/model', f'network_{plot_name}.pt'))

                    labels = []

            labels.append(batch['label'])

            temp_spikes = 0
            for retry in range(5):
                # Run the network on the input.
                network.run(inputs=inputs, time=time)
                # Get spikes from the network
                temp_spikes = spikes["Y"].get("s").squeeze()
                if temp_spikes.sum().sum() < 2:
                    inputs["X"] *= batch['encoded_image'].view(int(time / dt), 1, 1, plt_sqrt, plt_sqrt).to(device)
                else:
                    break

            # Get voltage recording.
            exc_voltages = som_voltage_monitor.get("v")
            # Add to spikes recording.
            spike_record[step % update_interval].copy_(temp_spikes, non_blocking=True)

            # Optionally plot various simulation information.
            if plot and step % plot_interval == 0:
                inpt = inputs["X"].view(time, input_neuron).sum(0).view(plt_sqrt, plt_sqrt)
                input_exc_weights = network.connections[("X", "Y")].w
                square_weights = get_square_weights(
                    input_exc_weights.view(input_neuron, som_neurons), n_sqrt, plt_sqrt
                )
                square_assignments = get_square_assignments(assignments, n_sqrt)
                spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
                voltages = {"Y": exc_voltages}

                spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
                assigns_im = plot_assignments(
                    square_assignments, im=assigns_im, save=save_assaiments_fn
                )
                perf_ax = plot_performance(accuracy, ax=perf_ax, save=save_performance_fn)
                voltage_ims, voltage_axes = plot_voltages(
                    voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
                )
                np.save(os.path.join(exp_root, f'assignments_visualization_{plot_name}.npy'),
                        square_assignments.detach().clone().cpu().numpy())
                plt.pause(1e-8)

            network.reset_state_variables()
            pbar.set_description_str("Train progress: ")
            pbar.update()

        print("\n Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))

    print("Training complete.\n")

    # Save individual acc csv for this pt
    combined_data = list(zip(accuracy["all"], accuracy["proportion"]))
    csv_file_name = os.path.join("acc", f'{plot_name}_acc.csv')
    with open(csv_file_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['All', 'Proportion'])
        writer.writerows(combined_data)

    # Save assignments and proportion
    np.save(os.path.join('model_weights/weight', f'assignments_{plot_name}_last.npy'), assignments.cpu().numpy())
    np.save(os.path.join('model_weights/weight', f'proportions_{plot_name}_last.npy'), proportions.cpu().numpy())

    # record max acc for summary
    max_all = np.max(accuracy["all"])
    max_prop = np.max(accuracy["proportion"])
    summary_records.append([pt_basename, max_all, max_prop])
    print(f"Record summary: {pt_basename} | max_all={max_all:.2f}, max_prop={max_prop:.2f}")

# ========= Save dataset summary csv (Option A: Train_Out/{dataset_name}_summary_acc.csv) =========
summary_csv_path = os.path.join(f"Train_Out/{args.dataset_name}", f"{args.dataset_name}_summary_acc.csv")
with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["pt_filename", "all_max_acc", "proportion_max_acc"])
    writer.writerows(summary_records)
print(f"\n==== Dataset {args.dataset_name} all experiments finished ====")
print(f"Summary csv saved to {summary_csv_path}")
