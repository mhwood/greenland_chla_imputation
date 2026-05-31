
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_loss_from_training_summary_file(project_folder, resolution, model_version):
    summary_path = os.path.join(project_folder, 'Data', str(resolution) + 'km Model', model_version,
                                f'{model_version}_training_summary.txt')

    epochs = []
    batches = []
    train_losses = []
    test_losses = []
    with open(summary_path, 'r') as f:
        for line in f:
            if 'Train Loss' in line:
                line = line.split(' | ')

                epoch = int(line[0].split()[1].split('/')[0])
                epochs.append(epoch)

                batch = int(line[1].split()[1].split('/')[0])
                batches.append(batch)

                train_loss = float(line[2].split()[2])
                train_losses.append(train_loss)

                test_loss = float(line[3].split()[2])
                test_losses.append(test_loss)

    return epochs,batches,train_losses,test_losses


def plot_training_loss(project_folder, resolution, model_version,
                       epochs, batches, train_losses, test_losses):

    epochs = np.array(epochs)
    n_epochs = np.max(epochs)
    epoch_bounds = {}
    for epoch in range(1,n_epochs+1):
        indices = np.where(epochs==epoch)[0]
        epoch_bounds[epoch] = [indices[0],indices[-1]]

    min_loss = np.min([np.min(test_losses), np.min(train_losses)])
    max_loss = np.max([np.max(test_losses), np.max(train_losses)])
    loss_range = max_loss-min_loss

    fig =plt.figure(figsize=(10, 8))

    plt.subplot(2,1,1)
    plt.plot(train_losses, label='Training Loss')
    plt.ylabel('Loss on Training Data')
    plt.title(f'{model_version} Loss Over Epochs')
    plt.grid(linestyle='--', linewidth=0.5)
    for epoch in range(1,n_epochs+1):
        if epoch%2==0:
            rect = Rectangle((epoch_bounds[epoch][0],min_loss-loss_range),
                             epoch_bounds[epoch][1]-epoch_bounds[epoch][0], 3*loss_range,
                             facecolor='green', alpha=0.2)
            plt.gca().add_patch(rect)
    y_min = np.min(train_losses)
    y_max = np.max(train_losses)
    y_range = y_max-y_min
    plt.gca().set_ylim([y_min-0.1*y_range, y_max+0.1*y_range])
    plt.gca().set_xlim([0,len(batches)])

    plt.subplot(2,1, 2)
    plt.plot(test_losses, label='Training Loss')
    # plt.xlabel('Cumulative Batch Number')
    plt.ylabel('Loss on Testing Data')
    plt.grid(linestyle='--', linewidth=0.5)
    for epoch in range(1,n_epochs+1):
        if epoch%2==0:
            rect = Rectangle((epoch_bounds[epoch][0], min_loss - loss_range),
                             epoch_bounds[epoch][1] - epoch_bounds[epoch][0], 3 * loss_range,
                             facecolor='green', alpha=0.2)
            plt.gca().add_patch(rect)
    y_min = np.min(test_losses)
    y_max = np.max(test_losses)
    y_range = y_max - y_min
    plt.gca().set_ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])
    plt.gca().set_xlim([0, len(batches)])

    output_file = os.path.join(project_folder, 'Figures','Model Assessment',
                               f'{model_version}_training_loss.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet2'
resolution = 15

epochs,batches,train_losses,test_losses = \
    read_loss_from_training_summary_file(project_folder, resolution, model_version)

plot_training_loss(project_folder, resolution, model_version,
                   epochs, batches, train_losses, test_losses)












