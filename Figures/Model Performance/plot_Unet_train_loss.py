
import os
import numpy as np
import matplotlib.pyplot as plt


def read_loss_from_training_summary_file(project_folder, resolution, model_version):
    summary_path = os.path.join(project_folder, 'Data', str(resolution) + 'km Model', model_version,
                                f'{model_version}_training_summary.txt')

    losses = []
    with open(summary_path, 'r') as f:
        for line in f:
            if 'Train Loss' in line:
                loss_str = line.split('Train Loss: ')[1].split(' |')[0]
                losses.append(float(loss_str))

    return losses


def plot_training_loss(project_folder, resolution, model_version, losses):

    fig =plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title(f'{model_version} Training Loss Over Time')
    plt.legend()
    plt.grid()

    output_file = os.path.join(project_folder, 'Figures','Model Assessment',
                               f'{model_version}_training_loss.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet1'
resolution = 15

losses = read_loss_from_training_summary_file(project_folder, resolution, model_version)

plot_training_loss(project_folder, resolution, model_version, losses)












