
import os
import numpy as np
import matplotlib.pyplot as plt



def get_availability_list():

    data_availability = {}

    sst_availability = {'MUR':[2002,2024]}
    data_availability['SST'] = sst_availability

    sst_availability = {'SMAP':[2015,2024]}
    data_availability['SSS'] = sst_availability

    currents_availability = {'OSCAR':[1993,2024]}
    data_availability['Currents'] = currents_availability

    chl_availability = {'OC-CCIv6':[1997,2024]}
    data_availability['Chlorophyll'] = chl_availability

    return data_availability

def plot_data_availability_gantt(project_folder, data_availability):

    fig, ax = plt.subplots(figsize=(7,8))

    min_year = 1992
    max_year = 2025

    data_types = list(data_availability.keys())

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    counter = 0.5
    for i, data_type in enumerate(data_types):

        sources = data_availability[data_type].keys()

        height = len(sources)+1
        plt.plot([min_year, max_year], [counter-1.5+height, counter-1.5+height], 'k-', linewidth=1)

        for source in sources:
            start_year, end_year = data_availability[data_type][source]
            plt.plot([start_year, end_year], [counter, counter], '-', label=source, color=colors[i], linewidth=2)
            plt.text(end_year + 0.1, counter, source, va='center')

            counter += 1

    plt.ylim(0, counter + 0.5)
    plt.xlim(min_year, max_year)
    ax.set_xlabel('Year')
    ax.set_title('Data Availability')
    ax.legend()

    plt.savefig(os.path.join(project_folder, 'Figures', 'Data_Availability_Gantt.png'), dpi=300)
    plt.close(fig)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Biology'

data_availability = get_availability_list()

plot_data_availability_gantt(project_folder, data_availability)





