### >>> Import ###
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
### <<< Import ###

### >>> Functions ###
def loss_plot_new(loss_matrix, EMA = True, metric = 'loss', recalibration = False, title = 'Loss', ylim = 1e+04, alpha = 0.2, save = False, path = '../data/results/loss/plots/', filename = ''):
    iterations = loss_matrix[:, 0].astype(float)
    training_loss = loss_matrix[:, 1].astype(float)
    evaluation_loss1 = loss_matrix[:, 2].astype(float)
    evaluation_loss2 = loss_matrix[:, 3].astype(float)
    if recalibration == False:
        evaluation_loss3 = loss_matrix[:, 4].astype(float)

    if EMA != True:
        plt.figure(figsize=(10, 6))

        plt.plot(iterations, training_loss, label='Calibration ' + str(metric), marker='o')
        plt.plot(iterations, evaluation_loss1, label='Evaluation ' + str(metric) + ' 1', marker='o')
        plt.plot(iterations, evaluation_loss2, label='Evaluation ' + str(metric) + ' 2', marker='o')
        if recalibration == False:
            plt.plot(iterations, evaluation_loss3, label='Evaluation ' + str(metric) + '3', marker='o')

        plt.ylim(0,ylim)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(str(metric.capitalize()))
        plt.legend()
        plt.grid(True)
        plt.show()

    if EMA == True and recalibration == False:
        training_loss = pd.Series(training_loss).ewm(alpha=alpha).mean()
        evaluation_loss1 = pd.Series(evaluation_loss1).ewm(alpha=alpha).mean()
        evaluation_loss2 = pd.Series(evaluation_loss2).ewm(alpha=alpha).mean()
        if recalibration == False:
            evaluation_loss3 = pd.Series(evaluation_loss3).ewm(alpha=alpha).mean()

        # Plot
        plt.figure(figsize=(10, 6))

        # EMA
        plt.plot(iterations, training_loss, label=f'Calibration ' + str(metric), linestyle='--')
        plt.plot(iterations, evaluation_loss1, label=f'Evaluation ' + str(metric) + ' 1', linestyle='--')
        plt.plot(iterations, evaluation_loss2, label=f'Evaluation ' + str(metric) + ' 2', linestyle='--')
        if recalibration == False:
            plt.plot(iterations, evaluation_loss3, label=f'Evaluation ' + str(metric) + ' 3', linestyle='--')

        plt.ylim(0,ylim)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(str(metric.capitalize()))
        plt.legend()
        plt.grid(True)
        if save == True:
            plt.savefig(str(path) + str(filename) + '.png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.show()

    if recalibration == True:
        training_loss = pd.Series(training_loss).ewm(alpha=alpha).mean()
        evaluation_loss1 = pd.Series(evaluation_loss1).ewm(alpha=alpha).mean()
        evaluation_loss2 = pd.Series(evaluation_loss2).ewm(alpha=alpha).mean()

        # Plot
        plt.figure(figsize=(10, 6))

        # EMA
        plt.plot(iterations, training_loss, label=f'Recalibration ' + str(metric), linestyle='--', color = "blue")
        plt.plot(iterations, evaluation_loss1, label=f'Evaluation ' + str(metric) + ' 2', linestyle='--', color = "green")
        plt.plot(iterations, evaluation_loss2, label=f'Evaluation ' + str(metric) + ' 3', linestyle='--', color = "red")

        plt.ylim(0, ylim)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(str(metric.capitalize()))
        plt.legend()
        plt.grid(True)
        if save == True:
            plt.savefig(str(path) + str(filename) + '.png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.show()
### <<< Functions ###

### >>> Parameter ###
name = 'ruven'
epoch = 200
epoch_recalibrated = 50
save = True
ylim_loss = 2e04
y_lim_dist = 200
### <<< Parameter ###

metric = 'loss'
data = np.load('../data/results/loss/' + str(name) + '_loss_' + str(epoch) + '.npy')
loss_plot_new(data, EMA = True, ylim = ylim_loss, metric= metric, save = save, filename = str(name) + '_' + str(metric), title = metric.capitalize() + ' - Calibration')
data = np.load('../data/results/loss/' + str(name) + '_recalibrated_loss_' + str(epoch_recalibrated) + '.npy')
loss_plot_new(data, recalibration=True, EMA = True, ylim = ylim_loss, metric= metric, save = save, filename = str(name) + '_' + str(metric) + '_recalibrated', title = metric.capitalize() + ' - Recalibration')

metric = 'distance'
data = np.load('../data/results/loss/' + str(name) + '_distance_' + str(epoch) + '.npy')
loss_plot_new(data, EMA = True, ylim = y_lim_dist, metric= metric, save = save, filename = str(name) + '_' + str(metric), title = metric.capitalize() + ' - Calibration')
data = np.load('../data/results/loss/' + str(name) + '_recalibrated_distance_' + str(epoch_recalibrated) + '.npy')
loss_plot_new(data, recalibration=True, EMA = True, ylim= y_lim_dist, metric= metric, save = save, filename = str(name) + '_' + str(metric) + '_recalibrated', title = metric.capitalize() + ' - Recalibration')

