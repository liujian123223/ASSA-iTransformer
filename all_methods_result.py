import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def all_methods_result(data_path):
    data = pd.read_csv(data_path)
    x = data['NAME-all'][:17]
    y_5min_mae = data['MAE-all'][:17]
    y_10min_mae = data['MAE-all-10'][:17]
    y_20min_mae = data['MAE-all-20'][:17]
    y_30min_mae = data['MAE-all-30'][:17]

    y_5min_mape= data['MAPE-all'][:17]
    y_10min_mape = data['MAPE-all-10'][:17]
    y_20min_mape = data['MAPE-all-20'][:17]
    y_30min_mape = data['MAPE-all-30'][:17]

    y_5min_rmse = data['RMSE-all'][:17]
    y_10min_rmse = data['RMSE-all-10'][:17]
    y_20min_rmse = data['RMSE-all-20'][:17]
    y_30min_rmse = data['RMSE-all-30'][:17]

    y_5min_r2 = data['R2-all'][:17]
    y_5min_r2 = [max(0, value) for value in y_5min_r2]
    y_10min_r2 = data['R2-all-10'][:17]
    y_10min_r2 = [max(0, value) for value in y_10min_r2]
    y_20min_r2 = data['R2-all-20'][:17]
    y_20min_r2 = [max(0, value) for value in y_20min_r2]
    y_30min_r2 = data['R2-all-30'][:17]
    y_30min_r2 = [max(0, value) for value in y_30min_r2]

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 3, 3, 3], 'width_ratios': [1]})
    axs[0].plot(x, y_5min_mae, label='5min', color='red', linestyle='-', marker='o', linewidth=3)
    axs[0].plot(x, y_10min_mae, label='10min', color='cyan', linestyle='-', marker='s', linewidth=3)
    axs[0].plot(x, y_20min_mae, label='20min', color='orange', linestyle='-', marker='h', linewidth=3)
    axs[0].plot(x, y_30min_mae, label='30min', color='palegreen', linestyle='-', marker='d', linewidth=3)
    axs[0].set_ylabel('MAE(W)', fontsize=14)
    axs[0].legend(loc='upper right')
    axs[0].set_xticks([])

    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].plot(x, 100 * y_5min_mape, label='5min', color='red', linestyle='-', marker='o', linewidth=3)
    axs[1].plot(x, 100 * y_10min_mape, label='10min', color='cyan', linestyle='-', marker='s', linewidth=3)
    axs[1].plot(x, 100 * y_20min_mape, label='20min', color='orange', linestyle='-', marker='h', linewidth=3)
    axs[1].plot(x, 100 * y_30min_mape, label='30min', color='palegreen', linestyle='-', marker='d', linewidth=3)
    axs[1].set_ylabel('MAPE(%)', fontsize=14)
    axs[1].set_xticks([])

    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[2].plot(x, y_5min_rmse, label='5min', color='red', linestyle='-', marker='o', linewidth=3)
    axs[2].plot(x, y_10min_rmse, label='10min', color='cyan', linestyle='-', marker='s', linewidth=3)
    axs[2].plot(x, y_20min_rmse, label='20min', color='orange', linestyle='-', marker='h', linewidth=3)
    axs[2].plot(x, y_30min_rmse, label='30min', color='palegreen', linestyle='-', marker='d', linewidth=3)
    axs[2].set_ylabel('RMSE(W)', fontsize=14)
    axs[2].set_xticks([])

    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[3].plot(x, y_5min_r2, label='5min', color='red', linestyle='-', marker='o', linewidth=3)
    axs[3].plot(x, y_10min_r2, label='10min', color='cyan', linestyle='-', marker='s', linewidth=3)
    axs[3].plot(x, y_20min_r2, label='20min', color='orange', linestyle='-', marker='h', linewidth=3)
    axs[3].plot(x, y_30min_r2, label='30min', color='palegreen', linestyle='-', marker='d', linewidth=3)

    axs[3].set_ylabel('R2', fontsize=14)
    axs[3].set_xticklabels(x, rotation=90)
    axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    plt.show()

