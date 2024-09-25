import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import shutil
from Subsequence_number_experiment import plot_subsequence_result
from all_methods_result import all_methods_result
colors = ['aqua','salmon','grey','orange','royalblue','gold','violet','#855a8a','limegreen','purple','seagreen',
          'yellow','lightcoral','indianred','steelblue','tomato','slategrey','red','azure','tan','skyblue','chartreuse',
		  'lavender','papayawhip','gainsboro','navajowhite','thistle','teal','indigo','cornsilk','sienna','dodgerblue'
		  'sage','darkmagenta']
house=1
it=1
time_step=it*5
decomposition_method='ssa'
house_csv = "house_{}it_{}_{}.csv".format(house,it,decomposition_method)


metric_Histogram_want_jianhua=['dTr','rDf','SVR','MLP','LSTM','NLSTM','iTransformer',
                               'EWT-LSTM','EWT-ITF','EMD-LSTM','VMD-LSTM','VMD-ITF',
                               'SWT-LSTM','SWT-ITF','SSA-LSTM','PROPOSED']

root_path = './experiment_result/result'
file_all = root_path+'/predict-truth/'
file_path_compare ='./experiment_result/result-compare/House_{}_{}min.csv'.format(house,time_step)
different_decomposition_csv = root_path+'/house{}/{}min/different_decomposition_method/'.format(house,time_step)
file_path_proposed =different_decomposition_csv+house_csv
save_decomposition_metric= root_path+'/house{}/{}min/house{}_{}min_decomposition_metric.csv'.format(house,time_step,house,time_step)
save_decomposition_csv = root_path+'/house{}/{}min/house{}_{}min_decomposition_csv.csv'.format(house,time_step,house,time_step)
metric_pic_save_dir = root_path+'/house{}/{}min/house{}_{}min_pic'.format(house,time_step,house,time_step)
os.makedirs(metric_pic_save_dir, exist_ok=True)
subsequence_result=root_path+"/Subsequence_number_experiment.csv"
def extract_files(folder_path, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    targer_house_step = "house{}_it{}".format(house,it)

    for filename in os.listdir(folder_path):
        if targer_house_step in filename:
            source_file = os.path.join(folder_path, filename)
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(source_file, target_file)

def metric_different_decomposition(file_compare,file_new,file_csv):

    result_filename = file_csv
    folder_path = file_new  
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv_files_sorted = sorted(csv_files, key=lambda file: file.split("_")[-1].split(".")[0])


    result_df = pd.DataFrame()

    for csv_file in csv_files_sorted:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        df = df.iloc[:, 1]

        result_df = pd.concat([result_df, df], axis=1)

    result_df.to_csv(result_filename, index=False,header=[f'{csv_file.split("_")[-1].split(".")[0]}' for csv_file in csv_files_sorted])
    different_decomposition_1 = pd.read_csv(result_filename)
    data_compare = pd.read_csv(file_compare)


    plt.figure(figsize=(20, 10))
    plt.rc('font', size=16)
    plt.xlabel("Time ({} minutes)".format(time_step), fontsize=18)
    plt.ylabel("Energy Consumption (kWh)", fontsize=18)
    plt.subplots_adjust(left=0.05, bottom=0.125, right=0.99, top=0.99)
    plt.xticks(fontsize=14)
    plt.grid(True, linestyle=":", color="gray", linewidth="0.5", axis='both')

    plt.plot(different_decomposition_1.loc[:, 'swt'][0:120], color=colors[1], label='SWT-itransformer', linewidth=1,
             linestyle='-',
             marker='s',
             markersize=5)
    plt.plot(different_decomposition_1.loc[:, 'ewt'][0:120], color=colors[2], label='EWT-itransformer', linewidth=1,
             linestyle='-',
             marker='s',
             markersize=5)
    plt.plot(different_decomposition_1.loc[:, 'vmd'][0:120], color=colors[3], label='VMD-itransformer', linewidth=1,
             linestyle='-',
             marker='s',
             markersize=5)
    plt.plot(different_decomposition_1.loc[:, 'emd'][0:120], color=colors[4], label='EMD-itransformer', linewidth=1,
             linestyle='-',
             marker='s',
             markersize=5)

    plot_want = [1, 2, 3, 4, 6, 9, 10, 14, 18, 22, 27, 28, 29,30]
    for i, model_index in enumerate(plot_want):
        plt.plot(data_compare.iloc[:, model_index][0:120], label=data_compare.columns[model_index], linewidth=1,
                 marker='s', markersize=3, color=colors[i])
    plt.plot(data_compare.loc[:, 'real_data'][0:120], 'black', label='Actual data', linewidth=3, linestyle='--', marker='o',
         markersize=5)  # 13:133
    plt.plot(different_decomposition_1.loc[:, 'ssa'][0:120], 'red', label='{}-itransformer'.format(decomposition_method), linewidth=3,
             linestyle='-', marker='o',
             markersize=5)
   
    plt.subplots_adjust(top=0.95)
    plt.title("The energy consumption forecasting results of all compared and proposed methods in this study", fontsize=20)
    plt.legend(loc='upper right', ncol=2, fontsize=10)
    plt.savefig(os.path.join(metric_pic_save_dir, 'House{}_{}min_ALL_Method_compare.png'.format(house,time_step)))#5min_EMD_MAE_itransformer.png
    plt.show()

    base_column = data_compare['real_data']

    target_columns = ['DecisionTree','RandomForest','SVR','MLP','LSTM','NLSTM','iTransformer','SWT-LSTM', 'SWT-SLSTM', 'SWT-BiLSTM', 'SWT-NLSTM', 'EWT-LSTM', 'EWT-SLSTM', 'EWT-BiLSTM',
                      'EWT-NLSTM', 'VMD-LSTM', 'VMD-SLSTM', 'VMD-BiLSTM', 'VMD-NLSTM', 'EMD-LSTM', 'EMD-SLSTM',
                      'EMD-BiLSTM', 'EMD-NLSTM', 'SSA-SLSTM', 'SSA-BiLSTM', 'SSA-NLSTM']

    results = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'R2'])

    for column in target_columns:
        target_column = data_compare[column]
        mae = mean_absolute_error(base_column, target_column)
        mape = np.mean(np.abs((target_column - base_column) / base_column))
        rmse = np.sqrt(mean_squared_error(base_column, target_column))
        r2 = r2_score(base_column, target_column)

        if column=='DecisionTree':
            column='dTr'
        if column=='RandomForest':
            column='rDf'
        if column =='SSA-SLSTM':
            column='SSA-LSTM'

        results = results.append({'Model': column, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2}, ignore_index=True)

    results.to_csv(save_decomposition_metric, index=False, mode='w')

    csv_files = [file for file in os.listdir(different_decomposition_csv) if file.endswith('.csv')]
    csv_files_sorted = sorted(csv_files, key=lambda file: file.split("_")[-1].split(".")[0])

    with open(save_decomposition_metric, mode='a', newline='') as total_file:
        writer = csv.writer(total_file)
        last_row = None
        for file in csv_files_sorted:
            with open(os.path.join(different_decomposition_csv, file)) as csv_file:
                reader = csv.reader(csv_file)
                data = list(reader)
                if file.split("_")[-1].split(".")[0].upper() + "-ITF" == "SSA-ITF":
                    last_row = ["PROPOSED"] + data[1][2:]
                else:
                    row = [file.split("_")[-1].split(".")[0].upper() + "-ITF"] + data[1][2:]
                    writer.writerow(row)
        if last_row is not None:
            writer.writerow(last_row)

def plot_zhuzhuangtu(want):
    metric_want = want
    df = pd.read_csv(save_decomposition_metric)
    dm = 'ALL_Jianhua'
    num_line=len(df['MAE'])- 1
    line_name = 'Proposed'

    x = df.loc[df['Model'].isin(metric_want)]['Model'].tolist()
    colors = ['steelblue'] * (len(x) - 1) + ['red']


    df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')
    y = [val if val >= 0 else 0 for val in df.loc[df['Model'].isin(metric_want)]['MAE'].tolist()]
    y_SSA_itransformer = df['MAE'][num_line]
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(left=0.05, bottom=0.125, right=0.99, top=0.99)
    ax = plt.gca()
    ax.bar(x, y, width=0.5, color=colors)

    ax.axhline(y=y_SSA_itransformer, color='red', linestyle='--',
               label=line_name,lw=2)
    ax.set_title('MAE of different signal decomposition methods', fontsize=18)
    ax.set_ylabel('MAE', fontsize=15, labelpad=10)
    ax.tick_params(axis='x', rotation=90, labelsize=11)
    ax.tick_params(axis='y', labelsize=12)
    plt.legend(loc='upper right',prop={'size': 5})
    plt.tight_layout()
    plt.savefig(os.path.join(metric_pic_save_dir, 'House{}_{}min_{}_MAE_itransformer.png'.format(house,time_step,dm)))
    plt.show()

    x = df.loc[df['Model'].isin(metric_want)]['Model'].tolist()
    colors = ['steelblue'] * (len(x) - 1) + ['red']

    df['MAPE'] = pd.to_numeric(df['MAPE'], errors='coerce')
    y = [val if val >= 0 else 0 for val in df.loc[df['Model'].isin(metric_want)]['MAPE'].tolist()]
    y_SSA_itransformer = df['MAPE'][num_line]
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(left=0.05, bottom=0.125, right=0.99, top=0.99)
    ax = plt.gca()
    ax.bar(x, y, width=0.5, color=colors)
    ax.axhline(y=y_SSA_itransformer, color='red', linestyle='--',
               label=line_name,lw=2)
    ax.set_title('MAPE of different signal decomposition methods', fontsize=18)
    ax.set_ylabel('MAPE', fontsize=15, labelpad=10)
    ax.tick_params(axis='x', rotation=90, labelsize=11)
    ax.tick_params(axis='y', labelsize=12)
    plt.legend(loc='upper right',prop={'size': 5})
    plt.tight_layout()
    plt.savefig(os.path.join(metric_pic_save_dir, 'House{}_{}min_{}_MAPE_itransformer.png'.format(house,time_step,dm)))
    plt.show()

    x = df.loc[df['Model'].isin(metric_want)]['Model'].tolist()
    colors = ['steelblue'] * (len(x) - 1) + ['red']

    df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
    y = [val if val >= 0 else 0 for val in df.loc[df['Model'].isin(metric_want)]['RMSE'].tolist()]
    y_SSA_itransformer = df['RMSE'][num_line]

    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(left=0.05, bottom=0.125, right=0.99, top=0.99)
    ax = plt.gca()
    ax.bar(x, y, width=0.5, color=colors)
    ax.axhline(y=y_SSA_itransformer, color='red', linestyle='--',
               label=line_name,lw=2)
    ax.set_title('RMSE of different signal decomposition methods', fontsize=18)
    ax.set_ylabel('RMSE', fontsize=15, labelpad=10)
    ax.tick_params(axis='x', rotation=90, labelsize=11)
    ax.tick_params(axis='y', labelsize=12)
    plt.legend(loc='upper right',prop={'size': 5})
    plt.tight_layout()
    plt.savefig(os.path.join(metric_pic_save_dir, 'House{}_{}min_{}_RMSE_itransformer.png'.format(house,time_step,dm)))
    plt.show()

    x = df.loc[df['Model'].isin(metric_want)]['Model'].tolist()
    colors = ['steelblue'] * (len(x) - 1) + ['red']


    df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
    y = [val if val >= 0 else 0 for val in df.loc[df['Model'].isin(metric_want)]['R2'].tolist()]
    y_SSA_itransformer = df['R2'][num_line]
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(left=0.05, bottom=0.125, right=0.99, top=0.99)
    ax = plt.gca()
    ax.bar(x, y, width=0.5, color=colors)
    ax.axhline(y=y_SSA_itransformer, color='red', linestyle='--',
               label=line_name,lw=2)
    ax.set_title('R2 of different signal decomposition methods', fontsize=18)
    ax.set_ylabel('R2', fontsize=15, labelpad=10)
    ax.tick_params(axis='x', rotation=90, labelsize=11)
    ax.tick_params(axis='y', labelsize=12)
    plt.legend(loc='upper right',prop={'size': 5})
    plt.tight_layout()
    plt.savefig(os.path.join(metric_pic_save_dir, 'House{}_{}min_{}_R2_itransformer.png'.format(house,time_step,dm)))
    plt.show()

extract_files(file_all, different_decomposition_csv)

metric_different_decomposition(file_path_compare,different_decomposition_csv,save_decomposition_csv)

plot_zhuzhuangtu(metric_Histogram_want_jianhua)

all_methods_result(root_path+"/house1_all_methods.csv")

plot_subsequence_result(subsequence_result)