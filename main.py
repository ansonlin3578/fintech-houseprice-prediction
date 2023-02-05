import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import seaborn as sns
import pdb
class load_data(object):
    def load_train(path, dataset):
        df = pd.read_csv(path)
        dataset["train"]["data"] = df
        dataset["train"]["label"] = df["price"]
    def load_test(path, dataset):
        df = pd.read_csv(path)
        dataset["test"]["data"] = df
        dataset["test"]["label"] = df["price"]

dataset = {
            "train" : {
                        "data" : [],
                        "label" : []
                        },
            "test" : {
                        "data" : [],
                        "label" : []
                        }
            }
load_data.load_train(r"C:\Users\User\Desktop\碩一上修課資料\金融科技\final_project\dataset\realtor-data.csv", dataset)

data_ori = dataset["train"]["data"]
data_ori = data_ori.drop(columns=["full_address", "street", "city"])
data_ori = data_ori[data_ori["price"].notna()]
# for column in data_ori:
#     print("nan value in {} : ".format(column) , data_ori[column].isna().sum())

data_ori["sold_date"] = pd.to_datetime(data_ori["sold_date"])
data_ori = data_ori.sort_values(by="sold_date")
data_wo_nan = data_ori.dropna()
data_wo_nan = data_wo_nan.reset_index(drop=True)
data_ori = data_ori.reset_index(drop=True)

data_wo_nan_from2020 = data_wo_nan.loc[(data_wo_nan["sold_date"] >= "2020-01-01")]
data_wo_nan_from2020 = data_wo_nan_from2020.reset_index(drop=True)
# print("data_wo_nan_from2020:")
# for column in data_wo_nan_from2020:
#     print("nan value in {} : ".format(column) , data_wo_nan_from2020[column].isna().sum())
# print(data_wo_nan_from2020)
print("##############################\n")
data_wo_nan_from2020.to_csv("data_wo_nan_from2020.csv")

data_bed_bath_fill = data_ori.loc[(data_ori["sold_date"].notna()) & 
                                    (data_ori["house_size"].notna()) & 
                                    (data_ori["acre_lot"].notna()) & 
                                    (data_ori["zip_code"].notna())]
data_bed_bath_fill = data_bed_bath_fill.loc[(data_bed_bath_fill["sold_date"] >= "2020-01-01")]
data_bed_bath_fill = data_bed_bath_fill.reset_index(drop=True)
# print("data_bed_bath_fill:")
# for column in data_bed_bath_fill:
#     print("nan value in {} : ".format(column) , data_bed_bath_fill[column].isna().sum())
# print(data_bed_bath_fill)
data_bed_bath_fill.to_csv("data_bed_bath_fill.csv")
######################################## date interval ################################################

after_filled = pd.read_csv(f"./After_filled.csv")
if not os.path.exists(f"./time_interval.csv"):
    after_filled = after_filled.drop(columns=["Unnamed: 0"])
    for column in after_filled:
        print("nan value in {} : ".format(column) , after_filled[column].isna().sum())

    after_filled["sold_date"] = pd.to_datetime(after_filled["sold_date"])
    after_filled["time_interval"] = np.nan
    after_filled = after_filled.reset_index(drop=True)

    # year_interval = int(after_filled["sold_date"].iloc[-1].year) - int(after_filled["sold_date"].iloc[0].year)
    # total_interval_nums = year_interval * 3
    initial_date = datetime(year=2020, month=1, day=1)
    interval_label = 0
    for i in range(len(after_filled)):
        if after_filled["sold_date"].iloc[i] > initial_date + relativedelta(months=4):
            interval_label += 1
            initial_date = initial_date + relativedelta(months=4)

        after_filled["time_interval"].iloc[i] = interval_label

    print(after_filled)
    after_filled.to_csv(f"./time_interval.csv", index=False)

##################################################################################
to_corr_matrix = pd.read_csv(f"./time_interval.csv")
time_interval = to_corr_matrix
to_corr_matrix = to_corr_matrix.drop(columns=["status", "state", "sold_date"])
corr_matrix = to_corr_matrix.corr()
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="YlGnBu", annot=True)
plt.savefig(f"./corr_matrix.png")
plt.close()

time_interval = time_interval.drop(columns=["status", "sold_date"])
gp_by_state = time_interval.groupby("state")
for state, df in gp_by_state:
    print("state name : ", state )
    pass

################################## data form 2010 ############################################
data_wo_nan_from2010 = data_wo_nan.loc[(data_wo_nan["sold_date"] >= "2010-01-01")]
data_wo_nan_from2010 = data_wo_nan_from2010.reset_index(drop=True)
print("data_wo_nan_from2010:")
for column in data_wo_nan_from2010:
    print("nan value in {} : ".format(column) , data_wo_nan_from2010[column].isna().sum())
print(data_wo_nan_from2010)
data_wo_nan_from2010.to_csv("data_wo_nan_from2010.csv")

if not os.path.exists(f"./time_interval_2010.csv"):
    data_wo_nan_from2010["time_interval"] = np.nan
    
    initial_date = datetime(year=2010, month=1, day=1)
    interval_label = 0
    for i in range(len(data_wo_nan_from2010)):
        if data_wo_nan_from2010["sold_date"].iloc[i] > initial_date + relativedelta(months=2):
            interval_label += 1
            initial_date = initial_date + relativedelta(months=2)
        data_wo_nan_from2010["time_interval"].iloc[i] = interval_label

    print(data_wo_nan_from2010)

    data_wo_nan_from2010.to_csv(f"./time_interval_2010.csv", index=False)

to_corr_matrix = pd.read_csv(f"./time_interval_2010.csv")
time_interval = to_corr_matrix
to_corr_matrix = to_corr_matrix.drop(columns=["status", "state", "sold_date"])
corr_matrix = to_corr_matrix.corr()
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="YlGnBu", annot=True)
plt.savefig(f"./corr_matrix_2010.png")
plt.close()

time_interval = time_interval.drop(columns=["status", "sold_date"])
gp_by_state = time_interval.groupby("state")
# for state, df in gp_by_state:
#     print("state name : ", state, "\t", "house numbers : ", len(df))
######################################## date interval ################################################

data_need_fill_2010 = data_ori.loc[(data_ori["sold_date"].notna()) & 
                                    (data_ori["house_size"].notna()) & 
                                    (data_ori["acre_lot"].notna()) & 
                                    (data_ori["zip_code"].notna())]
data_need_fill_2010 = data_need_fill_2010.loc[(data_need_fill_2010["sold_date"] >= "2010-01-01")]
data_need_fill_2010 = data_need_fill_2010.reset_index(drop=True)
# for column in data_need_fill_2010:
#     print("nan value in {} : ".format(column) , data_need_fill_2010[column].isna().sum())
if not os.path.exists(f"./need_fill_2010.csv"):
    data_need_fill_2010["time_interval"] = np.nan
    
    initial_date = datetime(year=2010, month=1, day=1)
    interval_label = 0
    for i in range(len(data_need_fill_2010)):
        if data_need_fill_2010["sold_date"].iloc[i] > initial_date + relativedelta(months=2):
            interval_label += 1
            initial_date = initial_date + relativedelta(months=2)
        data_need_fill_2010["time_interval"].iloc[i] = interval_label

    print(data_need_fill_2010)

    data_need_fill_2010.to_csv(f"./need_fill_2010.csv", index=False)

need_fill_2010 = pd.read_csv(f"./need_fill_2010.csv")
need_fill_2010 = need_fill_2010.drop(columns=["status", "sold_date"])
gp_by_state = need_fill_2010.groupby("state")
# for state, df in gp_by_state:
#     print("state name : ", state, "\t", "house numbers : ", len(df))
#############################################################################################
after_filled_2010 = pd.read_csv(f"./After_filled_2010.csv")
test_from_2022_01_01 = after_filled_2010.loc[(data_need_fill_2010["sold_date"] >= "2022-01-01")]
test_from_2022_01_01 = test_from_2022_01_01.drop(columns=["Unnamed: 0"])
test_from_2022_01_01 = test_from_2022_01_01.reset_index(drop=True)
if not os.path.exists(f"./test_from_2022_01_01.csv"):
    print(test_from_2022_01_01)
    test_from_2022_01_01.to_csv(f"./test_data_from20220101.csv")

training_data = after_filled_2010[(after_filled_2010["sold_date"] >= "2010-01-01") &( after_filled_2010["sold_date"] < "2022-01-01")]
training_data = training_data.drop(columns=["Unnamed: 0"])
training_data = training_data.reset_index(drop=True)
if not os.path.exists(f"./training_data.csv"):
    print(training_data)
    training_data.to_csv(f"./training_data.csv")

to_corr_matrix = pd.read_csv(f"./After_filled_2010.csv")
to_corr_matrix = to_corr_matrix.drop(columns=["Unnamed: 0", "status", "sold_date"])
corr_matrix = to_corr_matrix.corr()
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="YlGnBu", annot=True)
plt.savefig(f"./final_corr_matrix.png")
plt.show()
