import datetime
import pickle
from copy import deepcopy as copy
import pandas as pd


# This script takes the historic edits of the cities and extracts the time-entries accordingly.
# The code is easy to read and the way we derived temporal information from the edits is straight-forward.


# load already collected data
with open("histories", "rb") as f:
    histories = pickle.load(f)
    
# Minor Cleaning (1)
# Removing POIs from the data, which have malformed historic entries, specifically POIs which 
# were never created according to the data
    
histories = { city : [poi_hist for poi_hist in histories[city] if list(poi_hist.keys())[0] == 1] for city in histories}

# Transform poi historic edits into global city entries 
def get_entries(histories):
    data = list()
    for poi_history in histories:
        entries = list(poi_history.keys())
        for entry in poi_history:
            if entries.index(entry) != 0:
                prev_key = entries[entries.index(entry) - 1]
                prev_d = poi_history[prev_key]
            d = copy(poi_history[entry])
            d["entry"] = entry
            d["tag_add"] = 0
            d["tag_del"] = 0
            d["tag_change"] = 0
            d["loc_change"] = 0
            d["create"] = 0
            d["delete"] = 0
            d["modify"] = 0
            d["recreate"] = 0
            #determine action type
            if entry == 1:
                d["create"] = 1
            elif not d["visible"]:
                d["delete"] = 1
            elif not prev_d["visible"]:
                d["recreate"] = 1   
            else:
                d["modify"] = 1
                #determine modify type
                #loc change
                if any([prev_d["lat"] != d["lat"], prev_d["lon"] != d["lon"]]):
                    d["loc_change"] = 1
                #tag add
                for tag in d["tag"]:
                    if tag not in prev_d["tag"]:
                        d["tag_add"] = 1
                        break

                #tag del
                for tag in prev_d["tag"]:
                    if tag not in d["tag"]:
                        d["tag_del"] = 1
                        break

                #tag change
                for tag in prev_d["tag"]:
                    if tag in d["tag"] and prev_d["tag"][tag] != d["tag"][tag]:
                        d["tag_change"] = 1
                        break
            data.append(d)
            
    return data


def impute_outliers(dat, z_thresh):
    data = dat.copy().values
    Z = zscore(data)
    for i in range(0, len(Z)):
        if Z[i] > z_thresh:
            print(i, data[i], Z[i])
            data[i] = data[i-1]
    return pd.Series(data, index=dat.index)

# transform to dataframe with proper datetime index.
def to_df(data, freq="m", impute_daily_outliers=False, ZT=20):
    df = pd.DataFrame(data)
    df["new_mapper"] = df.index.map(lambda i : 1 if df.uid[i] not in df.uid[:i].to_list() else 0)
    df.index = df.timestamp
    del df.index.name
    df = df[['create', 'delete', 'modify', 'recreate', 'tag_add', 'tag_del', 'tag_change',
           'loc_change', "new_mapper"]]
    if impute_daily_outliers:
        df = df.resample("d").sum()
        for col in df:
            df[col] = impute_outliers(df[col], z_thresh=ZT)
        
    df["activity"] = df[["create", "modify", "delete", "recreate"]].sum(axis=1) 
    
    return df.resample(freq).sum()


data_raw = { city : get_entries(histories[city]) for city in histories }

data_m = { city : to_df(data_raw[city], freq="m") for city in data_raw}

data_w = { city : to_df(data_raw[city], freq="w") for city in data_raw}


data_2w = { city : to_df(data_raw[city], freq="2w") for city in data_raw}

data_3w = { city : to_df(data_raw[city], freq="3w") for city in data_raw}

#save
with open("data_m", "wb") as f:
    pickle.dump(data_m, f)
with open("data_w", "wb") as f:
    pickle.dump(data_w, f)
with open("data_2w", "wb") as f:
    pickle.dump(data_2w, f)
with open("data_3w", "wb") as f:
    pickle.dump(data_3w, f)
    









