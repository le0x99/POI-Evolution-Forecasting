import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBoost
from copy import deepcopy as copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
plt.rcParams["figure.figsize"] = (5, 3.5)
plt.rcParams["figure.dpi"] = 150
plt.style.use("seaborn");plt.style.use("classic")
from itertools import product
%config InlineBackend.figure_format='retina'


#load data
with open("data_w", "rb") as f:
    data_w = pickle.load(f)
    
with open("data_2w", "rb") as f:
    data_2w = pickle.load(f)
    
with open("data_3w", "rb") as f:
    data_3w = pickle.load(f)
    
with open("data_m", "rb") as f:
    data_m = pickle.load(f)
    
    
cities = list(data_m.keys())
## The baseline Models

# A) V_T ~ V_{T-1}

def baseline1(data, city, cols, test_size):
    dat = data[city][cols].copy()
    y = dat[-test_size:]
    #y_train = dat[:-test_size]
    pred = dat[-test_size-1:].shift(1).dropna()
    SE = (y - pred )**2
    #AE = np.abs(pred - y)
    #SE_naive = (y_train.mean() - y)**2
    #AE_naive = np.abs((y_train.mean() - y))
    #RRSE = np.sqrt((SE.sum()/SE_naive.sum()))
    #RAE = AE.sum()/AE_naive.sum()
    RMSE = np.sqrt(SE.mean())
    #NRMSE = RMSE/dat.std()
    res = RMSE.to_dict()
    res["MRMSE"] = RMSE.mean()
    res["city"] = city
    return res


# B) V_T ~ AVG(V_{T-2 : T-1})
        
def baseline2(data, city,cols, test_size):
    dat = data[city][cols].copy()
    y = dat[-test_size:]
    #y_train = dat[:-test_size]
    pred = dat[-test_size-2:].shift(1).rolling(2).mean().dropna()
    SE = (y - pred )**2
    #AE = np.abs(pred - y)
    #SE_naive = (y_train.mean() - y)**2
    #AE_naive = np.abs((y_train.mean() - y))
    #RRSE = np.sqrt((SE.sum()/SE_naive.sum()))
    #RAE = AE.sum()/AE_naive.sum()
    RMSE = np.sqrt(SE.mean())
    #NRMSE = RMSE/dat.std()
    res = RMSE.to_dict()
    res["MRMSE"] = RMSE.mean()
    res["city"] = city
    return res

def build_model(method, dat, max_lag, n_test,
            max_depth, lr, alpha, tolerance, temporal_features=True):
    cols = target_vector
    lag=max_lag
    n_trees = 200
    #start with a fresh df
    df = dat[cols].copy()
    if temporal_features and method != "VAR":
        if df.index.freq == "m":
            df["month"] = df.index.month
        else:
            df["week"] = df.index.week
            df["month"] = df.index.month
    # create objective matrix
    for col in cols:
        for l in range(1, lag+1):
            df[col+"(t-%s)" % l] = df[col].shift(l)
    df = df.dropna()
    X = df[df.columns.difference(cols)].values
    y = df[cols].values
    y_train, y_test = y[:-n_test], y[-n_test:]#make sanity check
    X_train, X_test = X[:-n_test], X[-n_test:]#make sanity check
    
    #Train 
    if method =="RF":
        RF = RandomForestRegressor(n_estimators=n_trees,
                                   max_depth = max_depth, n_jobs=-1)
        RF.fit(X_train, y_train)
        pred = RF.predict(X_test)
    elif method =="GB":
        GB = MultiOutputRegressor(GBoost(learning_rate=lr,
                                         n_estimators=n_trees,
                                         max_depth=max_depth), n_jobs=-1)
        GB.fit(X_train, y_train)
        pred = GB.predict(X_test)
    elif method =="VAR":
        VAR = Lasso(alpha=alpha, tol=tolerance)
        VAR.fit(X_train, y_train)
        pred = VAR.predict(X_test)
    else:
        raise ValueError("Method unknown.")

    true = y_test
    SE = (pd.DataFrame(true, columns=cols) - pd.DataFrame(pred, columns=cols))**2
    
    #RMSE per variable
    #np.sqrt(SE.mean())
    # Mean RMSE for all of the vars
    #np.sqrt(SE.mean()).mean()
    return SE

 
                                                                                             
def best_fit(method, data, params, test_size, start_training="2008"):
    global target_vector
    results = list()
    param_combinations = product(*list(params.values()))
    for combination in param_combinations:
        for city in data:
            combi = { list(params.keys())[i] : combination[i] for i in range(len(combination)) }
            combi["city"] = city

            dat_ = data[city][start_training:]
            SE = build_model(method=method, dat=dat_, max_lag=combi.get("max_lag"),
                             n_test=test_size,
                             max_depth=combi.get("max_depth"), lr=combi.get("lr"),
                             alpha=combi.get("alpha"), tolerance=combi.get("tolerance"))
            RMSE = np.sqrt(SE.mean())
            for x in target_vector:
                combi["RMSE(% s)" % x] = RMSE.get(x)
            combi["MRMSE"] = RMSE.mean()
            results.append(combi)    
            
    return pd.DataFrame(results)



def compare_models(data, model_params, test_size):
    model_results = {}
    #Model Param Search
    print("Fitting the linear Model...")
    df_var = best_fit("VAR", data, model_params["VAR"], test_size)
    print("Fitting the Random Forest...")
    df_rf = best_fit("RF", data, model_params["RF"], test_size)
    #Baseline Models
    res_bl1 = pd.DataFrame([baseline1(data, city,
                                             target_vector,
                                             test_size) for city in data])
    res_bl2 = pd.DataFrame([baseline2(data, city,
                                             target_vector,
                                             test_size) for city in data])
    for city in data:
        d_var = df_var[df_var["city"] == city].loc[df_var[df_var["city"] == city].MRMSE.idxmin()].to_dict()
        d_rf = df_rf[df_rf["city"] == city].loc[df_rf[df_rf["city"] == city].MRMSE.idxmin()].to_dict()
        del d_rf["city"], d_var["city"]
        
        d_rf["Model"] = "RF"
        d_var["Model"] = "LVAR"
       
        
        dat_ = [d_var, d_rf]
        
        BL1 = res_bl1[res_bl1["city"] == city].to_dict()
        BL2 = res_bl2[res_bl2["city"] == city].to_dict()
        
        dd = {"RMSE(%s)" % i : list(BL1[i].values())[0] for i in BL1 if i not in ["MRMSE", "city"]}
        dd["Model"] = "BL1"
        dd["MRMSE"] = list(BL1["MRMSE"].values())[0]
        dat_.append(dd)
        dd = {"RMSE(%s)" % i : list(BL2[i].values())[0] for i in BL2 if i not in ["MRMSE", "city"]}
        dd["MRMSE"] = list(BL2["MRMSE"].values())[0]
        dd["Model"] = "BL2"
        dat_.append(dd)
        
        model_results[city] = pd.DataFrame(dat_)
        model_results[city].index = model_results[city].Model
    
    return model_results

## Experiment Setup

# vector of interest
target_vector = ['create', 'modify','tag_add',
            'tag_del', 'tag_change','loc_change',
            'new_mapper']
## Conditional Performance Metrics

# The prediction error of a model depends on the following modeling conditions:
    # Frequency : Time Horizon (1w, 2w, 3w, 4w)
    # City 


# Performance vs Forecast Horizon
# Performance vs Forecast Horizon
def horizon_performance(city):
    return pd.DataFrame({"VAR" : [frequency[city].MRMSE.LVAR for frequency in [model_results_w, model_results_2w, model_results_3w, model_results_m]],
              "RF" : [frequency[city].MRMSE.RF for frequency in [model_results_w, model_results_2w, model_results_3w, model_results_m]],
              "BL1" : [frequency[city].MRMSE.BL1 for frequency in [model_results_w, model_results_2w, model_results_3w, model_results_m]],
              "BL2" : [frequency[city].MRMSE.BL2 for frequency in [model_results_w, model_results_2w, model_results_3w, model_results_m]]},
             index = ["1w", "2w", "3w", "4w"])

def city_performance(frequency, obj, normalize=True):
    obj_error = "RMSE(% s)" % obj
    if frequency == "w":
        model_results = model_results_w
        data = data_w
    elif frequency == "2w":
        model_results = model_results_2w
        data = data_2w
    elif frequency == "3w":
        model_results = model_results_3w
        data = data_3w
    elif frequency == "4w":
        model_results = model_results_m
        data = data_m
    if normalize:
        return pd.DataFrame({"LVAR" : [(model_results[city][obj_error]/(data[city][obj].quantile(0.75) - data[city][obj].quantile(0.25))).LVAR for city in cities],
                                     "RF" : [(model_results[city][obj_error]/(data[city][obj].quantile(0.75) - data[city][obj].quantile(0.25))).RF for city in cities],
                                     "BL1"  : [(model_results[city][obj_error]/(data[city][obj].quantile(0.75) - data[city][obj].quantile(0.25))).BL1 for city in cities],
                                     "BL2"  : [(model_results[city][obj_error]/(data[city][obj].quantile(0.75) - data[city][obj].quantile(0.25))).BL2 for city in cities]},index=cities)
    else:
        return pd.DataFrame({"LVAR" : [model_results[city][obj_error].LVAR for city in cities],
                             "RF" : [model_results[city][obj_error].RF for city in cities],
                             "BL1"  : [model_results[city][obj_error].BL1 for city in cities],
                             "BL2"  : [model_results[city][obj_error].BL2 for city in cities]},index=cities)

def compare_objective(model_results, obj):
    return pd.DataFrame({"LVAR" : [model_results[city][obj].LVAR for city in cities],
                                 "RF" : [model_results[city][obj].RF for city in cities],
                                 "BL1"  : [model_results[city][obj].BL1 for city in cities],
                                 "BL2"  : [model_results[city][obj].BL2 for city in cities]},
                                index=cities)


    
