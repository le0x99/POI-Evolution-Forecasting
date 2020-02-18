import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy as copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["figure.dpi"] = 200
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

# vector of interest
target_vector = ['create', 'modify','tag_add',
            'tag_del', 'tag_change','loc_change',
            'new_mapper']

                
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
    #res["MRMSE"] = RMSE.mean()
    res["city"] = city
    return res

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
    #res["MRMSE"] = RMSE.mean()
    res["city"] = city
    return res
    
def model_predictions(frequency, city, intercept=True, testing_years=1):
    if frequency == "1w":
        data = data_w[city]
        test_size = 48 
        max_lag = 4
    elif frequency == "2w":
        data = data_2w[city]
        test_size = 24
        max_lag = 2
    elif frequency == "3w":
        test_size = 16
        data = data_3w[city]
        max_lag = 2
    elif frequency == "4w":
        data = data_m[city]
        test_size = 12
        max_lag = 2
    
    test_size = test_size*testing_years
    cols = ['create', 'modify','tag_add',
            'tag_del', 'tag_change','loc_change',
            'new_mapper']
    #1 Obtain baseline predictions and errors
    pred_BL1 = data[-test_size-1:].shift(1).dropna()
    
    #2 Obtain MA predictions
    pred_BL2 = data[-test_size-2:].shift(1).rolling(2).mean().dropna()
    
    
    #2 Fit the statistical models
    # create objective matrix
    df = data.copy()
    for col in cols:
        for l in range(1, max_lag+1):
            df[col+"(t-%s)" % l] = df[col].shift(l)
            
    df = df.dropna()
    df_rf = df.copy()
    #add additional temporal features for the RF
    df_rf["week"] = df_rf.index.week
    df_rf["month"] = df_rf.index.month
    X_rf = df_rf[df_rf.columns.difference(cols)].values
    X = df[df.columns.difference(cols)].values
    y = df[cols].values
    y_train, y_test = y[:-test_size], y[-test_size:]
    X_train, X_test = X[:-test_size], X[-test_size:]
    X_train_rf, X_test_rf = X_rf[:-test_size], X_rf[-test_size:]
    
    #Train VAR
    VAR = LinearRegression(fit_intercept=intercept)
    VAR.fit(X_train, y_train)
    pred_VAR = VAR.predict(X_test)
    #Train RF
    RF = RandomForestRegressor(n_estimators=200,
                               max_depth = None, n_jobs=-1)
    RF.fit(X_train_rf, y_train)
    pred_RF = RF.predict(X_test_rf)
    return {"True" : pd.DataFrame(y_test, columns=cols, index= df[-test_size:].index),
            "BL1" : pd.DataFrame(pred_BL1, columns=cols, index= df[-test_size:].index),
            "BL2" : pd.DataFrame(pred_BL2, columns=cols, index= df[-test_size:].index), 
            "VAR" : pd.DataFrame(pred_VAR, columns=cols, index= df[-test_size:].index), 
            "RF" : pd.DataFrame(pred_RF, columns=cols, index= df[-test_size:].index) }
            

def compare_predictions(variable, frequency, city):
    res = model_predictions(frequency, city)
    return pd.DataFrame({method : res[method][variable] for method in res}, index = res["True"].index)
    
    
  - - # # Run the actual experiment # # - -  
# City wise training and evaluation
res_dicts = []
for freq in ["1w", "2w", "3w", "4w"]:
    d = {}
    for city in cities:
        res = model_predictions(freq, city)
        RMSE_rf = np.sqrt(((res["True"] - res["RF"])**2).mean())
        RMSE_var = np.sqrt(((res["True"] - res["VAR"])**2).mean())
        RMSE_bl1 = np.sqrt(((res["True"] - res["BL1"])**2).mean())
        RMSE_bl2 = np.sqrt(((res["True"] - res["BL2"])**2).mean())
        df = pd.DataFrame([_.to_list() for _ in [RMSE_var, RMSE_rf, RMSE_bl1, RMSE_bl2]],
                               columns = target_vector,
                               index=["VAR", "RF", "BL1", "BL2"])
        df.index.name = "Model"
        #df["MVE"] = df.mean(axis=1)
        #df["SVE"] = df.sum(axis=1)
        d[city] = df
    res_dicts.append(d)
    
# Aggregate the results into a large dataframe in order to do analysis of the conditional performance of the models
# This is a trivial task and just makes plotting easier
def aggregate(res_w, res_2w, res_3w, res_4w):
    city, frequency, target, error, model = [], [], [], [], []
    targets = ["create", "modify", "tag_add", "tag_del", "tag_change", "loc_change", "new_mapper"]
    for c in res_w:
        model_results = res_w[c]
        for var in targets:
            for i in range(len(model_results[var])):
                error.append(model_results[var].values[i])
                model.append(model_results[var].index[i])
                target.append(var)
                city.append(c)   
                frequency.append("1w")
    for c in res_2w:
        model_results = res_2w[c]
        for var in targets:     
            for i in range(len(model_results[var])):
                error.append(model_results[var].values[i])
                model.append(model_results[var].index[i])
                target.append(var)
                city.append(c)   
                frequency.append("2w")
    for c in res_3w:
        model_results = res_3w[c]
        for var in targets:
            for i in range(len(model_results[var])):
                error.append(model_results[var].values[i])
                model.append(model_results[var].index[i])
                target.append(var)
                city.append(c)   
                frequency.append("3w")
    for c in res_4w:
        model_results = res_4w[c]
        for var in targets:
            for i in range(len(model_results[var])):
                error.append(model_results[var].values[i])
                model.append(model_results[var].index[i])
                target.append(var)
                city.append(c)   
                frequency.append("4w")
                
    return pd.DataFrame({"City" : city,
                         "Target" : target, "Error" :error,
                        "Frequency" : frequency, "Model" : model})
  
  
 # Save results frequency and city wise
res_dicts = []
for freq in ["1w", "2w", "3w", "4w"]:
    d = {}
    for city in cities:
        res = model_predictions(freq, city, testing_years=2)
        RMSE_rf = np.sqrt(((res["True"] - res["RF"])**2).mean())
        RMSE_var = np.sqrt(((res["True"] - res["VAR"])**2).mean())
        RMSE_bl1 = np.sqrt(((res["True"] - res["BL1"])**2).mean())
        RMSE_bl2 = np.sqrt(((res["True"] - res["BL2"])**2).mean())
        df = pd.DataFrame([_.to_list() for _ in [RMSE_var, RMSE_rf, RMSE_bl1, RMSE_bl2]],
                               columns = target_vector,
                               index=["VAR", "RF", "BL1", "BL2"])
        df.index.name = "Model"
        #df["MVE"] = df.mean(axis=1)
        #df["SVE"] = df.sum(axis=1)
        d[city] = df
    res_dicts.append(d)
    
# Restructure
model_results_w = res_dicts[0]
model_results_2w = res_dicts[1]
model_results_3w = res_dicts[2]
model_results_m = res_dicts[3]

# Make a normalized version of the results
model_results_w_norm = { city : model_results_w[city] / model_results_w[city].loc["BL1"] for city in model_results_w  }
model_results_2w_norm = { city : model_results_2w[city] / model_results_2w[city].loc["BL1"] for city in model_results_2w  }
model_results_3w_norm = { city : model_results_3w[city] / model_results_3w[city].loc["BL1"] for city in model_results_3w  }
model_results_m_norm = { city : model_results_m[city] / model_results_m[city].loc["BL1"] for city in model_results_m  }

data = aggregate(model_results_w, model_results_2w, model_results_3w, model_results_m)
data_norm = aggregate(model_results_w_norm, model_results_2w_norm, model_results_3w_norm, model_results_m_norm) 
 
#save results
with open("results_normalized", "wb") as f:
    pickle.dump(data_norm, f)
with open("results_absolute", "wb") as f:
    pickle.dump(data, f)
    
