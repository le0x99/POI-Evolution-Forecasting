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

for dat in [data_m, data_2w, data_3w, data_w]:
    del dat["Vienna"]
    
cities = list(data_m.keys())
## The baseline Models

# A) V_T ~ V_{T-1}

def baseline1(data, city, vec, n_test):
    dat = data[city][vec].copy()
    SE = (dat[-n_test:] - dat[-n_test:].shift(1) )**2
    RMSE = np.sqrt(SE.mean())
    res = RMSE.to_dict()
    res["MRMSE"] = RMSE.mean()
    res["city"] = city
    return res

#baseline1_results = pd.DataFrame([baseline1(data_w, city,
                                            # target_vector,
                                            # test_size) for city in data])

    

# B) V_T ~ AVG(V_{T-2 : T-1})
        
def baseline2(data, city, vec, n_test):
    dat = data[city][vec].copy()
    SE = (dat[-n_test:] - dat[-n_test:].shift(1).rolling(2).mean())**2
    RMSE = np.sqrt(SE.mean())
    res = RMSE.to_dict()
    res["MRMSE"] = RMSE.mean()
    res["city"] = city
    return res

#baseline2_results = pd.DataFrame([baseline2(city,
                                            # target_vector,
                                            # test_size) for city in data])

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

              
model_params = {"VAR" : {"alpha" : [1, 5, 15, 20], "tolerance" : [.15], "max_lag" : list(range(1, 5)) },
                "RF" :   {"max_depth" : [1, 10, 15, 20, None], "max_lag" : list(range(1, 5)) }
                }

model_results_m = compare_models(data_m, model_params, 12)
model_results_3w = compare_models(data_3w, model_params, 16)
model_results_2w = compare_models(data_2w, model_params, 24)
model_results_w = compare_models(data_w, model_params, 48)

## Conditional Performance Metrics

# The prediction error of a model depends on the following modeling conditions:
    # Frequency : Time Horizon (w, 2w, 3w, 4w)
    # City 


# Performance vs Forecast Horizon
def horizon_performance(city):
    return pd.DataFrame({"VAR" : [frequency[city].MRMSE.LVAR for frequency in [model_results_w, model_results_2w, model_results_3w, model_results_m]],
              "RF" : [frequency[city].MRMSE.RF for frequency in [model_results_w, model_results_2w, model_results_3w, model_results_m]],
              "BL1" : [frequency[city].MRMSE.BL1 for frequency in [model_results_w, model_results_2w, model_results_3w, model_results_m]],
              "BL2" : [frequency[city].MRMSE.BL2 for frequency in [model_results_w, model_results_2w, model_results_3w, model_results_m]]},
             index = ["1w", "2w", "3w", "4w"])


# Example 
horizon_performance("Paris").plot(style=".-")
horizon_performance("Amsterdam").plot(style=".-")

# Performance
def city_performance(frequency, obj):
    if frequency == "w":
        model_results = model_results_w
    elif frequency == "2w":
        model_results = model_results_2w
    elif frequency == "3w":
        model_results = model_results_3w
    elif frequency == "4w":
        model_results = model_results_m
        
    return pd.DataFrame({"LVAR" : [model_results[city][obj].LVAR for city in cities],
                                 "RF" : [model_results[city][obj].RF for city in cities],
                                 "BL1"  : [model_results[city][obj].BL1 for city in cities],
                                 "BL2"  : [model_results[city][obj].BL2 for city in cities]},index=cities)

# A look at the the models performances for weekly predictions
performance_w = city_performance("w", "MRMSE")
performance_w.plot.bar()
performance_w.boxplot(showmeans=True)
performance_w.describe()

# A look at the the models performances for 2w predictions
performance_2w = city_performance("2w", "MRMSE")
performance_2w.plot.bar()
performance_2w.boxplot(showmeans=True)
performance_2w.describe()

# A look at the the models performances for 3w predictions
performance_3w = city_performance("3w", "MRMSE")
performance_3w.plot.bar()
performance_3w.boxplot(showmeans=True)
performance_3w.describe()

# A look at the the models performances for 4w predictions
performance_4w = city_performance("4w", "MRMSE")
performance_4w.plot.bar()
performance_4w.boxplot(showmeans=True)
performance_4w.describe()

## Mean for all cities vs frequency
def aggregated_RMSE(objective="MRMSE", how="mean"):
    if how == "mean":
        return pd.DataFrame([city_performance(freq, objective).mean() for freq in ["w", "2w", "3w", "4w"]], index=["w", "2w", "3w", "4w"])
    elif how == "median":
        return pd.DataFrame([city_performance(freq, objective).median() for freq in ["w", "2w", "3w", "4w"]], index=["w", "2w", "3w", "4w"])
        
# Mean of all cities for the models and their relation to the frequency
aggregated_RMSE("MRMSE", "mean").plot(style=".-")
# Median of all cities for the models and their relation to the frequency
aggregated_RMSE("MRMSE", "median").plot(style=".-")

# Mean RMSE for new mapper in t+1 of all cities for the models and their relation to the frequency
aggregated_RMSE("RMSE(new_mapper)", "mean").plot(style=".-")
aggregated_RMSE('RMSE(create)', "mean").plot(style=".-")
aggregated_RMSE('RMSE(modify)', "mean").plot(style=".-")
aggregated_RMSE('RMSE(tag_add)', "mean").plot(style=".-")
aggregated_RMSE('RMSE(tag_del)', "mean").plot(style=".-")
aggregated_RMSE('RMSE(loc_change)', "mean").plot(style=".-")

# Action activity /in R
# Modify Activity /in R




## - - A) Monthly data experiment 

# First, obtain the baseline results for monthly data
# For each of the 20 cities, we use the last year (12M) to evaluate the performance in terms of RMSE

# Naive Predictor (BL1)

res_bl1_m = pd.DataFrame([baseline1(data_m, city,
                                             target_vector,
                                             test_size_m) for city in data_m])
res_bl1_m.index = res_bl1_m.city
res_bl1_m.MRMSE.plot.bar(grid=True);plt.title("MEAN % s" % res_bl1_m.MRMSE.mean())

# MA(2) Predictor (BL2)

res_bl2_m = pd.DataFrame([baseline2(data_m, city,
                                             target_vector,
                                             test_size_m) for city in data_m])
res_bl2_m.index = res_bl2_m.city
res_bl2_m.MRMSE.plot.bar(grid=True);plt.title("MEAN % s" % res_bl2_m.MRMSE.mean())
 
# Now for each of the model, search for an optimal set of parameters within a small set of possible params
# We dont want to overly hardcode the parameters since we are not using CV
res_var_m = best_fit("VAR", data_m, model_params["VAR"], test_size_m)
res_rf_m = best_fit("RF", data_m, model_params["RF"], test_size_m)


# Compare the performance of the models city wise

model_results_m = {}
df_rf = res_rf_m.copy()
df_var = res_var_m.copy()
for city in data_m:
    d_var = df_var[df_var["city"] == city].loc[df_var[df_var["city"] == city].MRMSE.idxmin()].to_dict()
    d_rf = df_rf[df_rf["city"] == city].loc[df_rf[df_rf["city"] == city].MRMSE.idxmin()].to_dict()
    del d_rf["city"], d_var["city"]
    
    d_rf["Model"] = "RF"
    d_var["Model"] = "LVAR"
   
    
    dat_ = [d_var, d_rf]
    
    BL1 = res_bl1_m[res_bl1_m["city"] == city].to_dict()
    BL2 = res_bl2_m[res_bl2_m["city"] == city].to_dict()
    
    dd = {"RMSE(%s)" % i : list(BL1[i].values())[0] for i in BL1 if i not in ["MRMSE", "city"]}
    dd["Model"] = "BL1"
    dd["MRMSE"] = list(BL1["MRMSE"].values())[0]
    dat_.append(dd)
    dd = {"RMSE(%s)" % i : list(BL2[i].values())[0] for i in BL2 if i not in ["MRMSE", "city"]}
    dd["MRMSE"] = list(BL2["MRMSE"].values())[0]
    dd["Model"] = "BL2"
    dat_.append(dd)
    
    model_results_m[city] = pd.DataFrame(dat_)
    model_results_m[city].index = model_results_m[city].Model

# Analysis of RMSE
metrics = ['MRMSE', 'RMSE(create)', 'RMSE(modify)',
           'RMSE(tag_add)',
           'RMSE(tag_del)', 'RMSE(tag_change)', 
           'RMSE(loc_change)', 'RMSE(new_mapper)']

for city in data_m:
    model_results_m[city][metrics].plot(color=["black","green", "blue", "orange","yellow","purple", "red", "grey"], kind="bar", grid=True)
    plt.title(city)
    
    
 
# Compare the MRMSE for the cities and their models

                            

compare_mrmse = compare_objective(model_results_m, "MRMSE")

## - - B) 3w data experiment 



## - - C) 2w data experiment 



## - - D) 1w data experiment 







    ## Analysis of RMSE
    metrics = ['MRMSE', 'RMSE(create)', 'RMSE(modify)',
               'RMSE(tag_add)',
               'RMSE(tag_del)', 'RMSE(tag_change)', 
               'RMSE(loc_change)', 'RMSE(new_mapper)']
    
    
    
    ## For each city, show performance of the 3 models
    for city in data:
        model_results[city][metrics].plot(color=["black","green", "blue", "orange","yellow","purple", "red", "grey"], kind="bar")
        plt.title(city)
        
        
    # Compare objectives  
    ## Compare the MRMSE for the cities and their models
    compare_mean = pd.DataFrame({"LVAR" : [model_results[city].MRMSE.LVAR for city in data],
                                 "RF" : [model_results[city].MRMSE.RF for city in data],
                                 "GB" : [model_results[city].MRMSE.GB for city in data],
                                 "BL1"  : [model_results[city].MRMSE.BL1 for city in data],
                                 "BL2"  : [model_results[city].MRMSE.BL2 for city in data]},
                                index=list(data.keys()))
                                
    
    
    compare_mean.plot.bar();plt.ylabel("$RMSE_{\mu}$")
    compare_mean.plot(kind="bar",grid=True);plt.ylabel("$RMSE_{\mu}$")
    
    ## For each city, show best params
    lvar_params = pd.DataFrame([ df[df["city"] == city].loc[df[df["city"] == city].MRMSE.idxmin()].to_dict() for city in data ])
    lvar_params.index = lvar_params.city;del lvar_params["city"]
    #lvar_params = lvar_params.reindex(["Heidelberg", "Sao Paulo", "Rome", "Stockholm", "Sydney", "New York", "Berlin", "Global"])
    
    
    # Best Models for each city and their parameters
    lvar_params
    
    
    # Compare single variable perofmrance
    for col in metrics:
        compare_col = pd.DataFrame({"LVAR" : [model_results[city][col].LVAR for city in data],
                                    "RF" : [model_results[city].MRMSE.RF for city in data],
                                     "BL1"  : [model_results[city][col].BL1 for city in data],
                                     "BL2"  : [model_results[city][col].BL2 for city in data]},
                                    index = data.keys())#.reindex(["Heidelberg", "Sao Paulo", "Rome", "Stockholm", "Sydney", "New York", "Berlin", "Global"])
        
        
        compare_col.plot.bar(grid=True);plt.ylabel(col)
                
        
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    