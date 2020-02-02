import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import zscore 
plt.rcParams["figure.figsize"] = (8, 5.5)
#plt.rcParams["figure.dpi"] = 80

#load data
with open("data", "rb") as f:
    data = pickle.load(f)
    
  
## Description of the underlying data:
    
    # We collected data for $n$ cities.
    # For each of those cities, we searched for POIs, where a POI is defined
    # as "HAS_NAME", "HAS_AMENITY", "IS_NODE"
    # We acquired all POIs for each of the cities.
    # Next, we gathered all historic edits of these POIS.
    # Code for data acquisition : "acquire_data.py"
    
    # For each of the POIs, there is a set of historic entries.
    # We used this entries to derive the relevant information about what a mapper exactly did at an entry.
    # We did this very efficiently and without the need to query any changesets, since the information
    # regarding what exactly happend at an entry can be extracted with some basic coding.
    # Each of the historic entries are then indexed by it's timestamp.
    # For each of the entries, we derived the following information:
        
        # Action-Type : Create, Modify, Delete, Recreate
        # DataType : Binary , where only one aciton type can be 1 
    
        # Modify-Type : Tag_add, Tag_del, Tag_ch, loc_ch
        # DataType : Binary, where multiple modification types can occur per entry
        
    ## Examplified for entry $i$:
        
        # x_i = [timestamp, create, modify, delete, recreate, tag_add, ..., loc_ch]
        # x_i = [01.04.2007, 0, 1, 0, 0, 1, 1, 0 ,1]
    
    # After all entries were properly strucutred as described above, the data
    # can be resampled to make it a time series ( indexing by t using the timestamp )
    # We then chose weekly time index (others are possible) to create the series
    
    
    # The code can be inspected at "restructure_data.py"
    # Note that we cannot capture POIs which were completely deleted in the past, since
    # we are only looking at the evolution of POIs that exist in present.
    # Therefore, there are no POI deletions, but there are historic deletions followed by a "revive" of another mapper
    
    # -> This implies that the most intersting part of the data is the modification
    # behaviour of the mappers and the overall evolution of POIs for a city or globally
    
    
## Some Visualizations

# cumulative overall POI activity per city
for city in data:
    data[city].activity.cumsum().plot();plt.legend(list(data.keys())) 
    
# cumulative POIs for the cities (POI city evolution)
for city in data:
    data[city].create.cumsum().plot(grid=True);plt.legend(list(data.keys())) 

# Evolution of POI mapper participation
for city in data:
    data[city].new_mapper.cumsum().plot(grid=True);plt.legend(list(data.keys())) 

# Mappers per POI (Starting 2010)
for city in data:
    (data[city].new_mapper.cumsum()/data[city].create.cumsum())["2010":].plot(grid=True);plt.legend(list(data.keys()))

# cumulative Action types for New York
data["New York"][["create","modify","delete"]].cumsum().plot()

# cumulative Modify-type for New York
data["New York"][["tag_add", "tag_del", "tag_change", "loc_change"]].cumsum().plot()






### - Data Aggregation for further description

# There are multiple way to aggregate the data.
# One way is to create a global activity dataframe, which sums up the entries.
# By doing that, the information of where the entry comes from is lost.
# This approach may be interesting for additional descriptive statistics.
# E.g. looking at the POI Evolution globally as opposed to city wise.


# Aggregated time series

agg = sum(data.values()).dropna()

# Global Evolution

# weekly POI activity
agg.activity.plot()

# Cumulative activity
agg.activity.cumsum().plot()

# Global POI action behaviour
agg[["create", "recreate", "delete", "modify"]].plot()

# Cumulative

agg[["create", "recreate", "delete", "modify"]].cumsum().plot()

# Global delete vs. recreate
agg[["recreate", "delete"]].cumsum().plot()



## Global Distributions (***)

# Action type dist
agg[["create","modify","delete"]].plot.hist(stacked=True,bins=70, grid=True)

# Mod type dist
agg[["tag_add", "tag_del", "tag_change", "loc_change"]].plot.hist(stacked=True,bins=70, grid=True)

## Differenced data distributions

# Action type dist
agg[["create","modify","delete"]].diff().plot.hist(stacked=True,bins=70, grid=True)
agg[["create","modify"]].diff().plot.density(1.5, grid=True)

# Mod type dist
agg[["tag_add", "tag_del", "tag_change", "loc_change"]].diff().plot.hist(stacked=True,bins=70, grid=True)
agg[["tag_add", "tag_del", "tag_change", "loc_change"]].diff().plot.density(1.5, grid=True)


### Analysis of Temporal data properties (GLOBAL)

# Autocorrelation 

ACF = lambda X, max_lag : pd.Series([X.autocorr(lag) for lag in range(1, max_lag)])

# Show ACF for action types together
max_lag = 200
autocorr = pd.DataFrame({"Create" : ACF(agg.create, max_lag),
                         "Modify" : ACF(agg.modify, max_lag),
                         "Delete" : ACF(agg.delete, max_lag)})
autocorr.plot(grid=True)

# Show ACF for modify types together
max_lag = 100
autocorr = pd.DataFrame({'tag_del' : ACF(agg.tag_del, max_lag),
                         'tag_change' : ACF(agg.tag_change, max_lag),
                         "loc_change" : ACF(agg.loc_change, max_lag),
                         "tag_add" : ACF(agg.tag_add, max_lag)})
autocorr.plot(grid=True)

#Show ACF for new poi mappers
ACF(agg.new_mapper, 100).plot(grid=True)


## Partial Autocorrelation (The real deal)
from statsmodels.tsa.stattools import pacf as PACF

# The partial autocorrelation is basically just the partial correlation between 
# X_t and X_{t-L}, where Z contains all smaller lags than L
# For example, if we want to compute the partial correlation between X_t and X_{t-5}
# We want to remove confo, such that in the general case, partial autocorrelation
# for lag L has Z = { l : l > 0 , l < L}

# Show PACF for action types together
max_lag = 100
p_autocorr = pd.DataFrame({"Create" : PACF(agg.create, max_lag),
                         "Modify" : PACF(agg.modify, max_lag),
                         "Delete" : PACF(agg.delete, max_lag)})
p_autocorr.plot(grid=True)#interesting

# Show PACF for modify types together
max_lag = 100
p_autocorr = pd.DataFrame({'tag_del' : PACF(agg.tag_del, max_lag),
                         'tag_change' : PACF(agg.tag_change, max_lag),
                         "loc_change" : PACF(agg.loc_change, max_lag),
                         "tag_add" : PACF(agg.tag_add, max_lag)})
p_autocorr.plot(grid=True)#interesting









### Data Aggregation for Predictive Modeling

# Another approach would be to first release the datapoints from its time dependancy
# Additionally, each entry within the aggregated dataset needs to get a dummy indiciating the originating city
# More precisely, if we wish to model $y_t ~ X_{t-L}$, then each city is transformed
# in a way such that we have row vectors like this form :

            # [y_t, X_{t-1}, X_{t-2}, ..., X_{t-L}], where y_t is the target variable



# After this transformation, the data can be shuffled and aggregated in order to train models.

# In any case, the acquisition of addditional cities is highly desirable
# because for weekly indexed data and a total of 12 years, we got 365*12/7=625 data-
# points per city, which is insufficient for learning, at least when were talking
# about more flexible methods.

