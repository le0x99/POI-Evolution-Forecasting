import osmapi
import random
import time
import requests
import datetime
from tqdm import tqdm
import pickle
from copy import deepcopy as copy
from pprint import pprint as pp
api = osmapi.OsmApi()
import pandas as pd

# This script acquires the static (present) POI data as well as the historic edits for each of the POIS.


def get_pois(area_id):
    overpass_url = "http://overpass-api.de/api/interpreter"
    #["addr:street"]["addr:housenumber"]
    overpass_query = """
[out:json][timeout:1000];
area(%s)->.searchArea;
(
  node["name"]["amenity"](area.searchArea);
);
out body;
>;
out skel qt;


""" % area_id
    response = requests.get(overpass_url, params = {"data" : overpass_query})
    data = response.json()
    elements = data["elements"]
    #Modify attributes
    for poi in elements:
        del poi["type"]
        poi["location"] = (poi["lat"], poi["lon"])
        del poi["lat"]
        del poi["lon"]
    return elements


cities = {"Heidelberg" : 3600285864, "Rome" : 3600041485, "Berlin" : 3600062422, "Sao Paulo" : 3600298285, "New York" : 3600175905, 
          "Sydney" : 3605750005, "Stockholm" : 3600398021, 
          "Amsterdam" : 3600271110, "Paris" : 3600007444,
           "Barcelona" : 3600347950,
          "London" : 3600065606, "Los Angeles" : 3600207359,
          "Oslo" : 3600406091, "Mexico City" : 3601376330,
          "Shanghai" : 3600913067,
          "Vienna" : 3600109166,
          "Moscow" : 3602555133,
          "Montreal" : 3601634158,
          "Mumbai" : 3607888990,
          "Washington DC" : 3600162069,
          "San Francisco" : 3600111968
          
          }
          
          
pois = {}
for city in cities:
    if city not in pois:
        print("Getting POIs from %s" % city)
        pois[city] = get_pois(cities[city])


#load already collected data (only needed if the acquisition process was stopped before)
with open("histories", "rb") as f:
    histories = pickle.load(f)
    
# Collect all data
for city in pois:
    if len(histories[city]) != 0:
        continue
    print("Getting historic data for %s" % city)
    for poi in tqdm(pois[city]):
        sleeper = random.choice([0,.2,.12,0,.25,.5,.33])
        hist = api.NodeHistory(NodeId=poi["id"])
        time.sleep(sleeper)
        histories[city].append(hist)
    print("Saving..")
    with open("histories", "wb") as f:
        pickle.dump(histories, f)
        
        

        