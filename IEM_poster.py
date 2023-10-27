# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.stats import t

# Step 1: Load the dataset
# Assume you have a dataset stored in a CSV file named 'data.csv'

def _readcleandata(path):
    """    
    This function reads desired path as input paramter and returns the data 
    after cleaning data from the file by replacing blank data with 0 and 
    tranposing the values.
    """
    base=pd.read_csv(path)
    base.replace('..',0,inplace=True)
    base=base.fillna(0)
    baset=base
    baset1 = baset.transpose()
    return baset, baset1


data, data1 = _readcleandata('India_Metrics.csv')

"""
Below code plots different plots based upon the indicators and performs cluster
and curve fitting along with ploynomial regression

"""
X = data[['GDP growth (annual %)', 'InflationPercentage']]  
kmeans = KMeans(n_clusters=5)  # Define the number of clusters
kmeans.fit(X)
labels = kmeans.labels_

# Visualizing Clustering Results
plt.scatter(X['GDP growth (annual %)'], X['InflationPercentage'], c=labels)
plt.xlabel('GDP (annual %)')
plt.ylabel('InflationPercentage')
plt.title('Clustering Results of GDP growth & InflationPercentage')
plt.show()

#Fitting using Polynomial Regression
# The target variable to fit
y = data['Exports of goods & services (% of GDP)']  
# Select a single feature for simplicity
X = data[['Imports of goods & services (% of GDP)']]  
poly_features = PolynomialFeatures(degree=2)  # Use degree 2 polynomial
X_poly = poly_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

#Visualizing Fitting Results
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('Exports of goods & services')
plt.ylabel('Imports of goods & services')
plt.title('Polynomial Regression Fitting Results')
plt.show()

#plotting Line plot
plt.plot(data["Years"], 
         data['High-technology exports (% of manufactured exports)'], 
         label= 'High-technology exports %', 
         color = 'blue', marker="o")

# Labelling x and y axis
plt.xlabel("YEARS")
plt.ylabel("Percentage")
plt.grid(lw = 0.3, ls = "dashdot")
plt.xticks([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
plt.title("Percentage of High-technology exports") 
plt.legend(loc = "upper left", edgecolor = "black")
plt.show()

#Plotting grouped bar chat 

year = np.arange(len(data['Years']))
values = ["2021", "2020", "2019", "2018", "2017", "2016", "2015", "2014", 
              "2023", "2012"]
wid = 0.45
# Location on x-ais to plot the Bar chart
second_bar = [i+wid for i in year]       
Exp = plt.bar(year, data['GDP growth (annual %)'], 
              width = wid, label = "GDP growth (annual %)", color = "red")
plt.bar_label(Exp, labels = data['GDP growth (annual %)'],
              label_type = 'edge', padding = 3)
Imp = plt.bar(second_bar, data['Gross capital formation (% of GDP)'],
              width = wid, label = "Gross capital formation (% of GDP)",  
              color = "green")
plt.bar_label(Imp, labels = data['Gross capital formation (% of GDP)'],
              label_type = 'edge', padding = 3)

# Labelling x and y axis
plt.xlabel("Years")
plt.ylabel("Percentage")
plt.xticks(year+wid*0.5, values)
plt.ylim(-10, 50)
plt.grid(axis ='y', lw = 0.3, ls = "dashdot")
plt.title("Percentage of GDP & Gross capital formation", fontstyle = "italic")
plt.legend(loc = "upper left", edgecolor = "black")
    
