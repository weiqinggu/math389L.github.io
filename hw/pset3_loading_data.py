# this is an example of how you might load the data to perform the regression
# task we are asking of you. This file details all the regression features
# and loads the data into A and b so you can run the regression methods.
import numpy as np
import pandas as pd

bikes = pd.read_csv('../data/bikes.csv')
A = np.stack((
    np.ones((bikes.shape[0])),     # intercept term
    (bikes.season == 1).values,    # season 1
    (bikes.season == 2).values,    # season 2
    (bikes.season == 3).values,    # season 3
    (bikes.season == 4).values,    # season 4
    bikes.holiday.values,          # is it a holiday?
    bikes.workingday.values,       # is it a working day?
    (bikes.hr == 0).values,        # it is midnight hour
    (bikes.hr == 1).values,        # it is 1:00am
    (bikes.hr == 2).values,        # it is 2:00am
    (bikes.hr == 3).values,        # it is 3:00am
    (bikes.hr == 4).values,        # it is 4:00am
    (bikes.hr == 5).values,        # it is 5:00am
    (bikes.hr == 6).values,        # it is 6:00am
    (bikes.hr == 7).values,        # it is 7:00am
    (bikes.hr == 8).values,        # it is 8:00am
    (bikes.hr == 9).values,        # it is 9:00am
    (bikes.hr == 10).values,       # it is 10:00am
    (bikes.hr == 11).values,       # it is 11:00am
    (bikes.hr == 12).values,       # it is 12:00pm
    (bikes.hr == 13).values,       # it is 1:00pm
    (bikes.hr == 14).values,       # it is 3:00pm
    (bikes.hr == 16).values,       # it is 4:00pm
    (bikes.hr == 17).values,       # it is 5:00pm
    (bikes.hr == 18).values,       # it is 6:00pm
    (bikes.hr == 19).values,       # it is 7:00pm
    (bikes.hr == 20).values,       # it is 8:00pm
    (bikes.hr == 21).values,       # it is 9:00pm
    (bikes.hr == 22).values,       # it is 10:00pm
    (bikes.hr == 23).values,       # it is 11:00pm
    bikes.atemp.values,            # percieved temperature
    (bikes.weathersit == 1).values,# clear/partly cloudy
    (bikes.weathersit == 2).values,# mist
    (bikes.weathersit == 3).values,# light snow/rain/thunder
    (bikes.weathersit == 4).values,# heavy
    bikes.windspeed.values,
), axis=1)
b = np.log1p(bikes.cnt.values) # log(1 + number of bikeshare users that hour)
