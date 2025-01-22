"""
@author: James
Program to remove images before they reach the actual algorithm. To do this, the weather from Hertford is
retrieved (closest place to Bayfordbury that gave accurate weather readings), and the forecast between
sunrise and sunset for the upcoming night in question is retrieved. 
"""
# import the modules
import python_weather
import asyncio
import os
import datetime as dt
import csv

##################################################################################### Weather #####################################################################################
# get weather data

async def get_cloud_data():
  """
  DESCRIPTION
  -----------
  Async function that will get the cloud coverage using the selected weather module. The idea is that its run once per day and gets the data for that night, saving it to
  a csv file. It just uses set values for the times of interest, as using sunrise and sunset cuts out a lot of useful weather data. Its set up to be more useful in winter,
  which is when it is anticipated to be more useful.

  PARAMETERS
  -----------
  NONE

  RETURNS
  -----------
  CSV file with cloud data for that night
  """
  # sets up the client for the weather library
  async with python_weather.Client(unit=python_weather.METRIC) as client:

    # the closest location to bayfordbury that will give actual data (using bayfordbury here gives wildly inaccurate values for some reason)
    weather = await client.get('Hertford')
    
    date = dt.date.today()                                              # getting the date it runs, so that just that night's data is added to the file

    with open("Cloud_cover.csv", "a") as outfile:                       # for now, as theres no archive of cloud weather data, this is left as append, in practice it can be
                                                                        # changed to write, to prevent the file from taking up too much storage. Need an archive so that
                                                                        # the algorithm can be tested with images accross the year, rather than just across a few days. 

      for daily in weather.daily_forecasts:                             # get the weather forecast for the next 3 days

        diff = daily.date - date

        if diff.days <= 1:                                              # removes the unwanted day at the end of the forecast
          for hourly in daily.hourly_forecasts:
              
              if diff.days == 0:                                        # covers the day of adding data
                if hourly.time >= dt.time(15, 0):                       # latest sunsets are at around 4, this is the closest weather data to that time
            
                  outfile.write(f"{daily.date},{hourly.time},{hourly.cloud_cover}\n") #writes to fille
              
              elif diff.days == 1:                                      # covers after midnight
                if hourly.time <= dt.time(9, 0):                        # latest sunrise is around 9, so this is the closest weather data to that time

                  outfile.write(f"{daily.date},{hourly.time},{hourly.cloud_cover}\n")
  
###################################################################################################################################################################################

class removeOutliers():

  def __init__(self, date, time, exp_time):
    self.date = date
    self.time = time
    self.exp_time = exp_time

  def get_weather_data(self):
    """
    DESCRIPTION
    -----------
    Runs the get_cloud_data function

    PARAMETERS
    -----------
    NONE

    RETURNS
    -----------

    """
    if __name__ == "__main__":

    #stops asyncio from breaking
      asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) 

      #runs the function
      asyncio.run(get_cloud_data())

################################################################################## Check clouds ##################################################################################

  def check_clouds(self):
    """
    DESCRIPTION
    -----------
    Function to get the data from the archive created with the get_cloud_data function. It will run with every image input into the algorithm.

    PARAMETERS
    -----------
    date : str
        Taken from the FIT header of the image
    time : str
       Taken from the FIT header of the image

    RETURNS
    -----------
    .....
    """

    date = self.date
    time = self.time

    with open("Cloud_cover.csv", "r") as outfile:
    
      data = csv.DictReader(outfile)
      weather_dict = {}

      for row in data:

        if row["date"] == date:
          weather_dict.update({dt.datetime.strptime(row[" time"], "%H:%M:%S"):int(row[" cloud cover"])})

    if weather_dict == {}:
      return True, "no data"
        
    time1 = dt.datetime(2024, 7, 2, 0, 0, 0)
    time2 = dt.datetime(2024, 7, 2, 1, 30, 0)

    time_diff = time2 - time1                                           # establishing a time difference for the next section. Only worked with datetime objects, but date irrelevent
                                                                        # as it cancels out
  
    for key in weather_dict:

      if key.time() == dt.time(0, 0, 0) and (dt.time(22, 30, 0) <= time.time() or dt.time(1, 30, 0) >= time.time()):
        weather = weather_dict[key]

      elif key - time <= time_diff:
        weather = weather_dict[key]

    if weather >= 40:
      return False, "too cloudy" # stops algorithm running if cloud cover is high enough
  
    else:
      return True, "data not indicative"

################################################################################## Exposure time ##################################################################################

  def check_expsure_time(self):
    """
    DESCRIPTION
    -----------
    Checks the exposure time of the AllSky image, and if its indicative of full cloud coverage, will prevent the algorithm from running, under the assumption theres no point

    PARAMETERS
    -----------
    expos_tim : str
             Exposure time from the FITs header of the image
  
    RETURNS
    -----------
    run_algo : bool
           Will be True if the exposure time indicates the algorithm needs to be run, and False otherwise
    """

    exp_time = self.exp_time

    if float(exp_time) >= 30:                                            # checks exposure time and changes run_algo and reason if needed
      run_algo = False # don't run as likely no clouds
      reason = "too high"


    elif float(exp_time) <= 1:
      run_algo = False # don't run as likely too many clouds
      reason = "too low"

  
    return run_algo, reason
  