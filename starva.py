import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
from pandas.io.json import json_normalize
import boto3
from datetime import datetime

auth_url = "https://www.strava.com/oauth/token"
activites_url = "https://www.strava.com/api/v3/athlete/activities"

payload = {
    'client_id': "89356",
    'client_secret': '572ec4d45cdae9b496013483c89e95efd9008c43',
    'refresh_token': '518dbf193e8dfc1d22c725ebf4585952f266920d',
    'grant_type': "refresh_token",
    'f': 'json'
}

print("Requesting Token...\n")
res = requests.post(auth_url, data=payload, verify=False)
access_token = res.json()['access_token']
print("Access Token = {}\n".format(access_token))

header = {'Authorization': 'Bearer ' + access_token}
param = {'per_page': 200, 'page': 1}


def data_extraction():

    my_dataset = requests.get(activites_url, headers=header, params=param).json()

    activities = json_normalize(my_dataset)
    #Create new dataframe with only columns I care about
    cols = ['name', 'upload_id', 'type', 'distance', 'moving_time',   
            'average_speed', 'max_speed','total_elevation_gain',
            'start_date_local'
        ]
    activities = activities[cols]

    #Break date into start time and date
    activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
    activities['start_time'] = activities['start_date_local'].dt.time
    activities['start_date_local'] = activities['start_date_local'].dt.date
    # Save as csv
    return activities.to_csv('/Users/ajit/Desktop/ELT-strava/activities.csv', index= False)

def data_extractionwiths3():
    
    my_dataset = requests.get(activites_url, headers=header, params=param).json()

    activities = json_normalize(my_dataset)
    #Create new dataframe with only columns I care about
    cols = ['name', 'upload_id', 'type', 'distance', 'moving_time',   
            'average_speed', 'max_speed','total_elevation_gain',
            'start_date_local'
        ]
    activities = activities[cols]

    #Break date into start time and date
    activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
    activities['start_time'] = activities['start_date_local'].dt.time
    activities['start_date_local'] = activities['start_date_local'].dt.date
    # Save as csv
    activities.to_csv('/tmp/activities.csv', index= False)
    s3_resource = boto3.resource('s3')
    date = datetime.now()
    filename = f'{date.year}/{date.month}/{date.day}/stravaactivities.csv'
    response = s3_resource.Object('stravadataanalysis', filename).upload_file("/tmp/activities.csv")
    return response

def lambda_hanler(event, context):
    data_extractionwiths3()

if __name__ == '__main__':
    data= data_extractionwiths3()

# # # Load back in
# activities = pd.read_csv('activities.csv')

# print(activities.head(5))
