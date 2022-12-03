import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
from pandas.io.json import json_normalize
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
my_dataset = requests.get(activites_url, headers=header, params=param).json()

activities = json_normalize(my_dataset)

# Save as csv
activities.to_csv('/Users/ajit/Desktop/ELT-strava/activities.csv')

# # Load back in
# activities = pd.read_csv('activities.csv')

