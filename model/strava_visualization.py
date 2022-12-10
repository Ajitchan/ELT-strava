import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

activities = pd.read_csv('/Users/ajit/Desktop/ELT-strava/activities.csv')
# Looking for Coorelation
dropNumColumn = ['name','type','start_date_local','start_time']
ndata = activities.drop(dropNumColumn, axis = 1)
corr_matrix = ndata.corr()
print(corr_matrix['distance'].sort_values(ascending = False))

def correlation_matrix_plot():
    dropNumColumn = ['name','type','start_date_local','start_time']
    ndata = activities.drop(dropNumColumn, axis = 1)
    # dataset.plot(kind = 'scatter', x = 'rot_per',y = 'diameter', alpha = 0.6)
    import seaborn as sns
    #dataset.info()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_data = ndata.select_dtypes(include=numerics)
    #num_data.info()
    plt.subplots(figsize=(15,12))
    sns.heatmap(num_data.corr(),annot=True,annot_kws={'size':10})
    return plt.show()
    #num_data.corr()

# plt.figure(figsize = (8,8))
# ax = sns.regplot(x="distance", y="total_elevation_gain",data= activities, scatter_kws={"color": "black"}, line_kws={"color": "red"})  
# plt.show()

# plt.figure(figsize = (8,8))
# ax = sns.regplot(x="average_heartrate", y="distance",data= activities, scatter_kws={"color": "black"}, line_kws={"color": "red"})  
# plt.show()

# distance = activities['distance']
# average_heartrate = activities['average_heartrate']


# plt.scatter( average_heartrate,distance, cmap='summer',
#             edgecolor='black', linewidth=1, alpha=1)

# plt.xlabel('average_heartrate')
# plt.ylabel('distance')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

