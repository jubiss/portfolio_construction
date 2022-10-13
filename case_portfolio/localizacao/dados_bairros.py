import pandas as pd

df = pd.read_csv('gps_data.csv')
lat_long = df.values
from geopy.geocoders import  GoogleV3

geolocator = GoogleV3(api_key='googl_api')
new_info = []
new_locations = []
for i in range(len(lat_long)):
    bairro = -1
    location = geolocator.reverse(lat_long[i])
    for j in location.raw.get('address_components'):
        if 'sublocality' in j.get('types'):
            bairro = j.get('long_name')
    if bairro == -1:
        bairro = location.raw.get('address_components')[0].get('long_name')
    new_info.append([location.address,bairro,location.point])
    new_locations.append(location)
geo_info = pd.DataFrame(new_info, columns=['endereço', 'bairro', 'point_location'])
geo_info['longitude'] = df['longitude']
geo_info['latitude'] = df['latitude']
