# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:25:15 2019

@author: HPH
"""

import pandas as pd

f = pd.read_excel('geo_data.xlsx')


result = []

for i in range(len(f)):
    latitude = f.iloc[i]['user_latitude']
    longitude = f.iloc[i]['user_longitude']
    d = {"coord":[longitude, latitude], "elevation":1}
    result.append(d)
    
import json
j = open('data.json', 'w', encoding='utf-8')
j.write(json.dumps(result))