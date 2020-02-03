# This file contains methodology for extracting and cleaning
# 2018 survey data from IPUMS USA
# Credit to @ameasure on GitHub for sharing the extracting protocol

%run ipums_lib.py
rows = row_generator(datapath = "usa_00201.dat", ddipath = "usa_00201.xml")

# Extract mapped data into list, count observations
data = []
for index, row in enumerate(rows):
    data.append(row)
print(len(data))

# Import necessary packages
import pandas as pd
import numpy as np

# Create dataframe
df = pd.DataFrame(data=data)

# Save raw data
df.to_csv('ipums_raw.csv', index=False)

# All data are objects, convert to numeric where applicable
df = df.apply(pd.to_numeric, errors='ignore')

# Keep only interesting variables
df = df.apply(pd.to_numeric, errors='ignore')

# After further inspection,
# Drop variables contributing little information
df = df.drop(columns = ['POVERTY', "FERTYR", "WIDINYR", "DIVINYR", "COUNTYFIP", "CITY", "BPL"])

# Create new variables from existing variables
df['AGEMARR'] = df['YRMARR']-df['BIRTHYR']
df['PCT_HHINC'] = df.INCTOT/df.HHINCOME*100 # see adjustments below
df['PCT_MTG_INC'] = df.MORTAMT1/ (df.HHINCOME/12)

# Simplify classifications
df.RACE = df.RACE.map({1:1, 2:2, 3:3, 4:4, 5:4, 6:4, 7:5, 8:5, 9:5})
df.HISPAN = df.HISPAN.map({0:0, 1:1, 2:2, 3:2, 4:2, 9:3})
df.MORTGAGE = df.MORTGAGE.map({0:0, 1:1, 3:2, 4:2})

def occupation(x):
  if x == 0:
    return np.nan
  elif x > 0 and x <= 960:
    return 'mgmt, biz, fin'
  elif x > 1000 and x <= 1980:
    return 'comp, eng, sci'
  elif x > 2000 and x <= 2920:
    return 'educ, legal, comm srvs, arts, media'
  elif x >= 3000 and x <= 3550:
    return 'healthcare'
  elif x > 2600 and x <= 4655:
    return 'service'
  elif x >= 4700 and x <= 4965:
    return 'sales and related'
  elif x >= 5000 and x <= 5940:
    return 'office and admin'
  elif x > 6000 and x <= 6130:
    return 'farm, fish, forrestry'
  elif x >= 6200 and x <=6950:
    return 'constr, extract'
  elif x >= 7000 and x<= 7640:
    return 'instal, maint, repair'
  elif x >= 7700 and x <= 8990:
    return 'production'
  elif x > 9000 and x <= 9920:
    return 'transpo and material moving'
  else:
    return 'other'

df['OCC_BROAD'] = df['OCC'].apply(occupation)

def ancestry(x):
  if x == 0:
    return np.nan
  elif x > 0 and x <= 99:
    return 'western european'
  elif x >= 100 and x <= 179:
    return 'eatern european'
  elif x >= 181 and x <= 195:
    return 'european, general'
  elif x >= 200 and x <= 296:
    return 'hispanic'
  elif x >= 300 and x <= 337:
    return 'west indies'
  elif x >= 360 and x <= 380:
    return 'non-hispanic south and central amer'
  elif x >= 400 and x <= 496:
    return 'north africa and southwest asia'
  elif x >= 500 and x <= 599:
    return 'subsaharan africa'
  elif x >= 600 and x <=695:
    return 'south asia'
  elif x >= 700 and x <= 796:
    return 'other asia'
  elif x >= 800 and x <= 870:
    return 'pacific'
  elif x > 900 and x <= 994:
    return 'north american'
  else:
    return 'other'

df['ANCESTR1'] = df['ANCESTR1'].apply(ancestry)

# Check for unknowns
df = df[df["VETSTAT"] != 0.0]
df[df["MIGRATE1"] == 9].shape
df = df[df['EMPSTAT'] != '0']

# Deal with NaN values
df.OCC_BROAD.fillna(value='other', inplace=True)
for index, value in enumerate(df.HHINCOME):
    if value != 0:
        df['PCT_HHINC'][index] = df.INCTOT[index]/value*100
        df['PCT_MTG_INC'][index] = df.MORTAMT1[index]/ (value/12) * 100
    else:
        df['PCT_HHINC'][index] = 100
        df['PCT_MTG_INC'][index] = 100

# Save cleaned data
df.to_csv('ipums_1.csv', index=False)
