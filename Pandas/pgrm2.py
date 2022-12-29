#Write a python program to generate the series of dates from 1st May,2021 to 12th May,2021 (both inclusive)

import pandas as pd
date_series = pd.date_range(start = '05-01-2021', end = '05-12-2021')

print(date_series)