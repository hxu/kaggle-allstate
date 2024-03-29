import pandas as pd
import numpy as np
from classes import *
data=pd.read_csv("../train-2.csv")
def rollup_data(x):
    '''Use with apply function on dataframe to create one row for each customer ID with all values of lagged variables'''
    variables={}
    variables['customer_ID']=[x['customer_ID'].iloc[0]]
    for possible_prior in range(1,13):
        for prior_var in x.keys()[3:]:
            if len(x[prior_var][x['shopping_pt']==possible_prior])!=0:
                variables[prior_var+str(possible_prior)]=[x[prior_var][x['shopping_pt']==possible_prior].iloc[0]]
            else:
                variables[prior_var+str(possible_prior)]=[np.nan]
    A=pd.DataFrame(variables)
    A.to_csv("data/rolled_up1.csv",header=False,index=False,mode='a')
    print variables['customer_ID']
    print A
    return
return_pd=truncate(data).groupby('customer_ID').apply(rollup_data)
#return_pd.to_csv("data/rolled_up.csv",index=False)
