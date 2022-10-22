import numpy as np

start_year = 1985
end_year = 2025
bit = 14

year_to_idx=[(0 if i <start_year else i-start_year+1) for i in range(end_year)]
positional_encoding=np.zeros((end_year-start_year+1,bit),dtype=np.float)
_2i =  np.arange(0,bit,2)
pos = np.expand_dims(np.arange(0,end_year-start_year+1),1)

positional_encoding[:,0::2] = np.sin(pos/(1000**(_2i/bit)))
positional_encoding[:,1::2] = np.cos(pos/(1000**(_2i/bit)))