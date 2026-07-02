import numpy as np
data = np.load('data_dump/amazon_Appliances_complaints.npy', allow_pickle=True)
print(type(data), data.shape)
print(data[0])