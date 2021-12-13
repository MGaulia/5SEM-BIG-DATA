from itertools import combinations
from mpi4py import MPI
import numpy as np
import pandas as pd

def get_correlation(data):
    names = data["names"]
    data = data["data"]
    corr = round( np.corrcoef(data[0], data[1])[0, 1], 3)
    return (names[0], names[1], corr)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # n cores

if rank == 0:
    df = pd.read_csv("data.csv")  # CREATE COMBINATIONS
    companies = list(df.columns)
    companies = list(combinations(companies, 2))
    data = [{"names": (compone, comptwo), "data": (np.array((df[compone])), np.array((df[comptwo])))} for
            compone, comptwo in companies]
    #data = data[:3305]
    data = [i for i in np.array_split(data, size)]
else:
    data = None

data = comm.scatter(data, root=0)

#print('Procesas {}:'.format(rank), set(sum([[i["names"][0], i["names"][1] ]for i in data ],[])))

result = [get_correlation(part) for part in data]

all = comm.gather(result, root=0)

if rank == 0:
    result = sum(all, [])
    result = pd.DataFrame(result, columns=['stock_one', 'stock_two', "corr"])
    result = result.sort_values(by=['corr'])
    print(result)
    result.to_csv('result.txt', header=None, index=None, sep=' ', mode='a')
