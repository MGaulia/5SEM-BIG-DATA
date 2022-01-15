from itertools import combinations
from mpi4py import MPI
import numpy as np
import pandas as pd

def get_correlation(data):
    # Function used to calulcate correlation for 2 given stocks
    names = data["names"]
    data = data["data"]
    corr = round( np.corrcoef(data[0], data[1])[0, 1], 3)
    return (names[0], names[1], corr)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # Number of cores to be used

if rank == 0:
    # Parent core who distributes and collects data
    # Reading data
    df = pd.read_csv("data.csv")
    companies = list(df.columns)
    # Creating every combination of stock pairs
    companies = list(combinations(companies, 2))
    # Preparing data for computing
    data = [{"names": (compone, comptwo), "data": (np.array((df[compone])), np.array((df[comptwo])))} for
            compone, comptwo in companies]
    # Splitting data into n as evenly as possible divided chunks
    data = [i for i in np.array_split(data, size)]
else:
    data = None

# Scattering data to all processes
data = comm.scatter(data, root=0)

# Optional printing to see which core gets which stocks
#print('Procesas {}:'.format(rank), set(sum([[i["names"][0], i["names"][1] ]for i in data ],[])))

# Calculating correlations for given stocks
result = [get_correlation(part) for part in data]

# Gathering all calculations
all = comm.gather(result, root=0)

if rank == 0:
    # Transforming from list of lists to list
    result = sum(all, [])
    # Forming dataset from results
    result = pd.DataFrame(result, columns=['stock_one', 'stock_two', "corr"])
    # Sorting by correlation coefficient values in non-descending order
    result = result.sort_values(by=['corr'])
    # Printing result
    print(result)
    # Outputing result to file
    result.to_csv('result.txt', header=None, index=None, sep=' ', mode='a')