# target_folder
target_folder = 'target/criterion/bbb'

bench_names = ["f32-glar-gemm", "f32-mkl-gemm"]

# now look at the file names at each target_folder+bench_name and extract them as integer to arrays for only first one
# since the others must have the same structure of subfolders

dim_array = []

import os.path as path
import os

# iterate subfoler of target_folder+bench_name
full_path = path.join(target_folder, bench_names[0])
for dirs in os.scandir(full_path):
    # extract the last part of the path
    # if name is report then skip
    if dirs.name == "report":
        continue
    dim = int(dirs.name.split('-')[-1])
    dim_array.append(dim)

# sort the array
dim_array.sort()
print(dim_array)

# now we have the dimensions in dim_array
import json

result_dict = {}
for bench_name in bench_names:
    # iterate the bench_names
    result_arr = []
    for dim_dir in dim_array:
        full_path = path.join(target_folder, bench_name, f"{dim_dir}")
        # now open full_path + base + estimates.json
        with open(path.join(full_path, "base", "estimates.json")) as f:
            base_estimates = json.load(f)
            median_nanosec = base_estimates["median"]["point_estimate"]
            median_sec = median_nanosec / 1e9
            result_arr.append(median_sec)


    result_dict[bench_name] = result_arr

print(result_dict)

def gflops_compute(dim, time):
    return 2 * dim ** 3 / time / 1e9

# now we have the csv file
# now we can plot the graph
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd

# create 2 dataframes where each row is a benchmark type and each column is a dimension and the value type, gflops and time
df = pl.DataFrame(result_dict)
# df = df.with_columns(pl.Series("dim", dim_array))
df = df.with_columns(pl.Series(name="dim", values=dim_array)) 
df = df.with_columns(
    gflops_compute(pl.col("dim"), pl.col("f32-glar-gemm")).alias("glar-gflops")
)
df = df.with_columns(
    gflops_compute(pl.col("dim"), pl.col("f32-mkl-gemm")).alias("mkl-gflops")
)
# plot the graph matplotlib
df = df.to_pandas()
graph = df.plot(x="dim", y=["glar-gflops", "mkl-gflops"])
graph.set_xlabel("Dimension")
graph.set_ylabel("GFLOPS")
# save the graph
plt.savefig("sgemm-gflops.png")