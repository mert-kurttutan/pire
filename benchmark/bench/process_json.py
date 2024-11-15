import json
import os
import pandas as pd

# use os agnostic path
BENCHMARK_FOLDER = "./benchmark_results"
BENCHMARK_IMAGE_FOLDER = "./benchmark_images"


def json_to_dataframe(json_path: str):

    with open(json_path, 'r') as f:
        data = json.load(f)

    if "Big" in data["dim_strategy"]:
        data["m"] = data["dim_strategy"]["Big"]
        data["n"] = data["dim_strategy"]["Big"]
        data["k"] = data["dim_strategy"]["Big"]
    elif "SmallM" in data["dim_strategy"]:
        data["m"] = data["dim_strategy"]["SmallM"]
        data["n"] = data["dim_strategy"]["Big"]
        data["k"] = data["dim_strategy"]["Big"]
    elif "SmallN" in data["dim_strategy"]:
        data["m"] = data["dim_strategy"]["Big"]
        data["n"] = data["dim_strategy"]["SmallN"]
        data["k"] = data["dim_strategy"]["Big"]
    elif "SmallK" in data["dim_strategy"]:
        data["m"] = data["dim_strategy"]["Big"]
        data["n"] = data["dim_strategy"]["Big"]
        data["k"] = data["dim_strategy"]["SmallK"]
    else:
        raise ValueError("dim_strategy not found")

    # take mean of times and crate times_mean, 
    # note times is list of list
    data['times_mean'] = [sum(x) / len(x) for x in data['times']]
    # now median
    data['times_median'] = [sorted(x)[len(x) // 2] for x in data['times']]
    # now min
    data['times_min'] = [min(x) for x in data['times']]
    # sorting key is important since we need to have a well defined equality between 
    # different bench_configs
    data['bench_config'] = json.dumps(data['bench_config'], sort_keys=True)
    del data['dim_strategy']
    del data['times']
    df = pd.DataFrame(data)
    df["gflops"] = 2 * df["m"] * df["n"] * df["k"] / df["times_min"] / 1e9
    return df

def plot_benchmark(run_idx: int):
    benchmar_run_folder_path = os.path.join(BENCHMARK_FOLDER, f"benchmark_run_{run_idx}")
    # go through all the json files and create a dataframe by merging them
    dataframes = []
    for file in os.listdir(benchmar_run_folder_path):
        if file.endswith(".json"):
            json_path = os.path.join(benchmar_run_folder_path, file)
            dataframe = json_to_dataframe(json_path)
            dataframes.append(dataframe)
    final_df = pd.concat(dataframes)
    # fix index
    final_df.reset_index(drop=True, inplace=True)

    # create a plot for each bench_name and
    # each plot has x as m, y as gflops
    # color as implementation
    # save figure as {bench_name}.png
    image_folder = os.path.join(BENCHMARK_IMAGE_FOLDER, f"benchmark_run_{run_idx}")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    for bench_name in final_df["bench_name"].unique():
        bench_df = final_df[final_df["bench_name"] == bench_name]
        plt.figure()
        # put dot on each point with line connecting and mark projection of point on x axis with vlines
        sns.lineplot(data=bench_df, x="m", y="gflops", hue="implementation", marker="o")
        # # now draw projection of point on x axis and mark with numerical value
        # for i in range(bench_df.shape[0]):
        #     plt.vlines(bench_df["m"].iloc[i], 0, bench_df["gflops"].iloc[i], linestyle="--", color="grey", alpha=0.5)
        #     plt.text(bench_df["m"].iloc[i], 0, str(bench_df["m"].iloc[i]), rotation=90, verticalalignment="bottom")

        plt.title(bench_name)
        plt.savefig(os.path.join(image_folder, f"{bench_name}_{run_idx}.png"))
        plt.close()

if __name__ == "__main__":
    # iteratoer thoruhg benchmark folder anc count how many run folders are there
    run_idx = 0
    while os.path.exists(os.path.join(BENCHMARK_FOLDER, f"benchmark_run_{run_idx}")):
        plot_benchmark(run_idx)
        run_idx += 1
