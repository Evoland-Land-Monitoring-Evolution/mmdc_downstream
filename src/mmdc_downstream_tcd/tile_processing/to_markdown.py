import os

import pandas as pd

# WORK = os.environ["WORK"]
# folder_results = WORK + "/results/MMDC/TABLES/"

folder_results = "/home/kalinichevae/jeanzay/results/MMDC/TABLES/"

csv_files = [c for c in os.listdir(folder_results) if c.endswith(".csv")]

for csv_file in csv_files:
    df = pd.read_csv(os.path.join(folder_results, csv_file))
    with open(os.path.join(folder_results, f"{csv_file.split('.')[0]}.md"), "w") as md:
        df.to_markdown(buf=md)
