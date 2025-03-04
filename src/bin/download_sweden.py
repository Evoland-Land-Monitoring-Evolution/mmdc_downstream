import os
import warnings
from pathlib import Path

import openeo

warnings.filterwarnings("ignore")

connection = openeo.connect("https://openeo.vito.be/openeo/1.2").authenticate_oidc()

launched_jobs = [job["title"] for job in connection.list_jobs()]


# folder = "/home/kalinichevae/Documents/BioMass/images/Sweden_all/"
folder = "/home/kalinichevae/scratch_data/BioMass/images/Sweden_all"

for job in connection.list_jobs():
    title = job["title"]

    job_id = job["id"]
    job = connection.job(job_id)
    status = job.status()

    tile, year = title.split("_")[-2:]
    export_folder = os.path.join(folder, year, tile)

    if int(year):
        if (
            "Sweden_S1_ASC" in title
            and status == "finished"
            and not Path(export_folder).isdir()
        ):
            results = job.get_results()
            print(title)

            if (
                not Path(export_folder).exists()
                and not Path(os.path.join(export_folder, "job-results.json")).exists()
            ):
                Path(export_folder).mkdir(exist_ok=True, parents=True)
                print(export_folder)
            results.download_files(export_folder)
