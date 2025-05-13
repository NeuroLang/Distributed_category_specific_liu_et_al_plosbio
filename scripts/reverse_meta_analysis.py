import subprocess
import os
from pathlib import Path

output_folder = Path(os.getcwd()) / "output"
if not output_folder.exists():
    output_folder.mkdir()

data_folder = Path(os.getcwd()) / "data"
print(data_folder)
for file in data_folder.glob("*.nii"):
    command = (
        "b2rio "
        f"--brain_path {file} "
        f"--output_file {output_folder / file.stem} "
        f"--output_summary {output_folder / file.stem}_summary"
    )
    print("Running command:\n", command)
    subprocess.call(command, shell=True)
    print("=" * 20)
