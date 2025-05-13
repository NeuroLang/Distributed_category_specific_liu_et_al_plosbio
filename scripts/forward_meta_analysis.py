# %%
import os
import pathlib

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from neurolang.utils.server import engines
from nilearn import plotting

# %%

folder = pathlib.Path(os.getcwd())
output_folder = folder / "output"
if not output_folder.exists():
    output_folder.mkdir()
nl = engines.NeurosynthEngineConf(folder).create()

# %%

query_digit_not_letter = """
VoxelReported (x, y, z, study) :- PeakReported(x2, y2, z2, study) & Voxel(x, y, z) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d < 4)
TermInStudy(term, study) :- TermInStudyTFIDF(term, tfidf, study)  & (tfidf > 0.01)
StudyOfInterest(study) :-  TermInStudy("digit", study) & ~TermInStudy("letter", study)
Activation(x, y, z) :- SelectedStudy(s) & VoxelReported(x, y, z, s)
ActivationGivenBoth(x, y, z, PROB) :- Activation(x, y, z) // (StudyOfInterest(s) & SelectedStudy(s))

Image(agg_create_region_overlay(x, y, z, p)) :- ActivationGivenBoth(x, y, z, p)
"""

query_not_digit_letter = """
VoxelReported (x, y, z, study) :- PeakReported(x2, y2, z2, study) & Voxel(x, y, z) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d < 4)
TermInStudy(term, study) :- TermInStudyTFIDF(term, tfidf, study)  & (tfidf > 0.01)
StudyOfInterest(study) :-  ~TermInStudy("digit", study) & TermInStudy("letter", study)
Activation(x, y, z) :- SelectedStudy(s) & VoxelReported(x, y, z, s)
ActivationGivenBoth(x, y, z, PROB) :- Activation(x, y, z) // (StudyOfInterest(s) & SelectedStudy(s))

Image(agg_create_region_overlay(x, y, z, p)) :- ActivationGivenBoth(x, y, z, p)
"""

# %%

queries = [
    (query_digit_not_letter, "query_digit_not_letter_meta.nii.gz"),
    (query_not_digit_letter, "query_not_digit_letter_meta.nii.gz"),
]

# %%
for query, fname in queries:
    print(fname)
    with nl.scope as e:
        nl.execute_datalog_program(query)
        res = nl.solve_all()

        active_region = (res["Image"].as_pandas_dataframe().iloc[0, 0]).spatial_image()
        nii = nib.Nifti1Image(active_region.get_fdata(), active_region.affine)
        nii.to_filename(str(output_folder / fname))


# %%
for _, fname in queries:
    fname = str(output_folder / fname)
    nii = nib.load(fname)
    data = nii.get_fdata()
    thr = np.percentile(data[data > 0], 95)
    plotting.plot_stat_map(nii, threshold=thr, title=fname, display_mode="mosaic")
    plt.savefig(fname.replace(".nii.gz", ".png"))
