# RSNA Intracranial Hemorrhage Detection

## Plan of action
- setup a GCP workspace (using [this guide](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/111029#latest-640644) as an example)
- try using fastai v2, with its native DICOM processing capabilities, based off of jhoward's notebooks from Kaggle (downloaded and saved in [](notebooks/jhoward))


## Submissions
- Forked v5 of jhowaards submission notebook
- made minor changes on number of epochs
- use downsampled TIF images for initial trining
- commited on Kaggle, which completed up to stage 'dcm-384-2' (running two epochs of DICOMs, downsampled to 384x384)
- saved s[1-4], dcm-384-[0-2] as a Kaggle dataset
- loaded 'dcm-384-2', used that in the 'Prepare for submission' section
- Currently commiting a version that runs 2 epochs of the full dataset

### Scores
- Submission based off of 'dcm-384-2' gives a score of 0.087 (currently top 50%)

## TODO
- pull jhoward's new notebooks that use JPG instead of TIFF
- use newer version of jhoward's notebook for submission, using fixed training vs. validation split and JPG images