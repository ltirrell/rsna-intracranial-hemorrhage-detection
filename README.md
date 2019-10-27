# RSNA Intracranial Hemorrhage Detection

## Plan of action
- setup a GCP workspace (using [this guide](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/111029#latest-640644) as an example)
- try using fastai v2, with its native DICOM processing capabilities, based off of jhoward's notebooks from Kaggle (downloaded and saved in [](notebooks/jhoward))


## Submissions
- Forked v11 of jhowaards submission notebook
- made minor changes on number of epochs etc
- use downsampled JPEG images for initial training
- commited on Kaggle, which completed up to stage 'dcm-256-bs64-ep2' (running two epochs of DICOMs, downsampled to 256x256)
- saved initial trainings, downsamped to 96x96, 160x160, and 256x256
- loaded 'dcm-256-bs64-ep2', used that in the 'Prepare for submission' section
- Currently commiting a version that runs 2 epochs of the dataset at 384x384, then at full size (512x512)

### Scores
- Submission based off of 'dcm-256-bs64ep2' gives a score of 0.082 (currently top 50%)

## TODO
- Get things up and running on GCP, so can play around with params etc more
