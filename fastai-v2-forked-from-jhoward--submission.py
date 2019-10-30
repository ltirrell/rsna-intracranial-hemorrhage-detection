#!/usr/bin/env python
# coding: utf-8

# In the notebook "[Cleaning the data for rapid prototyping](https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai)" I showed how to create a small, fast, ready-to-use dataset for prototyping our models. The dataset created in that notebook, along with the metadata files it uses, are now [available here](https://www.kaggle.com/jhoward/rsna-hemorrhage-jpg).
# 
# So let's use them to create a model! In this notebook we'll see the whole journey from pre-training using progressive resizing on our prototyping sample, through to fine-tuning on the full dataset, and then submitting to the competition.
# 
# In my testing overnight with this notebook on my local machine I was seeing scores that would land towards the top of the leaderboard with a single model, with just some minor tweaking. I'm intentionally not doing any tricky modeling in this notebook, because I want to show the power of simple techniques and simples architectures. You should take this as a starting point and experiment! e.g. try data augmentation methods, architectures, preprocessing approaches, using the DICOM metadata, and so forth...
# 
# We'll be using the fastai.medical.imaging library here - for more information about this see the notebook [Some DICOM gotchas to be aware of](https://www.kaggle.com/jhoward/some-dicom-gotchas-to-be-aware-of-fastai). We'll also use the same basic setup that's in the notebook.

# I've used up my GPU hours for the week, so I've commented out all the cells that use GPU. I'll run them tomorrow and update the notebook then.

# In[1]:


# !pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null
# !pip install git+https://github.com/fastai/fastai_dev@2e45c62dcc84ee2d4f38c1be80b54b9855c29f64'
# !pip install git+https://github.com/fastai/fastai_dev > /dev/null


# In[2]:


from fastai2.basics           import *
from fastai2.vision.all       import *
from fastai2.medical.imaging  import *
from fastai2.callback.tracker import *

np.set_printoptions(linewidth=120)
matplotlib.rcParams['image.cmap'] = 'bone'


# In[3]:


get_ipython().system(' ls ~/rsna_data')


# First we read in the metadata files (linked in the introduction).

# In[4]:


path = Path('/home/tirrell_le/rsna_data/')
path_trn = path/'stage_1_train_images'
path_tst = path/'stage_1_test_images'

path_inp = Path('/home/tirrell_le/jhoward_data')
path_xtra = path_inp
path_meta = path_xtra/'meta'/'meta'
path_jpg = path_xtra/'train_jpg'/'train_jpg'


# In[5]:


df_comb = pd.read_feather(path_meta/'comb.fth').set_index('SOPInstanceUID')
df_tst  = pd.read_feather(path_meta/'df_tst.fth').set_index('SOPInstanceUID')
df_samp = pd.read_feather(path_meta/'wgt_sample.fth').set_index('SOPInstanceUID')
bins = (path_meta/'bins.pkl').load()


# ## Train vs valid

# To get better validation measures, we should split on patients, not just on studies, since that's how the test set is created.
# 
# Here's a list of random patients:

# In[6]:


set_seed(42)
patients = df_comb.PatientID.unique()
pat_mask = np.random.random(len(patients))<0.8
pat_trn = patients[pat_mask]


# We can use that to take just the patients in a dataframe that match that mask:

# In[7]:


def split_data(df):
    idx = L.range(df)
    mask = df.PatientID.isin(pat_trn)
    return idx[mask],idx[~mask]

splits = split_data(df_samp)


# Let's double-check that for a patient in the training set that their images are all in the first split.

# In[8]:


df_trn = df_samp.iloc[splits[0]]
p1 = L.range(df_samp)[df_samp.PatientID==df_trn.PatientID[0]]
assert len(p1) == len(set(p1) & set(splits[0]))


# ## Prepare sample DataBunch

# We will grab our sample filenames for the initial pretraining.

# In[9]:


def filename(o): return os.path.splitext(os.path.basename(o))[0]

fns = L(list(df_samp.fname)).map(filename)
fn = fns[0]
fn


# We need to create a `DataBunch` that contains our sample data, so we need a function to convert a filename (pointing at a DICOM file) into a path to our sample JPEG files:

# In[10]:


def fn2image(fn): return PILCTScan.create((path_jpg/fn).with_suffix('.jpg'))
fn2image(fn).show();


# We also need to be able to grab the labels from this, which we can do by simply indexing into our sample `DataFrame`.

# In[11]:


htypes = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
def fn2label(fn): return df_comb.loc[fn][htypes].values.astype(np.float32)
fn2label(fn)


# If you have a larger GPU or more workers, change batchsize and number-of-workers here:

# In[12]:


bs,nw = 128,8


# We're going to use fastai's new [Transform Pipeline API](http://dev.fast.ai/pets.tutorial.html) to create the DataBunch, since this is extremely flexible, which is great for intermediate and advanced Kagglers. (Beginners will probably want to stick with the Data Blocks API). We create two transform pipelines, one to open the image file, and one to look up the label and create a tensor of categories.

# In[13]:


torch.cuda.is_available()


# In[14]:


tfms = [[fn2image], [fn2label,EncodedMultiCategorize(htypes)]]
dsrc = DataSource(fns, tfms, splits=splits)
nrm = Normalize(tensor([0.6]),tensor([0.25]))
aug = aug_transforms(p_lighting=0.)
batch_tfms = [IntToFloatTensor(), nrm, Cuda(), *aug]


# To support progressive resizing (one of the most useful tricks in the deep learning practitioner's toolbox!) we create a function that returns a dataset resized to a requested size:

# In[15]:


def get_data(bs, sz):
    return dsrc.databunch(bs=bs, num_workers=nw, after_item=[ToTensor],
                          after_batch=batch_tfms+[AffineCoordTfm(size=sz)])


# Let's try it out!

# In[16]:


dbch = get_data(128, 96)
xb,yb = to_cpu(dbch.one_batch())
dbch.show_batch(max_n=4, figsize=(9,6))
xb.mean(),xb.std(),xb.shape,len(dbch.train_dl)


# Let's track the accuracy of the *any* label as our main metric, since it's easy to interpret.

# In[17]:


def accuracy_any(inp, targ, thresh=0.5, sigmoid=True):
    inp,targ = flatten_check(inp[:,0],targ[:,0])
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()


# The loss function in this competition is weighted, so let's train using that loss function too.

# In[18]:


def get_loss(scale=1.0):
    loss_weights = tensor(2.0, 1, 1, 1, 1, 1).cuda()*scale
    return BaseLoss(nn.BCEWithLogitsLoss, pos_weight=loss_weights, floatify=True, flatten=False, 
        is_2d=False, activation=torch.sigmoid)

loss_func = get_loss(0.14*2)  #scaled due to resampling
opt_func = partial(Adam, wd=0.01, eps=1e-3)
metrics=[accuracy_multi,accuracy_any]


# Now we're ready to create our learner. We can use mixed precision (fp16) by simply adding a call to `to_fp16()`!

# In[19]:


def get_learner():
    dbch = get_data(128,128)
    learn = cnn_learner(dbch, resnet101, loss_func=loss_func, opt_func=opt_func, metrics=metrics)
    return learn.to_fp16()


# In[20]:


learn = get_learner()


# Leslie Smith's famous LR finder will give us a reasonable learning rate suggestion.

# In[26]:


lrf = learn.lr_find()


# ## Pretrain on sample

# Here's our main routine for changing the size of the images in our DataBunch, doing one fine-tuning of the final layers, and then training the whole model for a few epochs.

# In[21]:


cbs = []
def do_fit(bs,sz,epochs,lr, freeze=True,epochs_frozen=1):
    learn.dbunch = get_data(bs, sz)
    if freeze:
        if learn.opt is not None: learn.opt.clear_state()
        learn.freeze()
        learn.fit_one_cycle(epochs_frozen, slice(lr), cbs=cbs)
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr), cbs=cbs)

# Now we can train at 3 different sizes.

# In[28]:


do_fit(128, 96, 3, 2e-2)
learn.save('initial-96-bs128ep1,3')
lrf


# In[31]:


learn.lr_find()


# In[32]:


do_fit(128, 160, 4, 1e-3)
learn.save('initial-160-bs128ep1,4')
learn.lr_find()


# In[35]:


do_fit(128, 256, 5, 1e-3, epochs_frozen=2)
learn.save('initial-256-bs128ep2,5')
learn.lr_find()


# ## Scale up to full dataset

# Now let's fine tune this model on the full dataset. We'll need all the filenames now, not just the sample.

# In[22]:


fns = L(list(df_comb.fname)).map(filename)
splits = split_data(df_comb)  # use full dataset


# These functions are copied nearly verbatim from our [earlier cleanup notebook](https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai), so have a look there for details.

# In[23]:


def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


# In[24]:


def dcm_tfm(fn): 
    fn = (path_trn/fn).with_suffix('.dcm')
    try:
        x = fn.dcmread()
        fix_pxrepr(x)
    except Exception as e:
        print(fn,e)
        raise SkipItemException
    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))
    px = x.scaled_px
    return TensorImage(px.to_3chan(dicom_windows.brain,dicom_windows.subdural, bins=bins))


# In[25]:


dcm = dcm_tfm(fns[0])
show_images(dcm)
dcm.shape


# We have some slight changes to our data source

# In[26]:


tfms = [[dcm_tfm], [fn2label,EncodedMultiCategorize(htypes)]]
dsrc = DataSource(fns, tfms, splits=splits)
batch_tfms = [nrm, Cuda(), *aug]


# In[27]:


def get_data(bs, sz):
    return dsrc.databunch(bs=bs, num_workers=nw, after_batch=batch_tfms+[AffineCoordTfm(size=sz)])


# Now we can test it out:

# In[28]:


dbch = get_data(64,256)
x,y = to_cpu(dbch.one_batch())
dbch.show_batch(max_n=4)
x.shape


# In[29]:


# remove sample scaling
learn.loss_func = get_loss(1.0)


# For fine-tuning the final layers, we don't really need to use a whole epoch, so we'll use the `ShortEpochCallback` to just train for 10% of an epoch, before then unfreezing the model and training a bit more.

# In[30]:


def fit_tune(bs, sz, epochs, lr, freeze=True,epochs_frozen=1):
    dbch = get_data(bs, sz)
    learn.dbunch = dbch
    learn.opt.clear_state()
    if freeze:
        learn.freeze()
        learn.fit_one_cycle(epochs_frozen, slice(lr), cbs=ShortEpochCallback(pct=0.2))
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr))


# In[45]:


fit_tune(64, 192, 3, (1e-3/3))
learn.save('dcm-192-bs64ep1,3')
learn.lr_find()


# In[ ]:


fit_tune(64, 256, 3, 3e-4, epochs_frozen=2)
learn.save('dcm-256-bs64ep2,3')
learn.lr_find()


# In[33]:


import gc; gc.collect()
torch.cuda.empty_cache()


# In[34]:


learn.load('dcm-256-bs64ep2,3')  # had to restart kernel


# In[35]:


#@# ran this to create dcm-384-bs64-ShortEpoch0pt2-2,2.pth, seems to do the wrong thing
#   (short epochs on all runs)
def fit_tune2(bs, sz, epochs, lr, freeze=True,epochs_frozen=1):
    dbch = get_data(bs, sz)
    learn.dbunch = dbch

    if freeze:
        learn.opt.clear_state()
        learn.freeze()
        learn.fit_one_cycle(epochs_frozen, slice(lr), cbs=ShortEpochCallback(pct=0.2))
    
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr))




def fit_tune3(bs, sz, epochs, lr):
    """ jhoward's exact def"""
    dbch = get_data(bs, sz)
    learn.dbunch = dbch
    learn.opt.clear_state()
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr))
                                                    
# In[ ]:


fit_tune2(64, 384, 2, 3e-4, epochs_frozen=2)
learn.save('dcm-384-bs64-ShortEpoch0pt2-2,2')
#learn.lr_find()

# In[ ]:

# #@# stopped and need to restart the kerne;
# #@# re-ran from the beginning, except GPU stuff, then loaded from here
learn.load('dcm-384-bs64-ShortEpoch0pt2-2,2')

fit_tune3(64, 384, 3, 3e-4)
learn.save('dcm-384-bs64--ep3')

"""
epoch     train_loss  valid_loss  accuracy_multi  accuracy_any  time                                     
0         0.079415    0.081572    0.973997        0.949942      3:38:27                      
1         0.072198    0.079009    0.974389        0.949687      3:38:33                             
2         0.067840    0.077374    0.975158        0.951617      3:38:51 
"""

# stopped, then running in jupyter lab to see lr
learn.load('dcm-384-bs64--ep3')
fit_tune3(64, 384, 1, 3e-4)
learn.save('dcm-384-bs64--ep4')

"""                                                                                          
epoch     train_loss  valid_loss  accuracy_multi  accuracy_any  time    
3         0.067134    0.077271    0.975314        0.952148
"""

# In[ ]:
learn.load('dcm-384-bs64--ep4')
fit_tune(32, 512, 2, 3e-4, epochs_frozen=2)
learn.save('dcm-512-bs32ep1,2')
fit_tune3(32, 512, 1, 3e-4)
learn.save('dcm-512-bs32--ep3')


# ## Prepare for submission

# Now we're ready to submit. We can use the handy `test_dl` function to get an inference `DataLoader` ready, then we can check it looks OK.

# In[32]:


test_fns = [(path_tst/f'{filename(o)}.dcm').absolute() for o in df_tst.fname.values]
print(len(test_fns) * 6)


# In[33]:


# dbch = get_data(32,512)
dbch = get_data(64,256)


# In[34]:


tst = test_dl(dbch, test_fns)
x = tst.one_batch()[0]
x.min(),x.max()


# We pass that to `get_preds` to get our predictions, and then clamp them just in case we have some extreme values.

# In[35]:


preds,targs = learn.get_preds(dl=tst)
preds_clipped = preds.clamp(.00001, .9999)


# I'm too lazy to write a function that creates a submission file, so this code is stolen from Radek, with minor changes.

# In[36]:


ids = []
labels = []

for idx,pred in zip(df_tst.index, preds_clipped):
    for i,label in enumerate(htypes):
        ids.append(f"{idx}_{label}")
        predicted_probability = '{0:1.10f}'.format(pred[i].item())
        labels.append(predicted_probability)


# In[37]:


df_csv = pd.DataFrame({'ID': ids, 'Label': labels})
df_csv.to_csv(f'submission.csv', index=False)
df_csv.head()


# In[38]:


df_csv


# Run the code below if you want a link to download the submission file.

# In[39]:


from IPython.display import FileLink, FileLinks
FileLink('submission.csv')


# In[40]:


len(df_csv)


# In[41]:





# In[ ]:





# In[ ]:




