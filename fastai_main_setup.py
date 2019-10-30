from fastai2.basics           import *
from fastai2.vision.all       import *
from fastai2.medical.imaging  import *
from fastai2.callback.tracker import *
from fastai2.callback.all     import *

np.set_printoptions(linewidth=120)
matplotlib.rcParams['image.cmap'] = 'bone'

path = Path('/home/tirrell_le/rsna_data/')
path_trn = path/'stage_1_train_images'
path_tst = path/'stage_1_test_images'

path_inp = Path('/home/tirrell_le/jhoward_data')
path_xtra = path_inp
path_meta = path_xtra/'meta'/'meta'
path_jpg = path_xtra/'train_jpg'/'train_jpg'

df_comb = pd.read_feather(path_meta/'comb.fth').set_index('SOPInstanceUID')
df_tst  = pd.read_feather(path_meta/'df_tst.fth').set_index('SOPInstanceUID')
df_samp = pd.read_feather(path_meta/'wgt_sample.fth').set_index('SOPInstanceUID')
bins = (path_meta/'bins.pkl').load()

set_seed(42)
patients = df_comb.PatientID.unique()
pat_mask = np.random.random(len(patients))<0.8
pat_trn = patients[pat_mask]

def split_data(df):
    idx = L.range(df)
    mask = df.PatientID.isin(pat_trn)
    return idx[mask],idx[~mask]

# JPG training
splits = split_data(df_samp)
df_trn = df_samp.iloc[splits[0]]
p1 = L.range(df_samp)[df_samp.PatientID==df_trn.PatientID[0]]
assert len(p1) == len(set(p1) & set(splits[0]))


def filename(o): return os.path.splitext(os.path.basename(o))[0]

fns = L(list(df_samp.fname)).map(filename)
fn = fns[0]
fn

def fn2image(fn): return PILCTScan.create((path_jpg/fn).with_suffix('.jpg'))
fn2image(fn).show();

htypes = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
def fn2label(fn): return df_comb.loc[fn][htypes].values.astype(np.float32)
fn2label(fn)

bs,nw = 128,8
# torch.cuda.is_available()

tfms = [[fn2image], [fn2label,EncodedMultiCategorize(htypes)]]
dsrc = DataSource(fns, tfms, splits=splits)
nrm = Normalize(tensor([0.6]),tensor([0.25]))
aug = aug_transforms(p_lighting=0.)
batch_tfms = [IntToFloatTensor(), nrm, Cuda(), *aug]


def get_data(bs, sz):
    return dsrc.databunch(bs=bs, num_workers=nw, after_item=[ToTensor],
                          after_batch=batch_tfms+[AffineCoordTfm(size=sz)])


dbch = get_data(128, 96)
xb,yb = to_cpu(dbch.one_batch())
xb.mean(),xb.std(),xb.shape,len(dbch.train_dl)


# metrics/loss_func
def accuracy_any(inp, targ, thresh=0.5, sigmoid=True):
    inp,targ = flatten_check(inp[:,0],targ[:,0])
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()


def get_loss(scale=1.0):
    loss_weights = tensor(2.0, 1, 1, 1, 1, 1).cuda()*scale
    return BaseLoss(nn.BCEWithLogitsLoss, pos_weight=loss_weights, floatify=True, flatten=False, 
        is_2d=False, activation=torch.sigmoid)

loss_func = get_loss(0.14*2)  #scaled due to resampling
opt_func = partial(Adam, wd=0.01, eps=1e-3)
metrics=[accuracy_multi,accuracy_any]


def get_learner():
    dbch = get_data(128,128)
    learn = cnn_learner(dbch, resnet101, loss_func=loss_func, opt_func=opt_func, metrics=metrics)
    return learn.to_fp16()

learn = get_learner()

cbs = []
def do_fit(bs,sz,epochs,lr, freeze=True,epochs_frozen=1):
    learn.dbunch = get_data(bs, sz)
    if freeze:
        if learn.opt is not None: learn.opt.clear_state()
        learn.freeze()
        learn.fit_one_cycle(epochs_frozen, slice(lr), cbs=cbs)
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr), cbs=cbs)


# train on JPGS, commented out for now
# do_fit(128, 96, 3, 2e-2)
# learn.save('initial-96-bs128ep1,3')
# do_fit(128, 160, 4, 1e-3)
# learn.save('initial-160-bs128ep1,4')
# do_fit(128, 256, 5, 1e-3, epochs_frozen=2)
# learn.save('initial-256-bs128ep2,5')
# learn.lr_find()

#=============================================================
# train on full data

fns = L(list(df_comb.fname)).map(filename)
splits = split_data(df_comb)  # use full dataset


def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

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

dcm = dcm_tfm(fns[0])
dcm.shape


tfms = [[dcm_tfm], [fn2label,EncodedMultiCategorize(htypes)]]
dsrc = DataSource(fns, tfms, splits=splits)
batch_tfms = [nrm, Cuda(), *aug]

def get_data(bs, sz):
    return dsrc.databunch(bs=bs, num_workers=nw, after_batch=batch_tfms+[AffineCoordTfm(size=sz)])

dbch = get_data(64,256)
x,y = to_cpu(dbch.one_batch())
x.shape


learn.loss_func = get_loss(1.0)


# For fine-tuning the final layers, we don't really need to use a whole epoch, so we'll use the `ShortEpochCallback` to just train for 20% of an epoch, before then unfreezing the model and training a bit more.
def fit_tune(bs, sz, epochs, lr, freeze=True,epochs_frozen=1):
    dbch = get_data(bs, sz)
    learn.dbunch = dbch
    learn.opt.clear_state()
    if freeze:
        learn.freeze()
        learn.fit_one_cycle(epochs_frozen, slice(lr), cbs=ShortEpochCallback(pct=0.2))
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr))
    
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

# ---------    
# #@# start training
# fit_tune(64, 192, 3, (1e-3/3))
# learn.save('dcm-192-bs64ep1,3')
# fit_tune(64, 256, 3, 3e-4, epochs_frozen=2)
# learn.save('dcm-256-bs64ep2,3')

# learn.load('dcm-256-bs64ep2,3')  # had to restart kernel


# fit_tune2(64, 384, 2, 3e-4, epochs_frozen=2)
# learn.save('dcm-384-bs64-ShortEpoch0pt2-2,2')

# learn.load('dcm-384-bs64-ShortEpoch0pt2-2,2')
# fit_tune3(64, 384, 3, 3e-4)
# learn.save('dcm-384-bs64--ep3')
"""               
epoch     train_loss  valid_loss  accuracy_multi  accuracy_any  time 
0         0.079415    0.081572    0.973997        0.949942      3:38:27  
1         0.072198    0.079009    0.974389        0.949687      3:38:33       
2         0.067840    0.077374    0.975158        0.951617      3:38:51 
"""

# learn.load('dcm-384-bs64--ep3')
# fit_tune3(64, 384, 1, 3e-4)
# learn.save('dcm-384-bs64--ep4')
""" 
epoch     train_loss  valid_loss  accuracy_multi  accuracy_any  time
3         0.067134    0.077271    0.975314        0.952148          
"""

#---
# Scale up to full submission
def get_learner_full():
    dbch = get_data(32,512)
    learn = cnn_learner(dbch, resnet101, loss_func=loss_func, opt_func=opt_func, metrics=metrics)
    return learn.to_fp16()
learn = get_learner_full()

learn.load('dcm-384-bs64--ep4')
dbch = get_data(32, 512)
lr=1e-4
learn.dbunch = dbch
learn.opt.clear_state()
learn.unfreeze()
learn.fit_one_cycle(1, slice(lr))
learn.save('dcm-512-bs32--ep1')
create_submission('512-ep1')
learn.fit_one_cycle(1, slice(lr))
learn.save('dcm-512-bs32--ep2')
create_submission('512-ep2')
learn.fit_one_cycle(1, slice(lr))
learn.save('dcm-512-bs32--ep3')
create_submission('512-ep3')

# use mixup and ensemble?
from fastai2.vision.core import *

mixup = MixUp(0.3)
def get_learner_mixup():
    dbch = get_data(64,384)
    learn = cnn_learner(dbch, resnet101, loss_func=loss_func, opt_func=opt_func, metrics=metrics, cbs=mixup)
    return learn.to_fp16()


learn = get_learner_mixup()

learn.load('dcm-384-bs64--ep4')

dbunch = get_data(64, 384)
lr = 3e-4
learn.opt.clear_state()
learn.unfreeze()
learn.fit_one_cycle(1, slice(lr))
learn.save('dcm-384-bs32-mixup--ep1')
create_submission('384mixup-1')
learn.fit_one_cycle(1, slice(lr))
learn.save('dcm-384-bs32-mixup--ep3')
create_submission('384mixup-2')
learn.fit_one_cycle(1, slice(lr))
learn.save('dcm-384-bs32-mixup--ep3')
create_submission('384mixup-3')

# ============================
# ## Prepare for submission
def create_submission(id, bs=32, sz=512)
    test_fns = [(path_tst/f'{filename(o)}.dcm').absolute() for o in df_tst.fname.values]
    #print(len(test_fns) * 6)
    dbch = get_data(bs, sz) # size used for trained model
    tst = test_dl(dbch, test_fns)
    x = tst.one_batch()[0]
    x.min(),x.max()

    preds,targs = learn.get_preds(dl=tst)
    preds_clipped = preds.clamp(.00001, .9999)

    ids = []
    labels = []
    for idx,pred in zip(df_tst.index, preds_clipped):
        for i,label in enumerate(htypes):
            ids.append(f"{idx}_{label}")
            predicted_probability = '{0:1.10f}'.format(pred[i].item())
            labels.append(predicted_probability)

    df_csv = pd.DataFrame({'ID': ids, 'Label': labels})
    df_csv.to_csv(f'submission_{id}.csv', index=False)
    # df_csv.head()
    # len(df_csv)


# from IPython.display import FileLink, FileLinks
# FileLink('submission.csv')

