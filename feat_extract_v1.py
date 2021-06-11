import os
import sys
import time
import gc

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model

from decord import VideoReader

from tqdm import tqdm
import matplotlib.pylab as plt

# set opt_model with desired model name
opt_model = ''

if opt_model == 'i3d_resnet50_v1_kinetics400':
    opt_slowfast=False
    opt_num_classes=400
    opt_num_segments=1

if opt_model == 'slowfast_4x16_resnet50_kinetics400':
    opt_slowfast=True
    opt_num_segments=1
    opt_num_classes=400

if opt_model == 'resent50_v1b_sthsthv2':
    opt_slowfast=False
    opt_num_segments=32
    opt_num_classes = 174



# ------ o -------

opt_dtype='float32'
opt_gpu_id=0 
opt_hashtag=''
opt_input_size=224
opt_mode=None 
opt_new_height=256 
opt_new_width=340
opt_num_crop=1 
opt_resume_params=''

opt_ten_crop=False
opt_three_crop=False
opt_use_decord=True
opt_use_pretrained=True
opt_video_loader=True

# specify a split size for each video
split_size_sec = 1.0 

# video clips used in the experiment.
stim_folder = ''

# Save outputs
save_dir='features_avp'

# add in all videos as a dictionary
# ex: stim_info = {vid1 : video.mp4}
stim_info = {}    


# fill in with keys from stim_info
vidclips_use = [ ]


#%% 
gc.set_threshold(100, 5, 5)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# set env
if opt_gpu_id == -1:
    context = mx.cpu()
else:
    gpu_id = opt_gpu_id
    context = mx.gpu(gpu_id)

# get data preprocess
image_norm_mean = [0.485, 0.456, 0.406]
image_norm_std = [0.229, 0.224, 0.225]
if opt_ten_crop:
    transform_test = transforms.Compose([
        video.VideoTenCrop(opt_input_size),
        video.VideoToTensor(),
        video.VideoNormalize(image_norm_mean, image_norm_std) ])
    opt_num_crop = 10
elif opt_three_crop:
    transform_test = transforms.Compose([
        video.VideoThreeCrop(opt_input_size),
        video.VideoToTensor(),
        video.VideoNormalize(image_norm_mean, image_norm_std) ])
    opt_num_crop = 3
else:
    transform_test = video.VideoGroupValTransform(size=opt_input_size, 
                            mean=image_norm_mean, std=image_norm_std)
    opt_num_crop = 1

# get model
if opt_use_pretrained and len(opt_hashtag) > 0:
    opt_use_pretrained = opt_hashtag
classes = opt_num_classes
model_name = opt_model
net = get_model(name=model_name, nclass=classes, pretrained=opt_use_pretrained,
                feat_ext=True, num_segments=opt_num_segments, num_crop=opt_num_crop)
net.cast(opt_dtype)
net.collect_params().reset_ctx(context)
if opt_mode == 'hybrid':
    net.hybridize(static_alloc=True, static_shape=True)
if opt_resume_params != '' and not opt_use_pretrained:
    net.load_parameters(opt_resume_params, ctx=context)
    print('Pre-trained model %s is successfully loaded.' % (opt_resume_params))
else:
    print('Pre-trained model is successfully loaded from the model zoo.')
print("Successfully built model {}".format(model_name))



start_time = time.time()
for vii, vid_ii in enumerate(vidclips_use):
    keep_feats = []
    
    print('Processing video file: %s'%vid_ii)
    video_file = os.path.join(stim_folder,stim_info[vid_ii])
    vid_basename = os.path.splitext(stim_info[vid_ii])[0]

    vr = VideoReader(video_file, width=opt_new_width, height=opt_new_height)
    nframes = len(vr)
    fps_avg = vr.get_avg_fps()
    
    frame_duration_sec = 1./fps_avg 
    nframes_split = split_size_sec // frame_duration_sec # the number of frames in in each video bin
    nsplits = nframes//nframes_split
    
    
    frames_idx = np.arange(nframes) # array of frame indices
    
    video_bins = np.array_split(frames_idx,nsplits) # splits the video into arrays of frames
    
   
    
    for bii, bin_frames in enumerate(tqdm(video_bins)):
        
        
    
        video_data_init = vr.get_batch(bin_frames).asnumpy() # series of frames, 4 second video clip
        
        if opt_slowfast: # first 32 for fast, last 4 for slow 
            sample_id_list_slow = np.linspace(0,video_data_init.shape[0]-1,32).round().astype(int)
            # linspace is an approximation
            sample_id_list_fast = np.linspace(0,video_data_init.shape[0]-1,4).round().astype(int)
            sample_id_list = np.r_[sample_id_list_slow,sample_id_list_fast]
        else:
            sample_id_list = np.linspace(0,video_data_init.shape[0]-1,32).round().astype(int)
            
        
        
        # resampled list of frames
        # series of frames
        clip_input_init = [ video_data_init[spii,...] for spii in sample_id_list ]
        sys.exit()
        # --- send to net for feature extraction ---
        clip_input = transform_test(clip_input_init)
    
        
        # --- Reshape clip_input to network's input --- to be able to read it the right way
        clip_len = len(clip_input)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (clip_len, 3, opt_input_size, opt_input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    
        # --- pass input from network to extract features ---
        video_data = nd.array(clip_input)
        video_input = video_data.as_in_context(context)
        video_feat = net(video_input.astype(opt_dtype, copy=False))
    
        keep_feats.append( video_feat.asnumpy().squeeze() )

    
    # save features.
    keep_feats = np.asarray(keep_feats)
    np.save(os.path.join(save_dir,f'{vid_basename}_{opt_model}' ), keep_feats)

    
end_time = time.time()
print('Total feature extraction time is %4.2f minutes' % ((end_time - start_time) / 60))

