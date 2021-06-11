import os
import sys
import time
import argparse
import logging
import gc
from decord import VideoReader

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms

from gluoncv.data import Kinetics400Attr, UCF101Attr, SomethingSomethingV2Attr, HMDB51Attr, VideoClsCustom
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs

from tqdm import tqdm

import matplotlib.pylab as plt

# set opt_model with desired model name
opt_model = '' 

if opt_model == 'i3d_resnet50_v1_kinetics400':
    opt_slowfast=False
    opt_new_length=32
    opt_num_classes=400
    opt_num_segments=1

if opt_model == 'slowfast_4x16_resnet50_kinetics400' :
    opt_slowfast=True
    opt_new_length=64
    opt_num_segments=1
    opt_num_classes=400

if opt_model == 'resnet50_v1b_sthsthv2':
    opt_slowfast=False
    opt_new_length=32
    opt_num_segments=32
    opt_num_classes = 174

if opt_model == 'i3d_resnet50_v1_ucf101':
    opt_slowfast = False
    opt_new_length = 32
    opt_num_segments = 1
    opt_num_classes = 101


# set directories for saving features
opt_save_dir='./features'
# set directories for saving predictions
opt_text_dir = './ModelTesting'


opt_data_aug=''
opt_dtype='float32'
opt_gpu_id=0 
opt_hashtag=''
opt_input_size=224
opt_log_interval=10 
opt_mode=None 
opt_new_height=256 
opt_new_step=1
opt_new_width=340
opt_num_crop=1 

opt_resume_params=''

opt_slow_temporal_stride=16
opt_fast_temporal_stride=2 s

opt_ten_crop=False
opt_three_crop=False
opt_use_decord=True
opt_use_pretrained=True
opt_video_loader=True
opt_logging_file = 'predictions.log'
opt_save_logits = True
opt_save_preds = True

split_size_sec = 1.0 # e.g. 4.0 seconds

# Video clips used in the experiment.
stim_folder = ''

# add in all videos as a dictionary
# ex: stim_info = {vid1 : video.mp4}
stim_info = {}    


# fill in with keys from stim_info
vidclips_use = [ ]

#%% main function

makedirs(opt_save_dir)
gc.set_threshold(100, 5, 5)

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
        video.VideoNormalize(image_norm_mean, image_norm_std)
    ])
    opt_num_crop = 10
elif opt_three_crop:
    transform_test = transforms.Compose([
        video.VideoThreeCrop(opt_input_size),
        video.VideoToTensor(),
        video.VideoNormalize(image_norm_mean, image_norm_std)
    ])
    opt_num_crop = 3
else:
    transform_test = video.VideoGroupValTransform(size=opt_input_size, mean=image_norm_mean, std=image_norm_std)
    opt_num_crop = 1

# get model
if opt_use_pretrained and len(opt_hashtag) > 0:
    opt_use_pretrained = opt_hashtag
classes = opt_num_classes
model_name = opt_model
net = get_model(name=model_name, nclass=classes, pretrained=opt_use_pretrained,
                num_segments=opt_num_segments, num_crop=opt_num_crop)
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

# get classes list, if we are using a pretrained network from the model_zoo
classes = None
if opt_use_pretrained:
    if "kinetics400" in model_name:
        classes = Kinetics400Attr().classes
    elif "ucf101" in model_name:
        classes = UCF101Attr().classes
    elif "hmdb51" in model_name:
        classes = HMDB51Attr().classes
    elif "sthsth" in model_name:
        classes = SomethingSomethingV2Attr().classes
        
# build a pseudo dataset instance to use its children class methods
video_utils = VideoClsCustom(root=None,
                             setting=None,
                             num_segments=opt_num_segments,
                             num_crop=opt_num_crop,
                             new_length=opt_new_length,
                             new_step=opt_new_step,
                             new_width=opt_new_width,
                             new_height=opt_new_height,
                             video_loader=opt_video_loader,
                             use_decord=opt_use_decord,
                             slowfast=opt_slowfast,
                             slow_temporal_stride=opt_slow_temporal_stride,
                             fast_temporal_stride=opt_fast_temporal_stride,
                             data_aug=opt_data_aug,
                             lazy_init=True)

start_time = time.time()
for vii, vid_ii in enumerate(vidclips_use):
    
    print('Processing video file: %s'%vid_ii)
    video_file = os.path.join(stim_folder,stim_info[vid_ii])
    video_name = os.path.splitext(stim_info[vid_ii])[0]

    vr = VideoReader(video_file, width=opt_new_width, height=opt_new_height)
    nframes = len(vr)
    fps_avg = vr.get_avg_fps()
    
    frame_duration_sec = 1./fps_avg 
    nframes_split = split_size_sec // frame_duration_sec # the number of frames in in each video bin
    nsplits = nframes//nframes_split
    
    
    frames_idx = np.arange(nframes) # array of frame indices
    
    video_bins = np.array_split(frames_idx,nsplits) # splits the video into arrays of frames
    
   
    predictions = ''
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
        # --- send to net for feature extraction ---
        clip_input = transform_test(clip_input_init)
    
        # --- Reshape clip_input to network's input --- to be able to read it the right way
        clip_len = len(clip_input)
        clip_input = np.stack(clip_input, axis=0)
        
        if 'sthsth' not in opt_model and ('slowfast' in opt_model or 'i3d_resnet50' in opt_model):
            clip_input = clip_input.reshape((-1,) + (clip_len, 3, opt_input_size, opt_input_size))
            clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
        
        video_data = nd.array(clip_input)
        video_input = video_data.as_in_context(context)
        pred = net(video_input.astype(opt_dtype, copy=False))
        if opt_save_logits:
                logits_file = '%s_%s_logits.npy' % (model_name, video_name)
                np.save(os.path.join(opt_save_dir, logits_file), pred.asnumpy())
        pred = mx.nd.softmax(pred).asnumpy()

        pred_label = np.argsort(pred.squeeze())[::-1][:5]
        if opt_save_preds:
            preds_file = '%s_%s_preds.npy' % (model_name, video_name)
            np.save(os.path.join(opt_save_dir, preds_file), pred_label)
        pred = np.sort(pred.squeeze())[::-1][:5]
        predictions += "Time: %d\n" % (bii * split_size_sec)
        
        if classes:
            for i in range(len(pred_label)):
                predictions += '%04d/%04d: %s is predicted to class %s at %f\n' % (vii, len(video_bins), video_name, classes[pred_label[i]], pred[i])
    
end_time = time.time()
f = open(f'{opt_text_dir}/{opt_model}_predictions.txt', 'w')
f.write(predictions)
f.close()
print('Total feature extraction time is %4.2f minutes' % ((end_time - start_time) / 60))