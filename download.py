from modelscope.hub.snapshot_download import snapshot_download
# pre-trained models in different style
model_dir = snapshot_download('damo/cv_unet_person-image-cartoon_compound-models', cache_dir='.')
model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-3d_compound-models', cache_dir='.')
model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-handdrawn_compound-models', cache_dir='.')
model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-sketch_compound-models', cache_dir='.')
model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-artstyle_compound-models', cache_dir='.')

# pre-trained models trained with DCT-Net + Stable-Diffusion
model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-sd-design_compound-models', revision='v1.0.0', cache_dir='.')
model_dir = snapshot_download('damo/cv_unet_person-image-cartoon-sd-illustration_compound-models', revision='v1.0.0', cache_dir='.')


