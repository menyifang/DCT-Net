from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('damo/cv_unet_person-image-cartoon_compound-models', cache_dir='.')


