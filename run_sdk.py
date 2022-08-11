import cv2
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline

model_dir = snapshot_download('damo/cv_unet_person-image-cartoon_compound-models', cache_dir='.')
img_cartoon = pipeline('image-portrait-stylization', model=model_dir)

result = img_cartoon('input.png')

cv2.imwrite('result.png', result['output_img'])
print('finished!')



