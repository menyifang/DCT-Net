import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

##### DCT-Net
## anime style
img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                       model='damo/cv_unet_person-image-cartoon_compound-models')
result = img_cartoon('input.png')
cv2.imwrite('result_anime.png', result[OutputKeys.OUTPUT_IMG])

## 3d style
img_cartoon = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon-3d_compound-models')
result = img_cartoon('input.png')
cv2.imwrite('result_3d.png', result[OutputKeys.OUTPUT_IMG])

## handdrawn style
img_cartoon = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon-handdrawn_compound-models')
result = img_cartoon('input.png')
cv2.imwrite('result_handdrawn.png', result[OutputKeys.OUTPUT_IMG])

## sketch style
img_cartoon = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon-sketch_compound-models')
result = img_cartoon('input.png')
cv2.imwrite('result_sketch.png', result[OutputKeys.OUTPUT_IMG])

## artstyle style
img_cartoon = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon-artstyle_compound-models')
result = img_cartoon('input.png')
cv2.imwrite('result_artstyle.png', result[OutputKeys.OUTPUT_IMG])

#### DCT-Net + SD
## design style
img_cartoon = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon-sd-design_compound-models')
result = img_cartoon('input.png')
cv2.imwrite('result_sd_design.png', result[OutputKeys.OUTPUT_IMG])

## illustration style
img_cartoon = pipeline(Tasks.image_portrait_stylization,
                       model='damo/cv_unet_person-image-cartoon-sd-illustration_compound-models')
result = img_cartoon('input.png')
cv2.imwrite('result_sd_illustration.png', result[OutputKeys.OUTPUT_IMG])


print('finished!')
