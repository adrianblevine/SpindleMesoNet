import os
import sys

img_dir = '/projects/pathology_char/pathology_char_results/mesothelioma/images_512_coords/tumor_annotated_stride_0.5'

img_list = [x for x in os.listdir(img_dir) if x.endswith('png')]

slide_list = list(set([x.split('-')[0] for x in img_list]))

slide_dict = {slide: [img for img in img_list if slide + '-' in img] 
              for slide in slide_list}

cutoff1 = 20000
cutoff2 = 35000

n_deleted = 0
for slide in slide_list:
  imgs = slide_dict[slide]
  if len(imgs) > cutoff1:
    if cutoff2 > len(imgs) > cutoff1:
      keep_list = imgs[::2]
    elif len(imgs) > cutoff2:
      keep_list = imgs[::3]
    delete_list = list(set(imgs) - set(keep_list)) 
    print('{} will delete {}/{} images'.format(slide, len(delete_list),
                                               len(imgs)))
    n_deleted += len(delete_list)
    
    for img in delete_list:
      path = os.path.join(img_dir, img)
      os.remove(path)
  else:
    pass

print('number deleted:', n_deleted)
