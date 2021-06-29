import os
import glob
import sys

main_dirs = sys.argv[1:] 

for main_dir in main_dirs:
  subdirs = [x for x in glob.glob(os.path.join(main_dir, 'tumor*')) 
             if os.path.isdir(x)]

  for subdir in subdirs:
    print(subdir)
    img_list = glob.glob(os.path.join(subdir, '*VR1488*'))
    for img in img_list:
      new_path = img.replace('VR1488', 'VR14_88')
      os.rename(img, new_path)
