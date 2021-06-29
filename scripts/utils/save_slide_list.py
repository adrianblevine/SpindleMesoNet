import os
import sys

slide_dir = sys.argv[1]
output_file = sys.argv[2]

slide_list = [x for x in os.listdir(slide_dir) if x.endswith('svs')]

with open(output_file, 'w') as f:
  for slide in slide_list:
    f.write("{}\n".format(slide))

print('wrote {} slides in {} to {}'.format(len(slide_list), slide_dir, output_file))
