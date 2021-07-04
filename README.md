 SpindleMesoNet

Workflow:

1. Slide tiling and other preprocessing 
- stain_normalization.py - color normalization using Vahadane algorithm
- tiling.py - tesselates WSIs
- tissue_detection.py - detects regions of tissue, excluding whitespace and other slide artifacts

2. CNN training and deployment
- main_run.py - generates train/val/test splits and runs training
- trainer.py - runs CNN training in pytorch
- dataloader.py - pytorch dataloader
- pth_models.py - custom pytorch models
- prediction.py - tile level prediction

3. Slide level prediction
- slide_prediction.py - slide level prediction using average pooling
- rnn_prediction.py - slide level prediction using RNN based method
- save_predictive_tiles.py - save most predictive tiles to separate folder for manual review

4. Other
- misc.py - miscellaneous helper and utility functions
- heatmap_from_coords.py - generates a heatmap 
