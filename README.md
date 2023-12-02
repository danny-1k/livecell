## Segnet and Unet for Cell segmentation (Livecell dataset)

## Method
The cell images and segmentation masks are splitted into small patches (both for generalisation and resource efficiency).

The Segmentation models are then trained on a patch-level. Inference is done by "stitching" the generated patch by patch segmentation masks.

