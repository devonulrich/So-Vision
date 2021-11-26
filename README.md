# So-Vision
COS429 Final Project

## What we have so far (11/26)
In the .ipynb file, I have most of the code for applying face filters. There's two approaches that I tried in there --
an affine transform and a perspective transform. The affine transform is pretty simple & self-explanatory, but it
does a poor job on non-frontal faces. The perspective transform took a bit more work since we need 4 points on the
face, but I did some geometry in the .py file to get the eyes as two points, and two other points parallel to
the mouth.

## How to run
I'm using the COS429 enironment that they had us set up. The only other packages that I needed to install were
`mtcnn` and `tensorflow`, so get both through `pip`

The video program can be pretty slow (im getting ~1 FPS), but that's probably still fast enough for us to
process a dataset and add filters to each face.
