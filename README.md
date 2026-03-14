# FPN vs DMN — Interactive 3D Brain Visualization

**[Open Interactive Visualization](https://djuricsonja.github.io/fpn-dmn-network-visualization/both_networks_3d_glass.html)**

Interactive 3D visualization of whole-brain activation patterns during a working memory task, showing the antagonistic relationship between the fronto-parietal network (FPN) and the default mode network (DMN).

## Background

During working memory performance, the FPN is expected to show increased activation while the DMN is expected to deactivate. This was tested using data from the Human Connectome Project (HCP) working memory task.

This visualization was created as part of the Neurocognition module at Humboldt-Universität zu Berlin.

## Results

<p align="center">
  <img src="slices_pictures/axial_comparison.png" width="45%"/>
  <img src="slices_pictures/coronal_comparison.png" width="45%"/>
</p>

*FPN (warm colors, 2-back > 0-back) and DMN (cool colors, 0-back > 2-back), T > 6.13, p < .05 FWE. Rendered in MRIcroGL from SPM12 second-level results.*

## Methods

- **Data:** COPE images (2-back > 0-back contrast) from 40 HCP participants
- **Analysis:** One-sample t-test (second-level) in SPM12 (MATLAB)
- **Contrasts:** 2-back > 0-back (FPN) and 0-back > 2-back (DMN)
- **Threshold:** T > 6.13, p < .05 FWE-corrected
- **Visualization:** Python — activation maps projected as 3D scatter points inside a transparent brain surface mesh (MNI152 T1)

## Usage

Open the visualization in a browser and:
- **Rotate** — click and drag
- **Zoom** — scroll
- **Toggle networks** — click FPN or DMN in the legend to show/hide
