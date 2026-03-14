"""
Interactive 3D Brain Visualization: FPN vs DMN
===============================================
- Brain surface : MNI152 T1 (same space as SPM, includes cerebellum)
- FPN blobs     : all voxels T > 6.13, p < .05 FWE  (2back > 0back)
- DMN blobs     : all voxels T > 6.13, p < .05 FWE  (0back > 2back)
"""

import numpy as np
import nibabel as nib
from nilearn import datasets, image
from skimage import measure
import plotly.graph_objects as go


# PATHS, THRESHOLD & COLOR SCALES
from config import FPN_PATH as fpn_path, DMN_PATH as dmn_path
threshold = 6.13   # FWE-corrected height threshold from SPM results
t_max     = 10.48  # observed maximum T-value across both contrasts

# Blue→green colorscale for DMN (approximates MRIcroGL 5bluegrn to match the figure in the report)
bluegrn = [
    [0.00, 'rgb(0,   0,   160)'],
    [0.25, 'rgb(0,  40,   210)'],
    [0.50, 'rgb(0,  160,  190)'],
    [0.75, 'rgb(0,  210,  130)'],
    [1.00, 'rgb(0,  230,   60)'],
]

# LOAD & COMBINE ACTIVATION MAPS
fpn_img  = nib.load(fpn_path)
dmn_img  = nib.load(dmn_path)

# Flip DMN to negative so positive = FPN, negative = DMN in one image
dmn_data = dmn_img.get_fdata()
neg_dmn  = nib.Nifti1Image(-dmn_data, dmn_img.affine, dmn_img.header)

# Keep only voxels that independently exceed the threshold in their own contrast
# (prevents FPN and DMN signals from cancelling each other in overlapping voxels)
combined = image.math_img(
    f"np.where(fpn > {threshold}, fpn, 0) + np.where(neg_dmn < -{threshold}, neg_dmn, 0)",
    fpn=fpn_img, neg_dmn=neg_dmn
)

# EXTRACT SUPRATHRESHOLD VOXEL COORDINATES
data   = combined.get_fdata()
affine = combined.affine

# FPN: positive voxels above threshold
fpn_vox = np.array(np.where(data > threshold)).T
fpn_mni = nib.affines.apply_affine(affine, fpn_vox)
fpn_val = data[data > threshold]

# DMN: negative voxels below -threshold
dmn_vox = np.array(np.where(data < -threshold)).T
dmn_mni = nib.affines.apply_affine(affine, dmn_vox)
dmn_val = np.abs(data[data < -threshold])


# BUILD BRAIN SURFACE FROM MNI152 T1
# Uses the same coordinate space as SPM
# Marching cubes extracts a mesh at a grey-matter intensity threshold
mni_t1  = datasets.load_mni152_template(resolution=2)
t1_data = mni_t1.get_fdata()

surf_threshold = np.percentile(t1_data[t1_data > 0.01], 20)
verts, faces, _, _ = measure.marching_cubes(t1_data, level=surf_threshold)
verts_mni = nib.affines.apply_affine(mni_t1.affine, verts)
x, y, z   = verts_mni[:, 0], verts_mni[:, 1], verts_mni[:, 2]
i, j, k   = faces[:, 0], faces[:, 1], faces[:, 2]

# BUILD PLOTLY FIGURE
fig = go.Figure()

# Brain surface (semi-transparent, shaded by z for sulcal contrast)
fig.add_trace(go.Mesh3d(
    x=x, y=y, z=z, i=i, j=j, k=k,
    intensity=z,
    colorscale=[[0, 'rgb(60,60,60)'], [0.5, 'rgb(130,130,130)'], [1, 'rgb(200,200,200)']],
    showscale=False, opacity=0.3, name='Brain', hoverinfo='skip', showlegend=False,
    flatshading=False,
    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
    lightposition=dict(x=0, y=100, z=100),
))

# FPN activation blobs (colored by T-value, with colorbar)
# showlegend=True puts this trace directly in the legend — clicking it hides all FPN blobs
if len(fpn_mni) > 0:
    fig.add_trace(go.Scatter3d(
        x=fpn_mni[:, 0], y=fpn_mni[:, 1], z=fpn_mni[:, 2],
        mode='markers', hoverinfo='skip', showlegend=True,
        name='<span style="color:rgb(230,110,190)">FPN (2back > 0back)</span>',
        marker=dict(
            size=2.5, color=fpn_val, colorscale='Inferno',
            cmin=threshold, cmax=t_max, opacity=0.75,
            colorbar=dict(
                title=dict(text='T (FPN)', font=dict(color='white')),
                x=1.02, len=0.45, y=0.75, tickfont=dict(color='white'),
            ),
        )
    ))

# DMN activation blobs (colored by T-value, with colorbar)
if len(dmn_mni) > 0:
    fig.add_trace(go.Scatter3d(
        x=dmn_mni[:, 0], y=dmn_mni[:, 1], z=dmn_mni[:, 2],
        mode='markers', hoverinfo='skip', showlegend=True,
        name='<span style="color:rgb(80,140,255)">DMN (0back > 2back)</span>',
        marker=dict(
            size=2.5, color=dmn_val, colorscale=bluegrn,
            cmin=threshold, cmax=t_max, opacity=0.75,
            colorbar=dict(
                title=dict(text='T (DMN)', font=dict(color='white')),
                x=1.02, len=0.45, y=0.25, tickfont=dict(color='white'),
            ),
        )
    ))

fig.update_layout(
    title=dict(
        text="Fronto-Parietal Network (FPN) vs Default Mode Network (DMN)  |  T > 6.13, p < .05 FWE",
        font=dict(size=15, color='white')
    ),
    scene=dict(
        xaxis=dict(title='L ← → R (mm)', color='white',
                   backgroundcolor='rgb(15,15,15)', showgrid=False, zeroline=False, dtick=20),
        yaxis=dict(title='P ← → A (mm)', color='white',
                   backgroundcolor='rgb(15,15,15)', showgrid=False, zeroline=False, dtick=20),
        zaxis=dict(title='I ← → S (mm)', color='white',
                   backgroundcolor='rgb(15,15,15)', showgrid=False, zeroline=False, dtick=20),
        bgcolor='rgb(15,15,15)', aspectmode='data',
        camera=dict(eye=dict(x=1.5, y=0, z=0.5)),
    ),
    paper_bgcolor='rgb(15,15,15)',
    font_color='white',
    legend=dict(x=0.01, y=0.88, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')),
    annotations=[dict(
        text="Click on a network to hide or show it",
        xref='paper', yref='paper',
        x=0.01, y=0.97,
        xanchor='left', yanchor='top',
        showarrow=False,
        font=dict(color='white', size=11),
        bgcolor='rgba(0,0,0,0.5)',
        borderpad=4,
    )],
    margin=dict(l=0, r=0, t=40, b=0),
)

# SAVE
output_path = "both_networks_3d.html"
fig.write_html(output_path, include_plotlyjs=True)
