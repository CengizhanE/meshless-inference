import numpy as np

# 1- Define 128*128 image bounding box around the airfoil
# airfoil nose is at x=0, trailing edge is at x=1
X = np.linspace(-0.5, 1.5, 128)
Y = np.linspace(-0.5, 0.5, 128)
Z = 0.025 # mid-plane Z-coordinate for 2D OpenFOAM meshes

# 2- Open the OpenFOAM sampleFict to write
with open('system/sampleDict', 'w') as f:
    f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    object      sampleDict;\n}\n\n")
    f.write("type sets;\nsetFormat csv;\ninterpolationScheme cell;\nfields (p U);\n\n")
    f.write("sets\n(\n    ai_grid\n    {\n        type cloud;\n        axis xyz;\n        points\n        (\n")


    # 3- Generate the 16,384 points
    # we iterate Y in reverse so the first row is the top of the image
    # this prevents the final image from being flipped upside down in PyTorch
    for y in reversed(Y):
        for x in X:
            f.write(f" ({x:.5f} {y:.5f} {Z})\n")

    f.write(");\n }\n); \n")

print("Success: system/sampleDict generated with 128*128 coordinates.")