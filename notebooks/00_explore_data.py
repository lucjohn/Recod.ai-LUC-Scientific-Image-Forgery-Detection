# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python (forgery)
#     language: python
#     name: forgery
# ---

# %% [markdown]
# # Chapter 0: Exploring the data
#
# What does the competition data look like, and what does that mean for how we evaluate?
# All logic lives in the `forgery` package; this notebook only calls it and plots.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forgery.data import EXCLUDED_SUPPLEMENTAL_IDS, build_index, load_image, load_masks
from forgery.split import get_slice, load_split

index = build_index()
print(f"{len(index)} image files")

# %% [markdown]
# ## 1. How many images of each kind?
#
# Three sources: authentic training images, forged training images (each with a mask file),
# and a small supplemental set of forged images. Every image also belongs to a slice:
# train or val (our split), or supplemental (a separate small evaluation set).

# %%
slice_of = {}
for name in ("train", "val", "supplemental"):
    slice_of.update(dict.fromkeys(get_slice(name, index).sample_id, name))
index["slice"] = index.sample_id.map(slice_of)
pd.crosstab(index.source, index.slice, margins=True)

# %% [markdown]
# ## 2. Image sizes, forged area, and number of forged regions
#
# A mask has one layer ("instance") per forged region. The forged area fraction is the share
# of the image covered by the union of all layers.

# %%
forged = index[index.source != "authentic"].copy()
forged["area_fraction"] = [
    np.maximum.reduce(load_masks(p)).mean() for p in forged.mask_path
]

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].hist([index.width, index.height], bins=40, label=["width", "height"])
ax[0].set(title="Image size (px)", xlabel="pixels", ylabel="images")
ax[0].legend()
ax[1].hist(forged.area_fraction, bins=40)
ax[1].set(title="Forged area fraction (forged images)", xlabel="share of image", ylabel="images")
counts = forged.n_instances.value_counts().sort_index()
ax[2].bar(counts.index.astype(str), counts.values)
ax[2].set(title="Forged regions per image", xlabel="instances", ylabel="images")
plt.tight_layout()
plt.show()

print(forged.area_fraction.describe(percentiles=[0.1, 0.5, 0.9]).round(4))
print(counts.to_string())

# %% [markdown]
# ## 3. Twins: an authentic image and its forged copy
#
# Every authentic image has a forged twin with the same case ID. The difference image shows
# which pixels the forger changed; the mask marks the duplicated regions. Twins differ in only
# a tiny fraction of pixels, which is why the split must keep each pair on the same side:
# otherwise validation would mostly test whether a model memorised the original.
#
# Look closely: the changed pixels show only the **pasted copy**, but the mask marks **both the
# copy and the source region it was copied from** (case 33064: one changed blob, two masked
# blobs). The source is untouched, so it carries no trace of editing. It can only be found by
# noticing that it matches the copy, which is the core idea behind duplicate search.

# %%
twin_ids = sorted(set(index[index.source == "authentic"].case_id))
rng = np.random.default_rng(0)
examples = [568] + [int(c) for c in rng.choice([c for c in twin_ids if c != 568], 4, replace=False)]
by_key = index.set_index(["source", "case_id"])

fig, ax = plt.subplots(len(examples), 4, figsize=(16, 3.6 * len(examples)))
for r, case_id in enumerate(examples):
    authentic = load_image(by_key.loc[("authentic", case_id), "image_path"])
    forged_img = load_image(by_key.loc[("forged", case_id), "image_path"])
    masks = load_masks(by_key.loc[("forged", case_id), "mask_path"])
    diff = np.abs(authentic.astype(int) - forged_img.astype(int)).sum(-1)
    overlay = forged_img.astype(float)
    for i, mask in enumerate(masks):
        color = np.array(plt.cm.tab10(i)[:3]) * 255
        overlay[mask == 1] = 0.45 * overlay[mask == 1] + 0.55 * color
    panels = [
        (authentic, f"case {case_id}: authentic"),
        (forged_img, "forged"),
        (diff > 0, f"changed pixels ({(diff > 0).mean():.2%})"),
        (overlay.astype(np.uint8), f"mask overlay ({len(masks)} region(s))"),
    ]
    for c, (img, title) in enumerate(panels):
        ax[r, c].imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax[r, c].set_title(title)
        ax[r, c].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Supplemental IDs that also appear in training
#
# Four supplemental case IDs also exist among the forged training images. If they were the
# same picture, they would be scored twice; `EXCLUDED_SUPPLEMENTAL_IDS` would list them.

# %%
overlap = sorted(set(index[index.source == "supplemental"].case_id) & set(index[index.source == "forged"].case_id))
fig, ax = plt.subplots(len(overlap), 2, figsize=(10, 3.2 * len(overlap)))
for r, case_id in enumerate(overlap):
    supp = load_image(by_key.loc[("supplemental", case_id), "image_path"])
    train = load_image(by_key.loc[("forged", case_id), "image_path"])
    same = supp.shape == train.shape and np.array_equal(supp, train)
    verdict = "excluded" if case_id in EXCLUDED_SUPPLEMENTAL_IDS else "kept"
    for c, (img, label) in enumerate([(supp, "supplemental"), (train, "train forged")]):
        ax[r, c].imshow(img)
        ax[r, c].set_title(f"{case_id} {label} {img.shape[:2]}")
        ax[r, c].axis("off")
    print(f"case {case_id}: identical={same} -> {verdict}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. The validation split
#
# 20% of training cases, chosen separately among twin pairs and forged-only cases (stratified)
# with a fixed seed, and stored in `splits/val_case_ids.txt`.

# %%
val_cases = load_split()
twins = set(twin_ids)
val_rows = index[index.slice == "val"]
print(f"val cases: {len(val_cases)} ({len(val_cases & twins)} twin pairs, {len(val_cases - twins)} forged-only)")
print(f"val images: {len(val_rows)}")
print(val_rows.source.value_counts().to_string())
print(f"authentic share of val images: {(val_rows.source == 'authentic').mean():.3f}")
