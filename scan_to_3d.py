#!/usr/bin/env python3
"""
scan_to_3d.py - turn a medical scan into a smooth, color-coded, web-ready .glb model.
 
Accepts
  * DICOM series: a folder of .dcm files, a .zip of them, or one multi-frame .dcm
    (the downloadable CT / MR sample sets are usually this)
  * NIfTI (.nii / .nii.gz): either a SEGMENTATION with integer labels
    (1 = organ, 2 = tumor ...) or a raw CT / MR volume
 
Install:
    python -m pip install nibabel numpy scipy scikit-image trimesh pydicom
    python -m pip install fast-simplification      (optional: shrinks very dense meshes)
 
Examples:
    python scan_to_3d.py "C:\\scans\\BRAINIX"                       (folder of .dcm files)
    python scan_to_3d.py brainix.zip --layers tissue,bright -o brain.glb
    python scan_to_3d.py ct_chest.zip --layers skin,lungs,bone
    python scan_to_3d.py mra_vessels.zip --layers bright --percentile 97
    python scan_to_3d.py ct_chest.zip --threshold 300 -o bone.glb
    python scan_to_3d.py liver_0.nii.gz --names 1=liver,2=tumor      (labelled segmentation)
 
Layer presets for raw scans (these are intensity thresholds, not AI segmentation):
    CT : skin (body surface), bone (>= 300 HU), lungs (air inside the body)     default: skin,bone
    MR : tissue (Otsu threshold, outer anatomy), bright (top percentile:
         vessels in MRA, enhancing lesions)                                     default: tissue,bright
"""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path
 
import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage
from skimage import filters, measure
from trimesh.visual.material import PBRMaterial
 
# (RGB, opacity or None = "use --opacity", emissive glow)
LABEL_STYLE = {
    1: ((235, 95, 110), None, (0.0, 0.0, 0.0)),    # organ: pink-red
    2: ((190, 255, 40), 1.0, (0.30, 0.45, 0.0)),   # tumor: bright yellow-green, glowing
    3: ((60, 200, 255), 1.0, (0.0, 0.10, 0.20)),
    4: ((255, 160, 40), 1.0, (0.30, 0.12, 0.0)),
    5: ((190, 120, 255), 1.0, (0.12, 0.05, 0.2)),
}
PRESET_STYLE = {
    "skin": ((230, 180, 150), 0.25, (0, 0, 0)),
    "bone": ((240, 235, 215), 1.0, (0, 0, 0)),
    "lungs": ((120, 190, 255), 0.50, (0, 0, 0)),
    "tissue": ((225, 170, 150), 0.35, (0, 0, 0)),
    "bright": ((190, 255, 40), 1.0, (0.30, 0.45, 0.0)),
    "surface": ((240, 235, 215), 1.0, (0, 0, 0)),
}
 
 
# ----------------------------------------------------------------------------- loading
def to_canonical(data, affine):
    img = nib.as_closest_canonical(nib.Nifti1Image(np.asarray(data, np.float32), affine))
    return np.asarray(img.dataobj), np.array(img.header.get_zooms()[:3], dtype=float)
 
 
def load_nifti(path):
    img = nib.as_closest_canonical(nib.load(str(path)))  # axes -> (R, A, S)
    data = np.asarray(img.dataobj)
    while data.ndim > 3:
        data = data[..., 0]  # 4D: use the first volume
    return data, np.array(img.header.get_zooms()[:3], dtype=float)
 
 
def _rescale(ds, arr):
    return arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
 
 
def _stack_slices(pydicom, files):
    dss = [pydicom.dcmread(str(f)) for f in files]
    dss = [d for d in dss if "ImagePositionPatient" in d and "ImageOrientationPatient" in d]
    if len(dss) < 3:
        sys.exit(f"Need at least 3 slices to build a 3D model (found {len(dss)}). "
                 "A single DICOM file is just one flat image - use the whole series folder or zip.")
    iop = np.array(dss[0].ImageOrientationPatient, float)
    rowdir, coldir = iop[:3], iop[3:]
    normal = np.cross(rowdir, coldir)
    items = []
    for d in dss:
        pos = np.array(d.ImagePositionPatient, float)
        items.append((float(pos @ normal), pos, _rescale(d, d.pixel_array.astype(np.float32))))
    items.sort(key=lambda t: t[0])
    clean = []
    for t in items:  # drop odd-sized images and duplicate locations (e.g. repeated phases)
        if t[2].shape != items[0][2].shape or (clean and abs(t[0] - clean[-1][0]) < 1e-3):
            continue
        clean.append(t)
    proj = np.array([t[0] for t in clean])
    vol = np.stack([t[2].T for t in clean], axis=2)  # (col, row, slice)
    step = np.diff(proj)
    if step.max() - step.min() > 0.05 * step.mean():
        print("  note: uneven slice spacing, resampling to even spacing")
        from scipy.interpolate import interp1d
        even = np.linspace(proj[0], proj[-1], len(proj))
        vol = interp1d(proj, vol, axis=2)(even).astype(np.float32)
        proj = even
    dz = (proj[-1] - proj[0]) / (len(proj) - 1)
    dr, dc = (float(x) for x in dss[0].PixelSpacing)  # row spacing, column spacing
    aff = np.eye(4)
    aff[:3, 0], aff[:3, 1], aff[:3, 2], aff[:3, 3] = rowdir * dc, coldir * dr, normal * dz, clean[0][1]
    return vol, np.diag([-1, -1, 1, 1]) @ aff  # DICOM LPS -> RAS
 
 
def _multiframe(ds):
    arr = _rescale(ds, ds.pixel_array.astype(np.float32))  # (frames, rows, cols)
    dr = dc = dz = 1.0
    try:
        pm = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        dr, dc = (float(x) for x in pm.PixelSpacing)
        dz = float(getattr(pm, "SpacingBetweenSlices", getattr(pm, "SliceThickness", 1.0)))
    except Exception:
        print("  note: slice spacing not found in this file, assuming 1 mm; proportions may be off")
        if "PixelSpacing" in ds:
            dr, dc = (float(x) for x in ds.PixelSpacing)
    print("  note: multi-frame file, assuming axial slices")
    return arr.transpose(2, 1, 0), np.diag([-dc, -dr, dz, 1.0])
 
 
def load_dicom(path):
    try:
        import pydicom
    except ImportError:
        sys.exit("DICOM input needs pydicom:  python -m pip install pydicom")
    tmp = None
    if path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp.name)
        root = Path(tmp.name)
    else:
        root = path
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
 
    series = {}
    for f in files:
        try:
            h = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:
            continue
        if "Rows" in h:
            key = (str(getattr(h, "SeriesInstanceUID", "?")), str(getattr(h, "Modality", "?")))
            series.setdefault(key, []).append(f)
    if not series:
        sys.exit(f"No readable DICOM images found in {path}")
    (uid, modality), flist = max(series.items(), key=lambda kv: len(kv[1]))
    print(f"  found {len(series)} series; using the largest: {modality}, {len(flist)} image file(s)")
 
    try:
        first = pydicom.dcmread(str(flist[0]))
        if int(getattr(first, "NumberOfFrames", 1) or 1) > 1 and len(flist) == 1:
            vol, aff = _multiframe(first)
        else:
            vol, aff = _stack_slices(pydicom, flist)
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"Could not decode the pixel data ({e}).\nIf the files are compressed, run:\n"
                 "  python -m pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg")
    data, spacing = to_canonical(vol, aff)
    if tmp:
        tmp.cleanup()
    return data, spacing, modality
 
 
# ----------------------------------------------------------------------------- masks
def keep_components(mask, min_voxels, frac=0.0):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.flatnonzero((sizes >= min_voxels) & (sizes >= frac * sizes.max())) + 1
    return np.isin(lab, keep)
 
 
def keep_largest(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    return lab == (int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))) + 1)
 
 
def fill_axial(mask):
    out = mask.copy()
    for k in range(mask.shape[2]):
        out[:, :, k] = ndimage.binary_fill_holes(mask[:, :, k])
    return out
 
 
def raw_layers(data, kind, names, threshold, below, percentile):
    """Intensity-based layers for a raw (unsegmented) CT / MR volume: [(name, mask)]."""
    if threshold is not None:
        return [("surface", data <= threshold if below else data >= threshold)]
    out = []
    if kind == "CT":
        names = names or ["skin", "bone"]
        body = fill_axial(keep_largest(data > -300)) if {"skin", "lungs"} & set(names) else None
        for n in names:
            if n == "skin":
                out.append((n, body))
            elif n == "bone":
                out.append((n, data >= 300))
            elif n == "lungs":
                out.append((n, keep_components((data < -400) & body, 50, frac=0.05)))
            else:
                sys.exit(f"Unknown CT layer '{n}'. Choose from: skin, bone, lungs")
    else:
        names = names or ["tissue", "bright"]
        nz = data[data > 0]
        if nz.size < 1000:
            sys.exit("This volume looks empty.")
        for n in names:
            if n == "tissue":
                out.append((n, fill_axial(keep_largest(data > filters.threshold_otsu(nz[::5])))))
            elif n == "bright":
                out.append((n, data >= np.percentile(nz, percentile)))
            else:
                sys.exit(f"Unknown MR layer '{n}'. Choose from: tissue, bright")
    return out
 
 
# ----------------------------------------------------------------------------- meshing
def mask_to_mesh(mask, spacing, sigma_mm, step, min_voxels, taubin_iters, max_faces):
    mask = keep_components(mask, min_voxels)
    boxes = ndimage.find_objects(mask.astype(np.uint8))
    if not boxes:
        return None
    sl = boxes[0]
    lo = np.maximum(np.array([s.start for s in sl]) - 3, 0)
    hi = np.minimum(np.array([s.stop for s in sl]) + 3, mask.shape)
    crop = np.pad(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 2)  # pad so the surface closes
 
    vol = crop.astype(np.float32)
    if sigma_mm > 0:  # smooth the volume; sigma scaled per axis for anisotropic voxels
        vol = ndimage.gaussian_filter(vol, sigma=np.clip(sigma_mm / spacing, 0.5, 4.0))
    verts, faces, _, _ = measure.marching_cubes(
        vol, level=0.5, spacing=tuple(spacing), step_size=step, allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices=verts + (lo - 2) * spacing, faces=faces, process=True)
 
    if max_faces and len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        except Exception as e:
            print(f"  note: skipped simplification ({e}). python -m pip install fast-simplification")
    if taubin_iters:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=taubin_iters)
    return mesh
 
 
def parse_names(text):
    names = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            names[int(k)] = v.strip()
    return names
 
 
def main():
    ap = argparse.ArgumentParser(description="Medical scan (DICOM / NIfTI) -> color-coded GLB")
    ap.add_argument("input", help="DICOM folder, .zip, .dcm file, or .nii / .nii.gz")
    ap.add_argument("-o", "--output", help="output .glb (default: next to the input)")
    ap.add_argument("--layers", help="raw scans: comma list. CT: skin,bone,lungs  MR: tissue,bright")
    ap.add_argument("--threshold", type=float, help="raw scans: one custom surface at this intensity")
    ap.add_argument("--below", action="store_true", help="with --threshold: keep voxels at or below it")
    ap.add_argument("--percentile", type=float, default=98, help="MR 'bright' layer cutoff (default 98)")
    ap.add_argument("--names", help="segmentation label names, e.g. 1=liver,2=tumor")
    ap.add_argument("--no-envelope", action="store_true", help="build label 1 from label 1 only")
    ap.add_argument("--opacity", type=float, help="override opacity of the see-through outer layer")
    ap.add_argument("--sigma", type=float, default=1.2, help="volume smoothing in mm (default 1.2, 0 = off)")
    ap.add_argument("--taubin", type=int, default=12, help="mesh smoothing iterations (default 12)")
    ap.add_argument("--downsample", type=int, help="raw scans: keep every Nth voxel (auto for huge scans)")
    ap.add_argument("--step", type=int, default=1, help="marching cubes step; 2 = faster, coarser")
    ap.add_argument("--min-voxels", type=int, default=50, help="drop islands smaller than this")
    ap.add_argument("--max-faces", type=int, default=150000, help="reduce each mesh to this many faces (0 = off)")
    ap.add_argument("--scale", type=float, default=0.001, help="mm to GLB units (default 0.001 = meters)")
    a = ap.parse_args()
 
    src = Path(a.input)
    if not src.exists():
        sys.exit(f"Not found: {src}")
    stem = src.name.split(".")[0] or "scan"
    out = Path(a.output) if a.output else (src.parent / f"{stem}.glb")
 
    print(f"Loading {src} ...")
    is_nifti = src.name.lower().endswith((".nii", ".nii.gz"))
    kind = None
    if is_nifti:
        data, spacing = load_nifti(src)
    else:
        data, spacing, modality = load_dicom(src)
        kind = "CT" if modality == "CT" else "MR"
    print(f"  shape {data.shape}, voxel size {np.round(spacing, 2)} mm")
 
    layers = []  # (name, mask, rgb, alpha, glow)
    ml = float(np.prod(spacing)) / 1000.0
    sample = data[::2, ::2, ::2]
    is_seg = is_nifti and a.threshold is None and bool(
        float(data.min()) >= 0 and np.allclose(sample, np.round(sample)) and len(np.unique(sample)) <= 30)
 
    if is_seg:
        labels = [int(v) for v in np.unique(data) if v > 0]
        if not labels:
            sys.exit("The scan has no labeled voxels (everything is 0).")
        names, multi = parse_names(a.names), len(labels) > 1
        for lab in labels:
            mask = (data > 0) if (multi and lab == labels[0] and not a.no_envelope) else (data == lab)
            name = names.get(lab, "organ" if lab == 1 else "tumor" if lab == 2 else f"label_{lab}")
            rgb, alpha, glow = LABEL_STYLE.get(lab, ((200, 200, 200), 1.0, (0, 0, 0)))
            if alpha is None:
                alpha = (a.opacity if a.opacity is not None else 0.45) if multi else 1.0
            layers.append((name, mask, rgb, alpha, glow))
    else:
        kind = kind or ("CT" if float(data.min()) < -300 else "MR")
        f = a.downsample or (2 if data.size > 60e6 else 1)
        if f > 1:
            print(f"  downsampling by {f} for speed (use --downsample 1 to keep full detail)")
            data, spacing = data[::f, ::f, ::f], spacing * f
        ml = float(np.prod(spacing)) / 1000.0
        print(f"  raw {kind} volume - building surfaces from intensity presets")
        chosen = [s.strip() for s in a.layers.split(",")] if a.layers else None
        for name, mask in raw_layers(data, kind, chosen, a.threshold, a.below, a.percentile):
            rgb, alpha, glow = PRESET_STYLE.get(name, PRESET_STYLE["surface"])
            if a.opacity is not None and alpha < 1:
                alpha = a.opacity
            layers.append((name, mask, rgb, alpha, glow))
 
    meshes = []
    for name, mask, rgb, alpha, glow in layers:
        print(f"Layer '{name}': {float(mask.sum()) * ml:.1f} mL - meshing ...")
        m = mask_to_mesh(mask, spacing, a.sigma, a.step, a.min_voxels, a.taubin, a.max_faces)
        if m is None:
            print("  empty after cleanup, skipped")
            continue
        meshes.append((name, m, rgb, alpha, glow))
    if not meshes:
        sys.exit("No meshes were produced. Try --layers, --threshold, or a lower --min-voxels.")
 
    # glTF is Y-up: (R, A, S) -> (X, Y=S, Z=-A). One shared center keeps inner layers aligned.
    allv = np.vstack([m.vertices for _, m, *_ in meshes])
    center = (allv.min(0) + allv.max(0)) / 2
    scene = trimesh.Scene()
    for name, m, rgb, alpha, glow in meshes:
        v = m.vertices - center
        m.vertices = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]) * a.scale
        m.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
            name=name, baseColorFactor=[*rgb, int(alpha * 255)], metallicFactor=0.0,
            roughnessFactor=0.45, emissiveFactor=list(glow),
            alphaMode="BLEND" if alpha < 1 else "OPAQUE", doubleSided=True))
        scene.add_geometry(m, node_name=name, geom_name=name)
        print(f"  {name}: {len(m.vertices):,} vertices, {len(m.faces):,} faces")
 
    scene.export(str(out), file_type="glb")
    print(f"\nSaved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("Next: in the AR HUD click 'Add .glb files', pick this file and name it (e.g. 'brain').")
 
 
if __name__ == "__main__":
    main()
 