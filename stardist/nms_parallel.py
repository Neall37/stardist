import numpy as np
import ctypes
from time import time

# Load the CUDA kernel
nms_lib = ctypes.CDLL('./libnms.so')

# Define the argument types for the CUDA kernel function
nms_lib.launch_nms_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_bool),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

def _normalize_grid(grid, ndim):
    if np.isscalar(grid):
        grid = (grid,) * ndim
    return np.array(grid)

def _ind_prob_thresh(prob, prob_thresh, b=2):
    if b is not None and np.isscalar(b):
        b = ((b, b),) * prob.ndim

    ind_thresh = prob > prob_thresh
    if b is not None:
        _ind_thresh = np.zeros_like(ind_thresh)
        ss = tuple(slice(_bs[0] if _bs[0] > 0 else None,
                         -_bs[1] if _bs[1] > 0 else None) for _bs in b)
        _ind_thresh[ss] = True
        ind_thresh &= _ind_thresh
    return ind_thresh

def NMS_3d_parallel(dist, prob, rays, grid=(1, 1, 1), b=2, nms_thresh=0.5, prob_thresh=0.5, use_bbox=True, use_kdtree=True, verbose=False):
    """Non-Maximum-Suppression of 3D polyhedra"""
    dist = np.asarray(dist)
    prob = np.asarray(prob)

    assert prob.ndim == 3 and dist.ndim == 4 and dist.shape[-1] == len(rays) and prob.shape == dist.shape[:3]

    grid = _normalize_grid(grid, 3)

    if verbose:
        print(f"Predicting instances with prob_thresh = {prob_thresh} and nms_thresh = {nms_thresh}", flush=True)

    ind_thresh = _ind_prob_thresh(prob, prob_thresh, b)
    points = np.stack(np.where(ind_thresh), axis=1)
    if verbose:
        print(f"Found {len(points)} candidates")
    probi = prob[ind_thresh]
    disti = dist[ind_thresh]

    _sorted = np.argsort(probi)[::-1]
    probi = probi[_sorted]
    disti = disti[_sorted]
    points = points[_sorted]

    if verbose:
        print("Non-maximum suppression...")
    points = (points * np.array(grid).reshape((1, 3)))

    # Convert numpy arrays to ctypes
    disti_t = disti.astype(np.float32)
    points_t = points.astype(np.float32)
    rays_t = np.array(rays.vertices, dtype=np.float32)
    faces_t = np.array(rays.faces, dtype=np.int32)
    scores_t = probi.astype(np.float32)

    n_polys = disti_t.shape[0]
    n_rays = disti_t.shape[1]
    n_faces = faces_t.shape[0]

    result = np.zeros(n_polys, dtype=ctypes.c_bool)

    nms_lib.launch_nms_kernel(
        disti_t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        points_t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rays_t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        faces_t.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        scores_t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        n_polys, n_rays, n_faces,
        ctypes.c_float(nms_thresh),
        ctypes.c_int(use_bbox),
        ctypes.c_int(use_kdtree),
        ctypes.c_int(verbose)
    )

    if verbose:
        print(f"Keeping {np.sum(result)}/{len(result)} polyhedra")

    survivors = result.astype(np.bool)
    return points[survivors], probi[survivors], disti[survivors]