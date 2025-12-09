"""
File name: geometry_utils.py
Purpose: Perform geometric checks (bounds transforms, overlap tests, and
         overlap ratios) to verify whether DEM tiles and Sentinel scenes
         intersect before reprojection.
"""

from rasterio.warp import transform_bounds

def bounds_overlap(bounds1, bounds2):
    """
    Checks if two bounding boxes overlap.
    Each bounds is a 4-tuple: (left, bottom, right, top)
    """
    l1, b1, r1, t1 = bounds1
    l2, b2, r2, t2 = bounds2
    return not (r1 < l2 or r2 < l1 or t1 < b2 or t2 < b1)

def transform_bounds_to_match_crs(bounds, src_crs, dst_crs):
    """
    Transforms bounds from one CRS to another.
    """
    return transform_bounds(src_crs, dst_crs, *bounds, densify_pts=21)

def overlap_ratio(b1, b2):
    """Intersection area / min(area(b1), area(b2)) for a quick sanity check."""
    l1, btm1, r1, t1 = b1
    l2, btm2, r2, t2 = b2
    ix = max(0, min(r1, r2) - max(l1, l2))
    iy = max(0, min(t1, t2) - max(btm1, btm2))
    inter = ix * iy
    a1 = (r1 - l1) * (t1 - btm1)
    a2 = (r2 - l2) * (t2 - btm2)
    if inter == 0 or a1 == 0 or a2 == 0:
        return 0.0
    return inter / min(a1, a2)
