"""
K-means clustering for Direction of Arrival extraction from intensity maps.

This module implements weighted k-means clustering on spherical intensity maps
to extract Direction of Arrival (DoA) estimates from acoustic field predictions.
Uses Basemap for proper spherical geometry handling and includes overlap detection
to automatically reduce the number of clusters when sources are too close.

The clustering algorithm:
    1. Selects top-N intensity pixels as potential source locations
    2. Projects spherical coordinates to 2D map using Miller projection
    3. Applies weighted k-means with intensity as sample weights
    4. Checks for overlapping clusters (within 20° angular distance)
    5. Reduces number of clusters if overlaps detected, down to single cluster

Adapted from Roman et al. LAM repository for STARSS23 inference.

References
----------
.. [1] Roman et al., "UpLAM: Upsampling Latent Acoustic Map"
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from mpl_toolkits import basemap
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from lam_min.trainer.utils import cart2eq, wrapped_rad2deg


def distance_between_spherical_coordinates_rad(
    az1: float,
    ele1: float,
    az2: float,
    ele2: float
    ) -> float:
    """
    Compute great-circle angular distance between two spherical points.

    Uses the haversine-derived formula for numerical stability across
    all angular separations.

    Parameters
    ----------
    az1 : float
        Azimuth angle of first point in radians
    ele1 : float
        Elevation angle of first point in radians
    az2 : float
        Azimuth angle of second point in radians
    ele2 : float
        Elevation angle of second point in radians

    Returns
    -------
    float
        Angular distance between points in degrees [0, 180]

    Notes
    -----
    Formula: arccos(sin(ele1)*sin(ele2) + cos(ele1)*cos(ele2)*cos(|az1-az2|))
    Result is clipped to [-1, 1] before arccos to handle numerical errors.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Great-circle_distance
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def determine_similar_location(
    azi_rad1: float,
    lon_rad1: float,
    azi_rad2: float,
    lon_rad2: float,
    thresh_unify: float = 20.0,
) -> bool:
    """
    Check if two spherical locations are within threshold angular distance.

    Parameters
    ----------
    azi_rad1 : float
        Azimuth of first location in radians
    lon_rad1 : float
        Longitude of first location in radians
    azi_rad2 : float
        Azimuth of second location in radians
    lon_rad2 : float
        Longitude of second location in radians
    thresh_unify : float, optional
        Angular distance threshold in degrees (default: 20.0)

    Returns
    -------
    bool
        True if locations are closer than thresh_unify degrees, False otherwise

    Notes
    -----
    Used to detect overlapping cluster centroids during k-means refinement.
    Default threshold of 20° represents practical DoA resolution limits.
    """
    return distance_between_spherical_coordinates_rad(
        azi_rad1,
        lon_rad1,
        azi_rad2,
        lon_rad2) < thresh_unify


def get_kmeans_clusters(
    I: np.ndarray,
    R: np.ndarray,
    N_max: int = 50,
    max_sources: int = 3,
    intensity_threshold: float = 0.0,
    adaptive_k: bool = True,
    peak_ratio_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Direction of Arrival estimates via weighted k-means clustering.

    Applies k-means clustering to the top-intensity pixels of a spatial
    intensity map to identify distinct sound source directions. Uses
    intensity values as sample weights and automatically reduces the
    number of clusters if centroids overlap.

    Parameters
    ----------
    I : np.ndarray
        Intensity values for each spatial pixel, shape (N_px,)
    R : np.ndarray
        Cartesian coordinates of field-of-view pixels, shape (3, N_px)
        Each column is [x, y, z] on the unit sphere
    N_max : int, optional
        Number of top-intensity pixels to use for clustering (default: 50)
        Higher values include more data but may introduce noise
    max_sources : int, optional
        Maximum number of sources to detect (default: 3)
    intensity_threshold : float, optional
        Minimum relative intensity for a cluster to be kept (default: 0.0)
        Value between 0 and 1, where clusters with intensity < threshold * max_intensity
        are discarded. Set to 0 to disable filtering.
    adaptive_k : bool, optional
        If True, automatically determine optimal K using intensity peak analysis
        (default: True). Counts distinct peaks above peak_ratio_threshold.
    peak_ratio_threshold : float, optional
        Threshold for counting intensity peaks when adaptive_k=True (default: 0.3)
        Peaks below this fraction of max intensity are not counted as sources.

    Returns
    -------
    centroid_lon : np.ndarray
        Azimuth angles of detected sources in degrees [-180, 180]
    centroid_lat : np.ndarray
        Elevation angles of detected sources in degrees [-90, 90]

    Notes
    -----
    Algorithm steps:
    1. Select N_max pixels with highest intensity values
    2. (Optional) Estimate K adaptively from intensity distribution
    3. Convert to azimuth-elevation coordinates
    4. Project onto Miller map for proper spherical geometry
    5. Attempt k-means with K clusters (weighted by intensity)
    6. Check for overlapping centroids (within 20° angular distance)
    7. If overlaps exist, reduce K and retry until K=1 or no overlaps
    8. Filter clusters by relative intensity threshold

    The algorithm favours fewer clusters when sources are spatially close,
    preventing false detections from splitting single sources.

    Examples
    --------
    >>> I = np.random.rand(484)  # Intensity map
    >>> R = get_field()          # Spatial coordinates
    >>> az, el = get_kmeans_clusters(I, R, N_max=50, adaptive_k=True)
    >>> print(f"Found {len(az)} sources")

    See Also
    --------
    get_field : Generate spatial sampling coordinates
    distance_between_spherical_coordinates_rad : Compute angular distance
    """
    max_idx = I.argsort()[-N_max:][::-1]

    _, R_el, R_az = cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([np.min(R_el), np.max(R_el)])
    R_az_min, R_az_max = np.around([np.min(R_az), np.max(R_az)])

    bm = basemap.Basemap(
        projection="mill",
        llcrnrlat=R_el_min,
        urcrnrlat=R_el_max,
        llcrnrlon=R_az_min,
        urcrnrlon=R_az_max,
    )
    R_x, R_y = bm(R_az, R_el)

    weights = I[max_idx]

    # Determine K: either adaptive or use max_sources
    if adaptive_k:
        K = _estimate_num_sources(I, max_idx, peak_ratio_threshold, max_sources)
    else:
        K = max_sources

    # Try K clusters, reducing K if centroids overlap
    for _k in range(K, 0, -1):
        x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))
        # Match upstream behavior (n_init default was 10 in older sklearn).
        km_res = KMeans(n_clusters=_k, n_init=10).fit(x_y, sample_weight=weights)
        clusters = km_res.cluster_centers_
        centroid_lon, centroid_lat = bm(clusters[:, 0], clusters[:, 1], inverse=True)

        centroid_lon_rad = centroid_lon * np.pi / 180 # type: ignore
        centroid_lat_rad = centroid_lat * np.pi / 180 # type: ignore

        all_centroids_pairs = combinations(np.arange(_k), 2)
        centroids_overlap = False
        for _cent_pair in all_centroids_pairs:
            location_overlapping = determine_similar_location(
                centroid_lon_rad[_cent_pair[0]], # type: ignore
                centroid_lat_rad[_cent_pair[0]],
                centroid_lon_rad[_cent_pair[1]], # type: ignore
                centroid_lat_rad[_cent_pair[1]],
            )
            if location_overlapping:
                centroids_overlap = True
                break
        if not centroids_overlap:
            break

    # Apply intensity threshold filtering
    if intensity_threshold > 0 and len(centroid_lon) > 0:  # type: ignore
        centroid_lon, centroid_lat = _filter_by_intensity(
            I, R, centroid_lon, centroid_lat, intensity_threshold  # type: ignore
        )

    return centroid_lon, centroid_lat # type: ignore


def _estimate_num_sources(
    I: np.ndarray,
    max_idx: np.ndarray,
    peak_ratio_threshold: float,
    max_sources: int,
) -> int:
    """
    Estimate the number of sound sources from intensity distribution.

    Counts significant intensity peaks above the threshold. Uses a simple
    but effective approach: count how many intensity "tiers" exist in the
    top pixels, where a tier is defined by intensity dropping below a
    fraction of the previous tier's value.

    Parameters
    ----------
    I : np.ndarray
        Intensity values for each spatial pixel
    max_idx : np.ndarray
        Indices of top-N intensity pixels (sorted descending by intensity)
    peak_ratio_threshold : float
        Minimum ratio to max intensity to count as a source
    max_sources : int
        Maximum number of sources to return

    Returns
    -------
    int
        Estimated number of sources (1 to max_sources)
    """
    top_intensities = I[max_idx]
    max_intensity = top_intensities[0]

    if max_intensity <= 0:
        return 1

    # Normalise intensities relative to maximum
    normalised = top_intensities / max_intensity

    # Count intensity tiers: each time intensity drops below threshold
    # relative to max, but then has significant values, it's a new source
    n_sources = 1
    current_tier_min = 1.0  # Start at max

    for val in normalised:
        if val < peak_ratio_threshold:
            # Below noise floor, stop counting
            break
        # If value drops significantly from current tier, start new tier
        if val < current_tier_min * 0.5:
            n_sources += 1
            current_tier_min = val
            if n_sources >= max_sources:
                break
        else:
            # Track minimum in current tier
            current_tier_min = min(current_tier_min, val)

    return min(n_sources, max_sources)


def _filter_by_intensity(
    I: np.ndarray,
    R: np.ndarray,
    centroid_lon: np.ndarray,
    centroid_lat: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter cluster centroids by their intensity values.

    Removes centroids whose interpolated intensity falls below a threshold
    relative to the maximum centroid intensity.

    Parameters
    ----------
    I : np.ndarray
        Intensity values for each spatial pixel
    R : np.ndarray
        Cartesian coordinates of field-of-view pixels, shape (3, N_px)
    centroid_lon : np.ndarray
        Azimuth angles of centroids in degrees
    centroid_lat : np.ndarray
        Elevation angles of centroids in degrees
    threshold : float
        Minimum relative intensity (0 to 1)

    Returns
    -------
    filtered_lon : np.ndarray
        Filtered azimuth angles
    filtered_lat : np.ndarray
        Filtered elevation angles
    """
    if len(centroid_lon) == 0:
        return centroid_lon, centroid_lat

    # Convert centroids to Cartesian for distance computation
    centroid_cart = np.zeros((3, len(centroid_lon)))
    for i, (lon, lat) in enumerate(zip(centroid_lon, centroid_lat)):
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        centroid_cart[0, i] = np.cos(lat_rad) * np.cos(lon_rad)
        centroid_cart[1, i] = np.cos(lat_rad) * np.sin(lon_rad)
        centroid_cart[2, i] = np.sin(lat_rad)

    # Find intensity at each centroid (nearest neighbor interpolation)
    centroid_intensities = np.zeros(len(centroid_lon))
    for i in range(len(centroid_lon)):
        # Compute dot product to find nearest pixel
        dots = np.dot(centroid_cart[:, i], R)
        nearest_idx = np.argmax(dots)
        centroid_intensities[i] = I[nearest_idx]

    # Filter by relative threshold
    max_centroid_intensity = np.max(centroid_intensities)
    if max_centroid_intensity <= 0:
        return centroid_lon[:1], centroid_lat[:1]  # Return at least one

    keep_mask = centroid_intensities >= threshold * max_centroid_intensity

    # Always keep at least one centroid (the strongest)
    if not np.any(keep_mask):
        strongest_idx = np.argmax(centroid_intensities)
        keep_mask[strongest_idx] = True

    return centroid_lon[keep_mask], centroid_lat[keep_mask]

def cluster_sequence(
    intensity_maps: np.ndarray,
    R: np.ndarray,
    *,
    max_sources: int = 3,
    merge_radius_deg: float = 25.0,
    candidate_peak_ratio: float = 0.3,
    candidate_mass_ratio: float = 0.05,
    track_max_jump_deg: float = 20.0,
    track_max_gap: int = 3,
    track_min_frames: int = 3,
    track_min_active_ratio: float = 0.03,
    track_min_peak_rel: float = 0.0,
    band_maps: np.ndarray | None = None,
    band_peak_ratio: float = 0.2,
    activity_threshold: float = 0.01,
) -> list[list[dict[str, float | int]]]:
    """
    Sequence-level DoA extraction with peak picking and temporal tracking.

    For each frame, extracts up to *max_sources* candidates via iterative
    peak picking with angular suppression, then links candidates across
    frames into temporal tracks and keeps only persistent ones.

    Parameters
    ----------
    intensity_maps : np.ndarray
        Summed intensity maps, shape ``(n_frames, n_pixels)``.
    R : np.ndarray
        Unit-sphere field coordinates, shape ``(3, n_pixels)``.
    max_sources : int
        Maximum candidates to extract per frame.
    merge_radius_deg : float
        Angular radius (degrees) around each peak that is suppressed before
        searching for the next candidate.
    candidate_peak_ratio : float
        Minimum peak intensity relative to frame max for a candidate.
    candidate_mass_ratio : float
        Minimum energy share of a candidate's support region.
    track_max_jump_deg : float
        Maximum angular jump for linking a candidate to a track.
    track_max_gap : int
        Maximum consecutive empty frames before a track is closed.
    track_min_frames : int
        Minimum active frames for a track to survive.
    track_min_active_ratio : float
        Minimum fraction of total frames a track must cover.
    track_min_peak_rel : float
        Minimum mean peak_rel for a track to survive.
    band_maps : np.ndarray or None
        Per-band intensity maps ``(n_frames, n_bands, n_pixels)``.
    band_peak_ratio : float
        Fraction of band max required for a band to count as supporting.
    activity_threshold : float
        Frame-max / file-max below which a frame is considered silent.

    Returns
    -------
    list[list[dict[str, float | int]]]
        Per-frame predictions with ``"source_index"``, ``"azimuth"``,
        ``"elevation"`` keys.
    """
    n_frames = intensity_maps.shape[0]
    merge_cos = np.cos(np.radians(merge_radius_deg))
    file_max = float(intensity_maps.max())

    # Step 1: Extract candidates per frame with peak picking and angular suppression
    all_candidates: list[list[dict]] = []
    for f in range(n_frames):
        fmap = intensity_maps[f]
        # Skip frames that are too quiet relative to the file max, to avoid spurious candidates in silent segments.
        if file_max > 0 and float(fmap.max()) < activity_threshold * file_max:
            all_candidates.append([])
            continue

        # If band maps are provided, pass the per-band maps for this frame to the candidate picker for band-support scoring.
        fbands = band_maps[f] if band_maps is not None else None
        all_candidates.append(
            _pick_frame_candidates(
                fmap,
                R,
                merge_cos,
                max_sources,
                candidate_peak_ratio,
                candidate_mass_ratio,
                fbands,
                band_peak_ratio,
            )
        )

    # Step 2: Link candidates into tracks
    tracks = _link_tracks(all_candidates, n_frames, track_max_jump_deg, track_max_gap)

    # Step 3: Filter tracks by persistence and intensity
    eff_min = min(track_min_frames, max(1, n_frames))
    eff_ratio = track_min_active_ratio if n_frames >= track_min_frames else 0.0
    # The survival criteria for tracks are:
    # - Must have at least `eff_min` active frames (absolute minimum)
    # - Must cover at least `eff_ratio` fraction of the total frames (relative minimum)
    # - Must have a mean peak relative value above `track_min_peak_rel` (to ensure the track is not just barely above the noise floor)
    surviving = [
        t
        for t in tracks
        if len(t["frames"]) >= eff_min
        and (len(t["frames"]) / n_frames if n_frames > 0 else 0) >= eff_ratio
        and (np.mean(t["peak_rel"]) if t["peak_rel"] else 0) >= track_min_peak_rel
    ]

    # Convert surviving tracks into per-frame predictions
    result = _tracks_to_predictions(surviving, n_frames)

    # Fallback: for frames without surviving-track predictions, use the
    # old K=1 weighted k-means centroid (top-N_max pixels) which is proven
    # to be robust on diffuse maps.
    for f in range(n_frames):
        if not result[f] and all_candidates[f]:
            lon, lat = get_kmeans_clusters(
                intensity_maps[f], R, N_max=50, max_sources=1,
                intensity_threshold=0.0, adaptive_k=False,
                peak_ratio_threshold=0.3,
            )
            if len(lon) > 0:
                result[f].append(
                    {"source_index": -1, "azimuth": float(lon[0]),
                     "elevation": float(lat[0])}
                )

    return result


def _pick_frame_candidates(
    frame_map: np.ndarray,
    R: np.ndarray,
    merge_cos: float,
    max_sources: int,
    peak_ratio: float,
    mass_ratio: float,
    frame_bands: np.ndarray | None,
    band_peak_ratio: float,
) -> list[dict]:
    """
    Extract up to ``max_sources`` DoA candidates from one intensity frame.

    The algorithm is an iterative greedy peak picker with angular suppression:

    1. Find the highest-energy pixel on the unsuppressed map.
    2. Accept it if its value exceeds ``peak_ratio x frame_max``.
    3. Identify all pixels within the ``merge_cos`` cone around the peak.
       These form the candidate's support region.
    4. Compute the fraction of total frame energy inside the support region
       (*mass share*).  If the mass share is below ``mass_ratio`` the peak is
       considered noise and the region is suppressed without creating a
       candidate.
    5. Otherwise compute the intensity-weighted centroid of the support region
       on the unit sphere and record it as a candidate.
    6. Suppress the entire support region and return to step 1.

    For each accepted candidate the function records:

    - ``az_deg``/``el_deg``/``az_rad``/``el_rad`` — weighted-centroid direction
    - ``peak_rel`` — peak value relative to ``frame_max`` (always 1.0 for the
      first candidate, < 1.0 for weaker secondaries)
    - ``mass_share`` — fraction of frame energy in the support region
    - ``band_support`` — fraction of frequency bands whose energy within the
      support region exceeds ``band_peak_ratio x band_max``

    Parameters
    ----------
    frame_map : np.ndarray
        Summed intensity values, shape ``(n_pixels,)``.
    R : np.ndarray
        Unit-sphere pixel coordinates, shape ``(3, n_pixels)``.
    merge_cos : float
        Cosine of the angular suppression radius.  All pixels whose dot
        product with the current peak exceeds this threshold are grouped
        into the support region and suppressed afterwards.
    max_sources : int
        Maximum number of candidates to return.
    peak_ratio : float
        A peak is rejected if its value is below ``peak_ratio x frame_max``.
    mass_ratio : float
        A candidate is rejected if its support region contains less than
        ``mass_ratio`` of the total frame energy.
    frame_bands : np.ndarray or None
        Per-band intensity maps for this frame, shape
        ``(n_bands, n_pixels)``.  ``None`` disables band-support scoring.
    band_peak_ratio : float
        A band is counted as supporting if its energy in the support
        region reaches at least ``band_peak_ratio x band_max``.

    Returns
    -------
    list[dict]
        One dict per accepted candidate, with keys ``az_deg``, ``el_deg``,
        ``az_rad``, ``el_rad``, ``peak_rel``, ``mass_share``,
        ``band_support``.
    """
    frame_max = float(frame_map.max())
    if frame_max <= 0:
        return []

    total_energy = float(frame_map.sum())
    n_pixels = len(frame_map)
    suppressed = np.zeros(n_pixels, dtype=bool)
    candidates: list[dict] = []

    # Iteratively pick peaks and suppress their neighborhoods until we have enough candidates or run out of significant peaks.
    for _ in range(max_sources):
        live = frame_map * (~suppressed)
        peak_idx = int(np.argmax(live))
        peak_val = float(live[peak_idx])

        if peak_val < peak_ratio * frame_max:
            break

        # Pixels within merge_radius of peak
        with np.errstate(all="ignore"):  # suppress Apple Accelerate BLAS warnings
            dots = R.T @ R[:, peak_idx]
        nearby = (dots >= merge_cos) & (~suppressed)
        nearby_idx = np.flatnonzero(nearby)
        if len(nearby_idx) == 0:
            nearby_idx = np.array([peak_idx])

        weights_local = frame_map[nearby_idx]
        mass_share = float(weights_local.sum() / total_energy) if total_energy > 0 else 0.0

        if mass_share < mass_ratio:
            suppressed |= nearby
            continue

        # Weighted centroid on unit sphere
        centroid = np.average(R[:, nearby_idx], weights=weights_local, axis=1)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            suppressed |= nearby
            continue
        centroid /= norm

        az_rad = float(np.arctan2(centroid[1], centroid[0]))
        el_rad = float(np.arcsin(np.clip(centroid[2], -1, 1)))
        az_deg = float(np.degrees(az_rad))
        el_deg = float(np.degrees(el_rad))

        # Band support
        band_support = 0.0
        if frame_bands is not None:
            n_bands = frame_bands.shape[0]
            supporting = 0
            for b in range(n_bands):
                bmax = float(frame_bands[b].max())
                if bmax > 0 and float(frame_bands[b, nearby_idx].max()) >= band_peak_ratio * bmax:
                    supporting += 1
            band_support = supporting / n_bands if n_bands > 0 else 0.0

        candidates.append(
            {
                "az_deg": az_deg,
                "el_deg": el_deg,
                "az_rad": az_rad,
                "el_rad": el_rad,
                "peak_rel": float(peak_val / frame_max),
                "mass_share": mass_share,
                "band_support": band_support,
            }
        )

        suppressed |= nearby

    return candidates


def _link_tracks(
    all_candidates: list[list[dict]],
    n_frames: int,
    max_jump_deg: float,
    max_gap: int = 3,
) -> list[dict]:
    """
    Link per-frame candidates into temporal tracks using the Hungarian algorithm.

    On every frame with at least one candidate, the function solves a
    minimum-cost bipartite matching between the currently active tracks and
    the new candidates.  The cost between a track and a candidate is the
    great-circle angular distance between the track's last accepted position
    and the candidate direction; a match is only accepted when this distance
    is at most ``max_jump_deg``.

    **Gap tolerance.**  Active tracks are not immediately closed when a frame
    produces no matching candidate.  Instead each unmatched track accumulates
    a gap counter, and is only moved to the finished list once the gap
    exceeds ``max_gap`` consecutive frames.  This prevents short silences or
    low-energy segments from fragmenting a source into many short tracks.

    The returned track dicts contain parallel lists indexed over the frames at
    which the track was active:

    - ``frames``      — frame indices
    - ``positions``   — ``(az_deg, el_deg)`` tuples
    - ``az_rad``/``el_rad`` — same in radians (used for distance computation)
    - ``peak_rel``    — peak-relative intensity at each accepted frame
    - ``mass_share``  — energy-fraction of the support region
    - ``band_support`` — fraction of bands that supported the candidate

    Parameters
    ----------
    all_candidates : list[list[dict]]
        Per-frame candidate lists as returned by ``_pick_frame_candidates``.
        Length must equal ``n_frames``.
    n_frames : int
        Total number of frames in the sequence.
    max_jump_deg : float
        Maximum angular displacement (degrees) allowed between consecutive
        accepted positions in the same track.
    max_gap : int
        Maximum number of consecutive empty-match frames a track can survive
        before being finalised.  Default is ``3``.

    Returns
    -------
    list[dict]
        All finalised and still-active tracks after processing all frames.
    """
    # Each active item wraps a track dict and a gap counter.
    active: list[dict] = []  # [{"track": dict, "gap": int}]
    finished: list[dict] = []

    for f in range(n_frames):
        candidates = all_candidates[f]

        if not candidates:
            # Increment gap for every active track; close if exceeded.
            still_alive: list[dict] = []
            for item in active:
                item["gap"] += 1
                if item["gap"] > max_gap:
                    finished.append(item["track"])
                else:
                    still_alive.append(item)
            active = still_alive
            continue

        if not active:
            active = [{"track": _new_track(f, c), "gap": 0} for c in candidates]
            continue

        n_t = len(active)
        n_c = len(candidates)
        cost = np.full((n_t, n_c), 1e6)
        for t_i, item in enumerate(active):
            trk = item["track"]
            t_az = trk["az_rad"][-1]
            t_el = trk["el_rad"][-1]
            for c_i, cand in enumerate(candidates):
                cost[t_i, c_i] = distance_between_spherical_coordinates_rad(
                    t_az, t_el, cand["az_rad"], cand["el_rad"]
                )

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_t: set[int] = set()
        matched_c: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= max_jump_deg:
                _extend_track(active[r]["track"], f, candidates[c])
                active[r]["gap"] = 0
                matched_t.add(r)
                matched_c.add(c)

        new_active: list[dict] = []
        for t_i in range(n_t):
            if t_i in matched_t:
                new_active.append(active[t_i])
            else:
                active[t_i]["gap"] += 1
                if active[t_i]["gap"] > max_gap:
                    finished.append(active[t_i]["track"])
                else:
                    new_active.append(active[t_i])

        for c_i in range(n_c):
            if c_i not in matched_c:
                new_active.append({"track": _new_track(f, candidates[c_i]), "gap": 0})

        active = new_active

    finished.extend(item["track"] for item in active)
    return finished


def _new_track(frame: int, cand: dict) -> dict:
    """
    Initialise a new track dict from a single candidate observation.

    Parameters
    ----------
    frame : int
        Frame index at which the track begins.
    cand : dict
        Candidate dict as produced by ``_pick_frame_candidates``.

    Returns
    -------
    dict
        A track dict with all parallel sequence lists initialised to the
        single observation.
    """
    return {
        "frames": [frame],
        "positions": [(cand["az_deg"], cand["el_deg"])],
        "az_rad": [cand["az_rad"]],
        "el_rad": [cand["el_rad"]],
        "peak_rel": [cand.get("peak_rel", 1.0)],
        "mass_share": [cand.get("mass_share", 1.0)],
        "band_support": [cand.get("band_support", 0.0)],
    }


def _extend_track(track: dict, frame: int, cand: dict) -> None:
    """
    Append one candidate observation to an existing track in-place.

    Parameters
    ----------
    track : dict
        Track dict as produced by ``_new_track`` and extended by previous
        calls.
    frame : int
        Frame index of the new observation.
    cand : dict
        Candidate dict as produced by ``_pick_frame_candidates``.
    """
    track["frames"].append(frame)
    track["positions"].append((cand["az_deg"], cand["el_deg"]))
    track["az_rad"].append(cand["az_rad"])
    track["el_rad"].append(cand["el_rad"])
    track["peak_rel"].append(cand.get("peak_rel", 1.0))
    track["mass_share"].append(cand.get("mass_share", 1.0))
    track["band_support"].append(cand.get("band_support", 0.0))


def _tracks_to_predictions(
    tracks: list[dict],
    n_frames: int,
) -> list[list[dict[str, float | int]]]:
    """
    Flatten surviving tracks into per-frame prediction dicts.

    Each track contributes one prediction dict for every frame at which it
    was active.  The ``source_index`` encodes the position of the track in
    the input list, so predictions from the same track on different frames
    share an index.  Frames where no track was active produce an empty list.

    Parameters
    ----------
    tracks : list[dict]
        List of surviving track dicts, each with ``frames``, ``positions``,
        and quality-metric lists as stored by ``_extend_track``.
    n_frames : int
        Total number of frames in the sequence.  Determines the length of
        the returned list.

    Returns
    -------
    list[list[dict[str, float | int]]]
        Outer index is frame number; inner list contains one dict per
        predicted source with keys ``source_index``, ``azimuth`` (degrees),
        and ``elevation`` (degrees).
    """
    result: list[list[dict[str, float | int]]] = [[] for _ in range(n_frames)]
    for track_idx, track in enumerate(tracks):
        for i, f in enumerate(track["frames"]):
            az, el = track["positions"][i]
            result[f].append(
                {"source_index": track_idx, "azimuth": az, "elevation": el}
            )
    return result
