import numpy as np
#----------------------makeing sin wave ---------------------------
def multi_sine(n_samples=1000,noise=0.05,n_classes=2,gap=0.3,spacing=1,random_state=None):
    rng = np.random.default_rng(random_state)

    # Oversample to compensate for removed gap points
    X = rng.uniform(-3, 3, (n_samples * 2, 2))

    base = np.sin(X[:, 0])

    # Compute class index
    band_pos = (X[:, 1] - base) / spacing
    class_id = np.floor(band_pos).astype(int)

    # Distance from nearest boundary
    nearest_boundary = base + np.round(band_pos) * spacing
    dist = np.abs(X[:, 1] - nearest_boundary)

    mask = (
        (class_id >= 0) &
        (class_id < n_classes) &
        (dist > gap)
    )

    X = X[mask][:n_samples]

    base = np.sin(X[:, 0])
    band_pos = (X[:, 1] - base) / spacing
    y = np.floor(band_pos).astype(int)

    y = np.clip(y, 0, n_classes - 1)

    X += rng.normal(scale=noise, size=X.shape)

    return X, y

#-------------------making star--------------------
def radial_wedges(n_samples=5000,noise=0.02,n_classes=5,gap=0.2,random_state=None):
    rng = np.random.default_rng(random_state)

    # Oversample so we still get enough after removing gap points
    X = rng.uniform(-1, 1, (n_samples * 2, 2))

    angles = np.arctan2(X[:, 1], X[:, 0])
    angles = (angles + np.pi) / (2 * np.pi)   # normalize 0 → 1

    sector_pos = angles * n_classes
    frac = sector_pos - np.floor(sector_pos)

    # Keep points away from sector boundaries
    mask = (frac > gap) & (frac < 1 - gap)

    X = X[mask][:n_samples]
    angles = (np.arctan2(X[:, 1], X[:, 0]) + np.pi) / (2 * np.pi)

    y = (angles * n_classes).astype(int)

    X += rng.normal(scale=noise, size=X.shape)

    return X, y

#-----------------3D Dataset-------------------------
def helix_3d(n_samples=5000, noise=0.05, random_state=None):
    rng = np.random.default_rng(random_state)

    t = rng.uniform(0, 4*np.pi, n_samples)

    x = np.cos(t)
    y = np.sin(t)
    z = t / (4*np.pi)

    X = np.stack([x, y, z], axis=1)
    y_label = (t > 2*np.pi).astype(int)

    X += rng.normal(scale=noise, size=X.shape)
    return X, y_label

#------------------Flower Pot------------------
def torus_3d(n_samples=5000, noise=0.05, R=8, r=6, random_state=None):
    rng = np.random.default_rng(random_state)

    u = rng.uniform(0, 2*np.pi, n_samples)
    v = rng.uniform(0, 2*np.pi, n_samples)

    x = (R + r*np.cos(v)) * np.cos(u)
    y = (R + r*np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    X = np.stack([x, y, z], axis=1)
    y_label = (v > np.pi).astype(int)

    X += rng.normal(scale=noise, size=X.shape)
    return X, y_label

#------------------mobius strip------------------
def mobius(n_samples=50000,noise=0.02,gap=0.1,random_state=None):
    rng = np.random.default_rng(random_state)

    # Oversample so we still have enough after removing gap region
    u = rng.uniform(0, 2*np.pi, n_samples * 2)
    v = rng.uniform(-0.5, 0.5, n_samples * 2)

        # Circular distance helper
    def circular_dist(a, b):
        return np.minimum(np.abs(a - b), 2*np.pi - np.abs(a - b))

    dist0 = circular_dist(u, 0)
    dist1 = circular_dist(u, np.pi)

    mask = (dist0 > gap) & (dist1 > gap)

    u = u[mask][:n_samples]
    v = v[mask][:n_samples]

    x = (1 + v*np.cos(u/2)) * np.cos(u)
    y = (1 + v*np.cos(u/2)) * np.sin(u)

    X = np.stack([x, y], axis=1)

    y_label = (u > np.pi).astype(int)

    X += rng.normal(scale=noise, size=X.shape)

    return X, y_label

#-------------wave----------------------------
def piecewise_linear(n_samples=5000,noise=0.01,gap=0.5,random_state=None):
    
    rng = np.random.default_rng(random_state)

    # Oversample so we can remove gap points later
    X = rng.uniform(-2, 2, (n_samples * 2, 2))

    y = np.zeros(len(X), dtype=int)

    left = X[:, 0] < 0
    right = ~left

    # Decision functions
    f_left = X[:, 1] - (X[:, 0] + 1)
    f_right = X[:, 1] - (-X[:, 0])

    # Apply gap as margin shift
    y[left] = f_left[left] > gap
    y[right] = f_right[right] > gap

    # Remove points close to boundary → creates empty band
    keep = np.ones(len(X), dtype=bool)
    keep[left] = np.abs(f_left[left]) > gap
    keep[right] = np.abs(f_right[right]) > gap
    X = X[keep]
    y = y[keep]

    # Add noise
    X += rng.normal(scale=noise, size=X.shape)

    # Trim to requested sample size
    if len(X) > n_samples:
        idx = rng.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y

#-------------Parity Ring-----------------------
def parity_rings(n_samples=5000,noise=0.03,gap=0.15,random_state=None):
    
    rng = np.random.default_rng(random_state)

    # Oversample to compensate removed boundary points
    angles = rng.uniform(0, 2*np.pi, n_samples * 2)
    radius = rng.uniform(0, 2, n_samples * 2)

    X = np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ], axis=1)

    r_scaled = radius * 2
    ring_id = np.floor(r_scaled).astype(int)

    # Distance to nearest integer boundary (class change boundary)
    dist_to_boundary = np.abs(r_scaled - np.round(r_scaled))

    # Keep points away from class boundaries → gap between classes
    keep = dist_to_boundary > gap

    X = X[keep]
    ring_id = ring_id[keep]

    y = (ring_id % 2)

    # Add noise
    X += rng.normal(scale=noise, size=X.shape)

    # Trim to requested size
    if len(X) > n_samples:
        idx = rng.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y

#----------------CheckBoard-------------------
def checkerboard(n_samples=50000, noise=0, grid_size=0.2, gap=0.01, random_state=None):
    rng = np.random.default_rng(random_state)

    # Oversample so we still have enough after removing gap points
    X = rng.uniform(-2, 2, (n_samples * 2, 2))

    # Compute cell index
    gx = X[:, 0] * grid_size
    
    gy = X[:, 1] * grid_size

    fx = gx - np.floor(gx)
    fy = gy - np.floor(gy)

    # Keep points away from borders
    mask = (
        (fx > gap) & (fx < 1 - gap) &
        (fy > gap) & (fy < 1 - gap)
    )

    X = X[mask][:n_samples]

    gx = X[:, 0] * grid_size
    gy = X[:, 1] * grid_size

    y = ((np.floor(gx) + np.floor(gy)) % 2).astype(int)

    X += rng.normal(scale=noise, size=X.shape)

    return X, y

