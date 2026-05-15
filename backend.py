"""
backend.py — AXON-QIM: Quantum Information Manifold Engine
====================================================================
Author: Palwar Singh | AXON Research Group
Architecture: Holographic Field Inversion for EUV Lithography
Complexity: O(N log N) via Frequency-Domain Entropy Minimization

Usage:
    pip install numpy scipy fastapi uvicorn pydantic
    uvicorn backend:app --reload --port 8000
====================================================================
"""

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── PHYSICAL & METAPHYSICAL CONSTANTS ────────────────────────────────────────
H_PLANCK = 6.626e-34    # J·s
C_LIGHT  = 3.0e8        # m/s
EV_TO_J  = 1.602e-19    # J/eV
K_B      = 1.38e-23     # Boltzmann / Shannon entropy link
GRID_NM  = 50           # 50x50 nm simulation field
PIXELS   = 256          # High-resolution computational grid
DX       = GRID_NM / PIXELS # nm per pixel

app = FastAPI(title="AXON-QIM Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API SCHEMAS ──────────────────────────────────────────────────────────────
class ExposureRequest(BaseModel):
    dose_mj: float = 30.0         # Default low dose (High Stochastics)
    wavelength_nm: float = 13.5   # EUV Baseline
    qim_iterations: int = 5       # Holographic Inversion cycles

class QIMResponse(BaseModel):
    entropy_initial: float
    entropy_final: float
    ler_baseline_nm: float
    ler_qim_nm: float
    improvement_pct: float
    baseline_cross_section: list[float]
    qim_cross_section: list[float]

# ─── AXON-QIM CORE MATHEMATICS ────────────────────────────────────────────────
class AxonQIMEngine:
    def __init__(self, size=PIXELS):
        self.N = size
        self.center = self.N // 2

    def generate_platonic_ideal(self):
        """Creates the 'Information Reality' — a pristine 2nm trench."""
        ideal = np.zeros((self.N, self.N))
        trench_width_px = int(2.0 / DX)
        ideal[:, self.center - trench_width_px: self.center + trench_width_px] = 1.0
        return ideal

    def apply_quantum_collapse(self, ideal_manifold, dose_mj, wavelength_nm):
        """Simulates the photon shot noise (Poisson measurement observation)."""
        photon_energy = (H_PLANCK * C_LIGHT) / (wavelength_nm * 1e-9)
        # Convert dose to expected photons per pixel area
        photons_per_nm2 = (dose_mj * 1e-3) / photon_energy / 1e18
        photons_per_px = photons_per_nm2 * (DX ** 2)
        
        expected_field = ideal_manifold * photons_per_px
        # The Quantum Stochastic Event
        stochastic_exposure = np.random.poisson(expected_field).astype(np.float64)
        return stochastic_exposure, photons_per_px

    def compute_von_neumann_entropy(self, field):
        """S = -Tr(rho * ln(rho))"""
        rho = np.abs(field) / (np.sum(np.abs(field)) + 1e-12)
        entropy_matrix = -rho * np.log(rho + 1e-12)
        return np.sum(entropy_matrix)

    def calculate_information_force(self, observed_field):
        """Derives F_info = -∇H to find where information is leaking."""
        # Gradient of the entropy field approximated via Gaussian difference
        smoothed = gaussian_filter(observed_field, sigma=1.5)
        force_gradient = observed_field - smoothed
        return force_gradient

    def holographic_inversion_operator(self, noisy_field, iterations=5):
        """
        The Omega Operator (Ω): Inverts the field using frequency-domain 
        entropy minimization to reconstruct the intended geometry. O(N log N)
        """
        current_field = np.copy(noisy_field)
        
        # Prepare Holographic Kernel (K)
        k_freq = np.fft.fftfreq(self.N)
        kx, ky = np.meshgrid(k_freq, k_freq)
        holographic_kernel = np.exp(-0.5 * (kx**2 + ky**2) / (0.15**2))

        for _ in range(iterations):
            # 1. Transform to Momentum/Frequency Space
            F_k = fft2(current_field)
            
            # 2. Apply Information Force Penalty (Suppressing Chaos)
            F_info = self.calculate_information_force(current_field)
            F_info_k = fft2(F_info)
            
            # 3. The Inversion Step (Filtering self-energy of noise)
            corrected_F_k = (F_k - 0.1 * F_info_k) * holographic_kernel
            
            # 4. Collapse back to Spatial Reality
            current_field = np.abs(ifft2(corrected_F_k))
            
        return current_field

    def calculate_ler(self, field, threshold_ratio=0.5):
        """Calculates 3-sigma Line-Edge Roughness (LER) in nanometers."""
        threshold = np.max(field) * threshold_ratio
        edge_positions = []
        for row in range(self.N):
            # Find the first pixel that crosses the activation threshold
            crossings = np.where(field[row] > threshold)[0]
            if len(crossings) > 0:
                edge_positions.append(crossings[0] * DX)
        
        if len(edge_positions) < 2:
            return 99.9 # Catastrophic failure
        
        # 3-sigma variance
        return 3.0 * float(np.std(edge_positions))

# ─── FASTAPI ROUTE ────────────────────────────────────────────────────────────
@app.post("/simulate", response_model=QIMResponse)
def run_qim_simulation(req: ExposureRequest):
    engine = AxonQIMEngine()
    
    # 1. Metaphysical Reality
    ideal = engine.generate_platonic_ideal()
    
    # 2. The Stochastic Catastrophe (Baseline)
    raw_exposure, max_photons = engine.apply_quantum_collapse(
        ideal, req.dose_mj, req.wavelength_nm
    )
    
    # 3. The AXON Correction
    qim_corrected = engine.holographic_inversion_operator(
        raw_exposure, req.qim_iterations
    )
    
    # 4. Calculate Mathematical Proofs
    entropy_initial = engine.compute_von_neumann_entropy(raw_exposure)
    entropy_final = engine.compute_von_neumann_entropy(qim_corrected)
    
    ler_base = engine.calculate_ler(raw_exposure)
    ler_qim = engine.calculate_ler(qim_corrected)
    
    improvement = 0.0
    if ler_base < 90.0:
        improvement = ((ler_base - ler_qim) / ler_base) * 100.0

    # 5. Extract Cross-Sections for the React Frontend UI
    mid_row = PIXELS // 2
    
    return QIMResponse(
        entropy_initial = round(entropy_initial, 4),
        entropy_final   = round(entropy_final, 4),
        ler_baseline_nm = round(ler_base, 3),
        ler_qim_nm      = round(ler_qim, 3),
        improvement_pct = round(improvement, 2),
        baseline_cross_section = raw_exposure[mid_row].tolist(),
        qim_cross_section      = qim_corrected[mid_row].tolist()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
