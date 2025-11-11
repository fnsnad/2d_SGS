# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import types
from jax_cfd.spectral import utils as spectral_utils

def power_spectrum_2d( field, kx, ky ):
    """
    Compute the isotropic power spectrum of a 2D real-valued field using JAX.
    Args:
        field (jnp.ndarray): 2D real-valued array.
        box_size (float): Physical size of the domain (assumed square).
        bins (int): Number of bins for the isotropic spectrum.

    Returns:
        k_bin_centers (jnp.ndarray): 1D array of wavenumber magnitudes.
        Pk (jnp.ndarray): 1D power spectrum.
    """

    # Compute FFT and power
    if isinstance(field, tuple) or isinstance(field, list):
        Nx, Ny = field[0].shape
        #power = jnp.abs(jnp.fft.rfft2(field[0]))**2 + jnp.abs(jnp.fft.rfft2(field[1]))**2
        power = sum(jnp.abs(jnp.fft.rfft2(u))**2 for u in field)
    else:
        Nx, Ny = field.shape
        power = jnp.abs(jnp.fft.rfft2(field))**2

    # Build wavenumber grids
    k_mag = jnp.sqrt(((2*jnp.pi)*kx)**2 + ((2*jnp.pi)*ky)**2)

    # Flatten and bin by magnitude
    k_flat = k_mag.flatten()
    power_flat = power.flatten()

    bins = Nx//2
    k_bins = np.linspace(1, Nx//2, Nx//2)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    bin_idx = jnp.digitize(k_flat, k_bins) - 1  # bin indices
    # Remove out-of-range indices
    valid = (bin_idx >= 0) & (bin_idx < bins)
    bin_idx = bin_idx[valid]
    power_flat = power_flat[valid]

    # Compute sum and counts per bin
    Pk_sum = jnp.bincount(bin_idx, weights=power_flat, length=bins)
    #counts = jnp.bincount(bin_idx, length=bins)
    
    #Pk_sum /= Nx
    #Pk_sum /= Ny

    return k_bin_centers, Pk_sum[1:]

@jax.jit
def power_spectrum_2d_jax(field: jnp.ndarray, kx: jnp.ndarray, ky: jnp.ndarray):
    """
    Compute the isotropic power spectrum of a 2D scalar or vector field using JAX.

    Args:
        field (jnp.ndarray or tuple/list of jnp.ndarray): Scalar field (Nx, Ny) or vector field.
        kx, ky (jnp.ndarray): Wavenumber grids (Nx, Ny//2 + 1)

    Returns:
        k_bin_centers (jnp.ndarray): 1D array of k magnitudes (bin centers)
        Pk (jnp.ndarray): 1D power spectrum
    """
    def process_vector_field(f_vec):
        fft_result = jax.vmap(jnp.fft.rfft2)(f_vec)  # shape: (n_comp, Nx, Ny//2 + 1)
        power = jnp.sum(jnp.abs(fft_result)**2, axis=0)  # sum over components
        return power

    def process_scalar_field(f):
        return jnp.abs(jnp.fft.rfft2(f))**2

    # Handle scalar or vector field
    if isinstance(field, (tuple, list)):
        field_array = jnp.stack(field, axis=0)
        slice_size = jnp.prod(jnp.array(field_array.shape[1:]))
        field_array /= slice_size
        power = process_vector_field(field_array)
    else:
        field = field/field.size
        power = process_scalar_field(field)

    Nx, Ny_half = power.shape
    bins = Nx // 2

    # === Apply RFFT symmetry correction ===
    # Double power for all frequencies except ky=0 and ky=Ny/2
    ky_indices = jnp.arange(Ny_half)
    correction = jnp.ones(Ny_half)
    correction = correction.at[1:Ny_half - 1].set(2.0)  # double mid-range frequencies
    power = power * correction[None, :]  # broadcast across x

    # === Compute magnitude of wavenumber ===
    k_mag = jnp.sqrt((2 * jnp.pi * kx)**2 + (2 * jnp.pi * ky)**2)
    k_flat = k_mag.flatten()
    power_flat = power.flatten()

    # === Bin edges and centers ===
    k_bins = jnp.linspace(1.0, float(bins), bins)
    k_bin_edges = jnp.concatenate([jnp.array([0.0]), k_bins])
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    # Manual digitize
    def digitize(x, edges):
        return jnp.sum(x[..., None] >= edges, axis=-1) - 1

    bin_idx = digitize(k_flat, k_bin_edges)
    in_bounds = (bin_idx >= 0) & (bin_idx < bins)
    safe_bin_idx = jnp.where(in_bounds, bin_idx, 0)
    safe_power_flat = jnp.where(in_bounds, power_flat, 0.0)

    # Bin accumulation
    Pk_sum = jnp.bincount(safe_bin_idx, weights=safe_power_flat, length=bins)
    #counts = jnp.bincount(safe_bin_idx, weights=in_bounds.astype(jnp.float32), length=bins)
    #Pk_avg = jnp.where(counts > 0, Pk_sum / counts, 0.0)

    return power, Pk_sum



def make_flux_fn( model: time_stepping.ImplicitExplicitODE ):
#R: types.Array,
                  
    # Unpack the model
    Nx, Ny = model.grid.shape
    rhs = model.explicit_terms
    if model.k_filter is not None:
       filter_ = model.filter_
    else:
        filter_ = 1

    R = jnp.sqrt(model.kx**2 + model.ky**2)*2*jnp.pi
    # Create the filter kernel
    G_fn = lambda Delta: jnp.exp(- (R**2) * (Delta**2) / 24.0)
 
    def flux_fn_(state, Delta):
        
        # State should be in fourier space
        G = G_fn(Delta)
        rhs_state = rhs(state)
        rhs_state_G = rhs(state * G)
        rhs_diff = rhs_state* G - rhs_state_G
        rhs_diff *= filter_ 
        # Any linear term should drop out
        filtered_dEdt = jnp.fft.irfft2(rhs_diff, s = (Nx,Ny)) * jnp.fft.irfft2(state*G, s = (Nx,Ny)) 
        # Filter state and rhs  
        flux = jnp.mean(filtered_dEdt)
        rhs_diff = rhs_state * G *filter_
        filtered_dEdt = jnp.fft.irfft2(rhs_diff, s = (Nx,Ny)) * jnp.fft.irfft2(state*G, s = (Nx,Ny))
        #rhs_eval *= G
        #state *= G
        #E_diff = E_unfilt - jnp.fft.irfft2(state, s = (Nx,Ny)) * jnp.fft.irfft2(rhs_eval, s = (Nx,Ny))
        # Return flux and spatial profile
        return flux, filtered_dEdt

    return flux_fn_

def make_flux_dns_to_les_fn( model: time_stepping.ImplicitExplicitODE,
                             les_Delta: float ):

    # Unpack the model
    Nx, Ny = model.grid.shape
    rhs = model.explicit_terms
    if model.k_filter is not None:
       filter_ = model.filter_
    else:
        filter_ = 1

    R = jnp.sqrt(model.kx**2 + model.ky**2)*2*jnp.pi
    # Create the filter kernel
    G1 = jnp.exp(- (R**2) * (les_Delta**2) / 24.0)
    G_fn = lambda Delta: jnp.exp(- (R**2) * (Delta**2) / 24.0)
 
    def flux_fn_(state, Delta):
        
        # State should be in fourier space
        state *= G1
        G = G_fn(Delta)
        rhs_state = rhs(state)
        rhs_state_G = rhs(state * G)
        rhs_diff = rhs_state* G #- rhs_state_G
        rhs_diff *= filter_ 
        # Any linear term should drop out
        filtered_dEdt = jnp.fft.irfft2(rhs_diff, s = (Nx,Ny)) * jnp.fft.irfft2(state*G, s = (Nx,Ny)) 
        # Filter state and rhs  
        #rhs_eval *= G
        #state *= G
        #E_diff = E_unfilt - jnp.fft.irfft2(state, s = (Nx,Ny)) * jnp.fft.irfft2(rhs_eval, s = (Nx,Ny))
        # Return flux and spatial profile
        return jnp.mean(filtered_dEdt), filtered_dEdt

    return flux_fn_

def LCR_fluxes_fn(grid: types.grids.Grid ):

    kx, ky = grid.rfft_mesh()
    w_to_v = spectral_utils.vorticity_to_velocity(grid)
    two_pi_i = 2j*jnp.pi
    Nx,Ny = grid.shape

    def LCR_flux_fn( w_hat , Delta,
                     dealias: bool = False):
        
        
        # Create the filter kernel
        R = jnp.sqrt(kx**2 + ky**2)*2*jnp.pi
        G = jnp.exp(- (R**2) * (Delta**2) / 24.0)
        
        k_dealias = 2*jnp.pi/Delta*2
        mask = spectral_utils.brick_wall_filter_2d(grid,k_dealias/Nx)

        u, v = w_to_v( w_hat )

        u_bar = u*G
        v_bar = v*G
        u_pr = (1-G)*u
        v_pr = (1-G)*v

        if dealias:
            u_pr *= mask
            v_pr *= mask

        Sxx = jnp.fft.irfft2( two_pi_i * kx * u_bar )
        Syy = jnp.fft.irfft2( two_pi_i * ky * v_bar )
        Sxy = jnp.fft.irfft2(  two_pi_i *(kx * v_bar + ky * u_bar) )/2


        u = jnp.fft.irfft2(u_bar)
        v = jnp.fft.irfft2(v_bar)

        uu = jnp.fft.rfft2( u ** 2 )
        vv = jnp.fft.rfft2( v ** 2 )
        uv = jnp.fft.rfft2( u * v )

        # Compute trace(SL)
        Lxx = uu*G - jnp.fft.rfft2( jnp.fft.irfft2(u_bar*G)**2 )
        Lyy = vv*G - jnp.fft.rfft2( jnp.fft.irfft2(v_bar*G)**2 )
        Lxy = uv*G - jnp.fft.rfft2( jnp.fft.irfft2(v_bar*G)*jnp.fft.irfft2(u_bar*G) )

        Lxx  = jnp.fft.irfft2(Lxx)
        Lyy  = jnp.fft.irfft2(Lyy)
        Lxy  = jnp.fft.irfft2(Lxy)

        trSL = jnp.mean( Sxx*Lxx + Syy*Lyy + 2*Sxy*Lxy )

        # Compute trace(SC)
        up = jnp.fft.irfft2(u_pr)
        vp = jnp.fft.irfft2(v_pr)

        uu = 2*jnp.fft.rfft2( u * up )
        vv = 2*jnp.fft.rfft2( v * vp )
        uv = jnp.fft.rfft2( u * vp + v * up )

        Cxx = uu*G - 2*jnp.fft.rfft2( jnp.fft.irfft2(u_bar*G)*jnp.fft.irfft2(u_pr*G) )
        Cyy = vv*G - 2*jnp.fft.rfft2( jnp.fft.irfft2(v_bar*G)*jnp.fft.irfft2(v_pr*G) )
        Cxy = uv*G - jnp.fft.rfft2( jnp.fft.irfft2(u_bar*G)*jnp.fft.irfft2(v_pr*G) + jnp.fft.irfft2(v_bar*G)*jnp.fft.irfft2(u_pr*G) )
        
        if dealias:
            Cxx *= mask
            Cyy *= mask
            Cxy *= mask

        Cxx  = jnp.fft.irfft2(Cxx)
        Cyy  = jnp.fft.irfft2(Cyy)
        Cxy  = jnp.fft.irfft2(Cxy)

        trSC = jnp.mean( Sxx*Cxx + Syy*Cyy + 2*Cxy*Sxy )
        
        uu = jnp.fft.rfft2( up * up )
        vv = jnp.fft.rfft2( vp * vp )
        uv = jnp.fft.rfft2( up * vp )

        Rxx = uu*G - jnp.fft.rfft2( jnp.fft.irfft2(u_pr*G)**2 )
        Ryy = vv*G - jnp.fft.rfft2( jnp.fft.irfft2(v_pr*G)**2 )
        Rxy = uv*G - jnp.fft.rfft2( jnp.fft.irfft2(v_pr*G)*jnp.fft.irfft2(u_pr*G) )

        if dealias:
            Rxx *= mask
            Ryy *= mask
            Rxy *= mask

        Rxx  = jnp.fft.irfft2(Rxx)
        Ryy  = jnp.fft.irfft2(Ryy)
        Rxy  = jnp.fft.irfft2(Rxy)

        trSR = jnp.mean( Sxx*Rxx + Syy*Ryy + 2*Rxy*Sxy )

        return trSL, trSC, trSR    
    
    return LCR_flux_fn