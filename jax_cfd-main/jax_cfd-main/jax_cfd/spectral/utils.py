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

"""Helper functions for building pseudospectral methods."""

from typing import Callable, Tuple
from jax import debug, vmap, jit
import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.spectral import types as spectral_types
from jax_cfd.spectral.types import Array
import jax.numpy.fft as jnp_fft

def truncated_rfft(u: spectral_types.Array) -> spectral_types.Array:
  """Applies the 2/3 rule by truncating higher Fourier modes.

  Args:
    u: the real-space representation of the input signal

  Returns:
    Downsampled version of `u` in rfft-space.
  """
  uhat = jnp.fft.rfft(u)
  k, = uhat.shape
  final_size = int(2 / 3 * k) + 1
  return 2 / 3 * uhat[:final_size]


def padded_irfft(uhat: spectral_types.Array) -> spectral_types.Array:
  """Applies the 3/2 rule by padding with zeros.

  Args:
    uhat: the rfft representation of a signal

  Returns:
    An upsampled signal in real space which 3/2 times larger than the input
    signal `uhat`.
  """
  n, = uhat.shape
  final_shape = int(3 / 2 * n)
  smoothed = jnp.pad(uhat, (0, final_shape - n))
  assert smoothed.shape == (final_shape,), "incorrect padded shape"
  return 1.5 * jnp.fft.irfft(smoothed)


def truncated_fft_2x(u: spectral_types.Array) -> spectral_types.Array:
  """Applies the 1/2 rule to complex u by truncating higher Fourier modes.

  Args:
    u: the (complex) input signal

  Returns:
    Downsampled version of `u` in fft-space.
  """
  uhat = jnp.fft.fftshift(jnp.fft.fft(u))
  k, = uhat.shape
  final_size = (k + 1) // 2
  return jnp.fft.ifftshift(uhat[final_size // 2:(-final_size + 1) // 2]) / 2


def padded_ifft_2x(uhat: spectral_types.Array) -> spectral_types.Array:
  """Applies the 2x rule to complex F[u] by padding higher frequencies.

     Pads with zeros in the Fourier domain before performing the ifft
      (effectively performing 2x interpolation in the spatial domain)

  Args:
    uhat: the fft representation of signal

  Returns:
    An upsampled signal in real space interpolated to 2x more points than
    `jax.fft.ifft(uhat)`.
  """
  n, = uhat.shape
  final_size = n + 2 * (n // 2)
  added = n // 2
  smoothed = jnp.pad(jnp.fft.fftshift(uhat), (added, added))
  assert smoothed.shape == (final_size,), "incorrect padded shape"
  return 2 * jnp.fft.ifft(jnp.fft.ifftshift(smoothed))


def circular_filter_2d(grid: grids.Grid) -> spectral_types.Array:
  """Circular filter which roughly matches the 2/3 rule but is smoother.

  Follows the technique described in Equation 1 of [1]. We use a different value
  for alpha as used by pyqg [2].

  Args:
    grid: the grid to filter over

  Returns:
    Filter mask

  Reference:
    [1] Arbic, Brian K., and Glenn R. Flierl. "Coherent vortices and kinetic
    energy ribbons in asymptotic, quasi two-dimensional f-plane turbulence."
    Physics of Fluids 15, no. 8 (2003): 2177-2189.
    https://doi.org/10.1063/1.1582183

    [2] Ryan Abernathey, rochanotes, Malte Jansen, Francis J. Poulin, Navid C.
    Constantinou, Dhruv Balwada, Anirban Sinha, Mike Bueti, James Penn,
    Christopher L. Pitt Wolfe, & Bia Villas Boas. (2019). pyqg/pyqg: v0.3.0
    (v0.3.0). Zenodo. https://doi.org/10.5281/zenodo.3551326.
    See:
    https://github.com/pyqg/pyqg/blob/02e8e713660d6b2043410f2fef6a186a7cb225a6/pyqg/model.py#L136
  """
  kx, ky = grid.rfft_mesh()
  max_k = ky[-1, -1]

  circle = jnp.sqrt(kx**2 + ky**2)
  cphi = 0.65 * max_k
  filterfac = 23.6
  filter_ = jnp.exp(-filterfac * (circle - cphi)**4.)
  filter_ = jnp.where(circle <= cphi, jnp.ones_like(filter_), filter_)
  return filter_


def brick_wall_filter_2d(grid: grids.Grid,
                       k_filter: float = 2/3
                       ):
  """Low-pass brick wall filter in spectral space with configurable cutoff.
  Args:
      grid: A grids.Grid object with a `shape` attribute (n, m).
      k_filter: The fraction of the modes to retain (e.g., 2/3 for 2/3 rule).
  Returns:
      A 2D binary filter mask for spectral coefficients.
  """

  if grid.ndim == 2:
      n, m = grid.shape
      kx, ky = grid.rfft_mesh()
      # Calculate the maximum number of modes to keep
      #n_cutoff = int(k_filter * n) // 2
      #m_cutoff = int(k_filter * (m // 2 + 1))

      #filter_ = jnp.zeros((n, m // 2 + 1))
      # Set ones in the central band (low-frequency modes)
      #filter_ = filter_.at[:n_cutoff, :m_cutoff].set(1)
      #filter_ = filter_.at[-n_cutoff:, :m_cutoff].set(1)
      k_mag = jnp.sqrt(kx**2 + ky**2)
      kx_max = jnp.max(jnp.abs(kx))
      ky_max = jnp.max(jnp.abs(ky))
      k_max = k_filter * jnp.minimum(kx_max, ky_max)

      filter_ = (k_mag <= k_max).astype(kx.dtype)
      return filter_
  elif grid.ndim == 3:
      kx, ky, kz = grid.rfft_mesh()
      k_mag = jnp.sqrt(kx**2 + ky**2 + kz**2)
      kx_max = jnp.max(jnp.abs(kx))
      ky_max = jnp.max(jnp.abs(ky))
      kz_max = jnp.max(jnp.abs(kz))
      k_max = k_filter * jnp.minimum(kx_max, jnp.minimum(kz_max, ky_max) )
      filter_ = (k_mag <= k_max).astype(kx.dtype)
      return filter_


def exponential_filter(signal, alpha=1e-6, order=2):
  """Apply a low-pass smoothing filter to remove noise from 2D signal."""
  # Based on:
  # 1. Gottlieb and Hesthaven (2001), "Spectral methods for hyperbolic problems"
  # https://doi.org/10.1016/S0377-0427(00)00510-0
  # 2. Also, see https://arxiv.org/pdf/math/0701337.pdf --- Eq. 5

  # TODO(dresdner) save a few ffts by factoring out the actual filter, sigma.
  alpha = -jnp.log(alpha)
  n, _ = signal.shape  # TODO(dresdner) check square / handle 1D case
  kx, ky = jnp.fft.fftfreq(n), jnp.fft.rfftfreq(n)
  kx, ky = jnp.meshgrid(kx, ky, indexing="ij")
  eta = jnp.sqrt(kx**2 + ky**2)
  sigma = jnp.exp(-alpha * eta**(2 * order))
  return jnp.fft.irfft2(sigma * jnp.fft.rfft2(signal))


def vorticity_to_velocity(
    grid: grids.Grid
) -> Callable[[spectral_types.Array], Tuple[spectral_types.Array,
                                            spectral_types.Array]]:
  """Constructs a function for converting vorticity to velocity, both in Fourier domain.

  Solves for the stream function and then uses the stream function to compute
  the velocity. This is the standard approach. A quick sketch can be found in
  [1].

  Args:
    grid: the grid underlying the vorticity field.

  Returns:
    A function that takes a vorticity (rfftn) and returns a velocity vector
    field.

  Reference:
    [1] Z. Yin, H.J.H. Clercx, D.C. Montgomery, An easily implemented task-based
    parallel scheme for the Fourier pseudospectral solver applied to 2D
    Navier–Stokes turbulence, Computers & Fluids, Volume 33, Issue 4, 2004,
    Pages 509-520, ISSN 0045-7930,
    https://doi.org/10.1016/j.compfluid.2003.06.003.
  """
  kx, ky = grid.rfft_mesh()
  two_pi_i = 2 * jnp.pi * 1j
  laplace = two_pi_i ** 2 * (abs(kx)**2 + abs(ky)**2)
  laplace = laplace.at[0, 0].set(1)  # pytype: disable=attribute-error  # jnp-type

  def ret(vorticity_hat):
    psi_hat = -1 / laplace * vorticity_hat
    vxhat = two_pi_i * ky * psi_hat
    vyhat = -two_pi_i * kx * psi_hat
    return vxhat, vyhat

  return ret


def filter_step(step_fn: spectral_types.StepFn, filter_: spectral_types.Array):
  """Returns a filtered version of the step_fn."""
  def new_step_fn(state):
    return filter_ * step_fn(state)
  return new_step_fn


def spectral_curl_2d(mesh, velocity_hat):
  """Computes the 2D curl in the Fourier basis."""
  kx, ky = mesh
  uhat, vhat = velocity_hat
  return 2j * jnp.pi * (vhat * kx - uhat * ky)

def velocity_gradients_fft(vx, vy, kx, ky):
    vx_hat = jnp.fft.rfft2(vx)
    vy_hat = jnp.fft.rfft2(vy)

    dvx_dx_hat = 2j * jnp.pi * kx * vx_hat
    dvx_dy_hat = 2j * jnp.pi * ky * vx_hat
    dvy_dx_hat = 2j * jnp.pi * kx * vy_hat
    dvy_dy_hat = 2j * jnp.pi * ky * vy_hat

    dvx_dx = jnp.fft.irfft2(dvx_dx_hat, s=vx.shape)
    dvx_dy = jnp.fft.irfft2(dvx_dy_hat, s=vx.shape)
    dvy_dx = jnp.fft.irfft2(dvy_dx_hat, s=vy.shape)
    dvy_dy = jnp.fft.irfft2(dvy_dy_hat, s=vy.shape)

    return dvx_dx, dvx_dy, dvy_dx, dvy_dy

def velocity_second_derivatives_fft(vx, vy, kx, ky):
    vx_hat = jnp.fft.rfft2(vx)
    vy_hat = jnp.fft.rfft2(vy)

    d2vx_dx2_hat = (2j * jnp.pi * kx)**2 * vx_hat
    d2vx_dy2_hat = (2j * jnp.pi * ky)**2 * vx_hat
    d2vx_dxdy_hat = (2j * jnp.pi * kx) * (2j * jnp.pi * ky) * vx_hat

    d2vy_dx2_hat = (2j * jnp.pi * kx)**2 * vy_hat
    d2vy_dy2_hat = (2j * jnp.pi * ky)**2 * vy_hat
    d2vy_dxdy_hat = (2j * jnp.pi * kx) * (2j * jnp.pi * ky) * vy_hat

    d2vx_dx2 = jnp.fft.irfft2(d2vx_dx2_hat, s=vx.shape)
    d2vx_dy2 = jnp.fft.irfft2(d2vx_dy2_hat, s=vx.shape)
    d2vx_dxdy = jnp.fft.irfft2(d2vx_dxdy_hat, s=vx.shape)

    d2vy_dx2 = jnp.fft.irfft2(d2vy_dx2_hat, s=vy.shape)
    d2vy_dy2 = jnp.fft.irfft2(d2vy_dy2_hat, s=vy.shape)
    d2vy_dxdy = jnp.fft.irfft2(d2vy_dxdy_hat, s=vy.shape)

    return d2vx_dx2, d2vx_dy2, d2vx_dxdy, d2vy_dx2, d2vy_dy2, d2vy_dxdy

def velocity_derivatives_fft(u_hat, v_hat, kx, ky):
    two_pi_i = 2j * jnp.pi
    Dx = two_pi_i * kx
    Dy = two_pi_i * ky
    kx2 = (two_pi_i * kx)**2
    ky2 = (two_pi_i * ky)**2
    kxky = (two_pi_i * kx) * (two_pi_i * ky)

    # First-order derivatives
    du_dx_hat = Dx * u_hat
    du_dy_hat = Dy * u_hat
    dv_dx_hat = Dx * v_hat
    dv_dy_hat = Dy * v_hat
    du_dx = jnp.fft.irfft2(du_dx_hat)
    du_dy = jnp.fft.irfft2(du_dy_hat)
    dv_dx = jnp.fft.irfft2(dv_dx_hat)
    dv_dy = jnp.fft.irfft2(dv_dy_hat)

    # Second-order derivatives
    d2u_dx2_hat = kx2 * u_hat
    d2u_dy2_hat = ky2 * u_hat
    d2u_dxdy_hat = kxky * u_hat
    d2v_dx2_hat = kx2 * v_hat
    d2v_dy2_hat = ky2 * v_hat
    d2v_dxdy_hat = kxky * v_hat
    d2u_dx2 = jnp.fft.irfft2(d2u_dx2_hat)
    d2u_dy2 = jnp.fft.irfft2(d2u_dy2_hat)
    d2u_dxdy = jnp.fft.irfft2(d2u_dxdy_hat)
    d2v_dx2 = jnp.fft.irfft2(d2v_dx2_hat)
    d2v_dy2 = jnp.fft.irfft2(d2v_dy2_hat)
    d2v_dxdy = jnp.fft.irfft2(d2v_dxdy_hat)

    return du_dx, du_dy, dv_dx, dv_dy, d2u_dx2, d2u_dy2, d2u_dxdy, d2v_dx2, d2v_dy2, d2v_dxdy

def divergence_of_sgs_stress_fft(tau_xx, tau_xy, tau_yy, kx, ky):
    tau_xx_hat = jnp.fft.rfft2(tau_xx)
    tau_xy_hat = jnp.fft.rfft2(tau_xy)
    tau_yy_hat = jnp.fft.rfft2(tau_yy)

    d_tau_xx_dx_hat = 2j * jnp.pi * kx * tau_xx_hat
    d_tau_xy_dy_hat = 2j * jnp.pi * ky * tau_xy_hat
    d_tau_xy_dx_hat = 2j * jnp.pi * kx * tau_xy_hat
    d_tau_yy_dy_hat = 2j * jnp.pi * ky * tau_yy_hat

    d_tau_xx_dx = jnp.fft.irfft2(d_tau_xx_dx_hat, s=tau_xx.shape)
    d_tau_xy_dy = jnp.fft.irfft2(d_tau_xy_dy_hat, s=tau_xy.shape)
    d_tau_xy_dx = jnp.fft.irfft2(d_tau_xy_dx_hat, s=tau_xy.shape)
    d_tau_yy_dy = jnp.fft.irfft2(d_tau_yy_dy_hat, s=tau_yy.shape)

    div_tau_x = d_tau_xx_dx + d_tau_xy_dy
    div_tau_y = d_tau_xy_dx + d_tau_yy_dy

    return div_tau_x, div_tau_y

@jit
def fourier_filter(field, Delta):

    if field.ndim not in (2, 3, 4):
        raise ValueError("field must have 2D, 3D, or 4D shape")

    Nx, Ny = field.shape[:2]

    qx = jnp.fft.fftfreq(Nx) * Nx
    qy = jnp.fft.fftfreq(Ny) * Ny
    X, Y = jnp.meshgrid(qy, qx)
    R2 = X**2 + Y**2
    G = jnp.exp(-R2 * (Delta**2) / 24.0).astype(field.dtype)

    def filt2(x2d):
        F = jnp.fft.fft2(x2d, axes=(-2, -1))
        Y = F * G
        y = jnp.fft.ifft2(Y, axes=(-2, -1))
        return y.real.astype(field.dtype) if jnp.isrealobj(field) else y.astype(field.dtype)

    if field.ndim == 2:
        return filt2(field)

    elif field.ndim == 3:  # (Nx,Ny,C)
        return vmap(filt2, in_axes=2, out_axes=2)(field)

    else:  # (Nx,Ny,C,T)
        def filt3(x3d):  # x3d = (Nx,Ny,C)
            return vmap(filt2, in_axes=2, out_axes=2)(x3d)
        return vmap(filt3, in_axes=3, out_axes=3)(field)

def get_filter_fn( Delta, kx, ky ):
    
    # Calculate radial frequencies
    R = jnp.sqrt(kx**2 + ky**2)*2*jnp.pi
    # Create the filter kernel
    G = jnp.exp(- (R**2) * (Delta**2) / 24.0)
 
    # Apply the filter
    #filtered_field = jnp.zeros_like(field)
 
    def filter_fn(state):
       return state * G[(...,) + (None,) * (state.ndim - G.ndim)]
    
    return filter_fn


def down_res(x: jnp.ndarray, new_shape: Tuple) -> jnp.ndarray:
    """
    Downsample a 2D or 3D real input x from its current shape (Mx, My) or (Mx, My, L)
    to (Nx, Ny, L) (if 3D) or (Nx, Ny) (if 2D) by spectral truncation.
    Nx and Ny must be even and smaller than Mx, My.
    """
    if not jnp.isrealobj(x):
          x = jnp.fft.irfft2(x, axes=(0, 1)) # Transform to real space

    Mx, My = x.shape[:2]  # Get the first two dimensions
    is_3d = len(x.shape) == 3  # Check if the input is 3D
    
    Nx, Ny = new_shape

    assert Nx <= Mx and Ny <= My, "Output resolution must be <= input resolution"
    assert Mx % 2 == 0 and My % 2 == 0 and Nx % 2 == 0 and Ny % 2 == 0, "Only even dimensions supported"
 
    factor = (My / Ny) * (Mx / Nx)
    
    if is_3d:
        L = x.shape[2]  # The third dimension size

        x_hat = jnp.fft.rfft2( x, axes = (0,1) )
        x_hat_trunc = jnp.zeros((Nx, Ny // 2 + 1, L), dtype=x_hat.dtype)  # Prepare truncated array
        
        # Select low-frequency components
        x_hat_trunc = x_hat_trunc.at[:Nx // 2, ...].set(x_hat[:Nx // 2, :Ny // 2 + 1, :])
        x_hat_trunc = x_hat_trunc.at[-Nx // 2:, ...].set(x_hat[-Nx // 2:, :Ny // 2 + 1, :])
        
        # Inverse FFT and scaling
        x_downsampled = x_hat_trunc / factor
        
    else:
        # For 2D array: No third dimension, handle it as a normal 2D case
        x_hat = jnp.fft.rfft2(x)  # 2D Fourier transform
        x_hat_trunc = jnp.zeros((Nx, Ny // 2 + 1), dtype=x_hat.dtype)  # Prepare truncated array
        
        # Select low-frequency components
        x_hat_trunc = x_hat_trunc.at[:Nx // 2, :].set(x_hat[:Nx // 2, :Ny // 2 + 1])
        x_hat_trunc = x_hat_trunc.at[-Nx // 2:, :].set(x_hat[-Nx // 2:, :Ny // 2 + 1])
        
        # Inverse FFT and scaling
        x_downsampled = x_hat_trunc / factor
    
    return x_downsampled

def up_res(x: jnp.ndarray, new_shape: Tuple[int, int]) -> jnp.ndarray:
    """
    Upsample a 2D or 3D real input x from its current shape (Nx, Ny) or (Nx, Ny, L)
    to (Mx, My, L) (if 3D) or (Mx, My) (if 2D) by zero-padding in Fourier space.
    Nx and Ny must be even and smaller than Mx, My.
    """
    if not jnp.isrealobj(x):
        x = jnp.fft.irfft2(x, axes=(0, 1))  # Convert to real space if in Fourier domain

    Nx, Ny = x.shape[:2]
    Mx, My = new_shape
    is_3d = len(x.shape) == 3

    assert Mx >= Nx and My >= Ny, "Upsampled resolution must be >= input resolution"
    assert all(d % 2 == 0 for d in (Nx, Ny, Mx, My)), "Only even dimensions supported"

    factor = (My / Ny) * (Mx / Nx)

    if is_3d:
        L = x.shape[2]
        x_hat = jnp.fft.rfft2(x, axes=(0, 1))
        x_hat_padded = jnp.zeros((Mx, My // 2 + 1, L), dtype=x_hat.dtype)

        # Determine valid frequency indices for copy
        min_x = min(Nx // 2, Mx // 2)
        min_y = min(Ny // 2 + 1, My // 2 + 1)

        x_hat_padded = x_hat_padded.at[:min_x, :min_y, :].set(x_hat[:min_x, :min_y, :])
        x_hat_padded = x_hat_padded.at[-min_x:, :min_y, :].set(x_hat[-min_x:, :min_y, :])

        x_upsampled = jnp.fft.irfft2(factor * x_hat_padded, s=(Mx, My), axes=(0, 1))

    else:
        x_hat = jnp.fft.rfft2(x)
        x_hat_padded = jnp.zeros((Mx, My // 2 + 1), dtype=x_hat.dtype)

        min_x = min(Nx // 2, Mx // 2)
        min_y = min(Ny // 2 + 1, My // 2 + 1)

        x_hat_padded = x_hat_padded.at[:min_x, :min_y].set(x_hat[:min_x, :min_y])
        x_hat_padded = x_hat_padded.at[-min_x:, :min_y].set(x_hat[-min_x:, :min_y])

        x_upsampled = factor * x_hat_padded
        #x_upsampled = jnp.fft.irfft2(factor * x_hat_padded, s=(Mx, My))

    return x_upsampled

def chebyshev_forward_transform(v: Array, axis: int = -1) -> Array:
    """Compute forward Chebyshev transform using FFT.
    
    This implements the forward Chebyshev transform using the FFT method:
    1. Extend data to length 2N with V_{2N-j} = v_j for j=1,2,...,N-1
    2. Use FFT to compute Fourier coefficients
    3. Extract the first N+1 coefficients (k=0,1,...,N)
    
    Args:
        v: Input array with data at Chebyshev-Lobatto points
        axis: Axis along which to transform
        
    Returns:
        Array containing the Chebyshev coefficients
    """
    axis = axis if axis >= 0 else v.ndim + axis
    N = v.shape[axis] - 1  # N is the degree, so size is N+1
    
    # Step 1: Extend data to length 2N
    # V_{2N-j} = v_j for j=1,2,...,N-1
    extended_shape = list(v.shape)
    extended_shape[axis] = 2 * N
    V = jnp.zeros(extended_shape, dtype=v.dtype)
    
    # Copy original data: V_j = v_j for j=0,1,...,N
    slices_orig = [slice(None)] * v.ndim
    slices_orig[axis] = slice(0, N+1)
    V = V.at[tuple(slices_orig)].set(v)
    
    # Extend with reflection: V_{2N-j} = v_j for j=1,2,...,N-1
    for j in range(1, N):
        slices_ext = [slice(None)] * v.ndim
        slices_ext[axis] = 2*N - j
        slices_src = [slice(None)] * v.ndim
        slices_src[axis] = j
        V = V.at[tuple(slices_ext)].set(v[tuple(slices_src)])
    
    # Step 2: Compute FFT of extended data
    V_hat = jnp_fft.fft(V, axis=axis)
    
    # Step 3: Extract the first N+1 coefficients (k=0,1,...,N)
    result_slices = [slice(None)] * v.ndim
    result_slices[axis] = slice(0, N+1)
    result = V_hat[tuple(result_slices)]
    
    # Apply normalization: 1/(2N) for all coefficients
    result = result / (2 * N)
    
    return result


def chebyshev_backward_transform(v_hat: Array, 
                                 axis: int = -1) -> Array:
    """Compute backward Chebyshev transform using FFT.
    
    This implements the backward Chebyshev transform using the FFT method:
    1. Extend coefficients to length 2N with V_hat_{2N-k} = v_hat_k for k=1,2,...,N-1
    2. Use inverse FFT to compute values on equispaced grid
    3. Extract the first N+1 values (j=0,1,...,N)
    
    Args:
        v_hat: Input array with Chebyshev coefficients
        axis: Axis along which to transform
        
    Returns:
        Array containing the values at Chebyshev-Lobatto points
    """
    axis = axis if axis >= 0 else v_hat.ndim + axis
    N = v_hat.shape[axis] - 1  # N is the degree, so size is N+1
    
    # Step 1: Extend coefficients to length 2N
    # V_hat_{2N-k} = v_hat_k for k=1,2,...,N-1
    extended_shape = list(v_hat.shape)
    extended_shape[axis] = 2 * N
    V_hat_ext = jnp.zeros(extended_shape, dtype=v_hat.dtype)
    
    # Copy original coefficients: V_hat_k = v_hat_k for k=0,1,...,N
    slices_orig = [slice(None)] * v_hat.ndim
    slices_orig[axis] = slice(0, N+1)
    V_hat_ext = V_hat_ext.at[tuple(slices_orig)].set(v_hat)
    
    # Extend with reflection: V_hat_{2N-k} = v_hat_k for k=1,2,...,N-1
    for k in range(1, N):
        slices_ext = [slice(None)] * v_hat.ndim
        slices_ext[axis] = 2*N - k
        slices_src = [slice(None)] * v_hat.ndim
        slices_src[axis] = k
        V_hat_ext = V_hat_ext.at[tuple(slices_ext)].set(v_hat[tuple(slices_src)])
    
    # Step 2: Compute inverse FFT of extended coefficients
    V = jnp_fft.ifft(V_hat_ext, axis=axis)
    
    # Step 3: Extract the first N+1 values (j=0,1,...,N)
    result_slices = [slice(None)] * v_hat.ndim
    result_slices[axis] = slice(0, N+1)
    result = V[tuple(result_slices)]
    
    return result
