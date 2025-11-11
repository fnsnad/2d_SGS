"""Forcing functions for spectral equations."""

import jax
import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.spectral import types
from typing import Tuple

def random_forcing_module(grid: grids.Grid,
                          seed: int = 0,
                          n: int = 20,
                          offset=(0,)):
  """Implements the forcing described in Bar-Sinai et al. [*].

  Args:
    grid: grid to use for the x-axis
    seed: random seed for computing the random waves
    n: number of random waves to use
    offset: offset for the x-axis. Defaults to (0,) for the Fourier basis.
  Returns:
    Time dependent forcing function.

  [*] Bar-Sinai, Yohai, Stephan Hoyer, Jason Hickey, and Michael P. Brenner.
  "Learning data-driven discretizations for partial differential equations."
  Proceedings of the National Academy of Sciences 116, no. 31 (2019):
  15344-15349.
  """

  key = jax.random.PRNGKey(seed)

  ks = jnp.array([3, 4, 5, 6])

  key, subkey = jax.random.split(key)
  kx = jax.random.choice(subkey, ks, shape=(n,))

  key, subkey = jax.random.split(key)
  amplitude = jax.random.uniform(subkey, minval=-0.5, maxval=0.5, shape=(n,))

  key, subkey = jax.random.split(key)
  omega = jax.random.uniform(subkey, minval=-0.4, maxval=0.4, shape=(n,))

  key, subkey = jax.random.split(key)
  phi = jax.random.uniform(subkey, minval=0, maxval=2 * jnp.pi, shape=(n,))

  xs, = grid.axes(offset=offset)

  def forcing_fn(t):

    @jnp.vectorize
    def eval_force(x):
      f = amplitude * jnp.sin(omega * t - x * kx + phi)
      return f.sum()

    return eval_force(xs)

  return forcing_fn

def shear_background(
    grid: grids.Grid,
    scale: float = 1,
    swap_xy: bool = False) -> types.VelocityFn:
  """Returns a shear background forcing function."""

  Lx,Ly = grid.domain
  if grid.ndim == 2:
    x, y = grid.mesh()
    if swap_xy:
      dx = 2 * (x - Lx[0]) / (Lx[1] - Lx[0]) - 1
      v = grids.GridArray( scale*( dx ), grid.cell_faces[1], grid)
      u = grids.GridArray( jnp.zeros_like(v.data), grid.cell_faces[0], grid)
      w = -scale*jnp.ones_like(u) # w = dv/dx - du/dy
      f = (u, v)
    else:
      dy = 2 * (y - Ly[0]) / (Ly[1] - Ly[0]) - 1
      u = grids.GridArray( scale*( dy ), grid.cell_faces[0], grid)
      v = grids.GridArray(jnp.zeros_like(u.data), grid.cell_faces[1], grid)
      w = scale*jnp.ones_like(u)
      f = (u, v) 
  else:
    raise NotImplementedError("Only 2D grids are supported.")

  def vorticity():
        return w

  def forcing_fn():
    return f

  forcing_fn.vorticity = vorticity # Return the vorticity as an attribute function
  return forcing_fn


def linear_adv(
    grid: grids.Grid,
    scale: float = 1,
    swap_xy: bool = False) -> types.VelocityFn:
  """Returns a linear adv background function."""

  if grid.ndim == 2:
    x, y = grid.mesh()
    if swap_xy:
      v = grids.GridArray( scale * jnp.ones_like(x), grid.cell_faces[1], grid)
      u = grids.GridArray(jnp.zeros_like(v.data), grid.cell_faces[0], grid)
      f = (u, v)
    else:
      u = grids.GridArray( - scale * jnp.ones_like(x), grid.cell_faces[0], grid)
      v = grids.GridArray(jnp.zeros_like(u.data), grid.cell_faces[1], grid)
      f = (u, v) 
  else:
    raise NotImplementedError("Only 2D grids are supported.")

  def forcing_fn():
    return f

  return forcing_fn



def checkerboard_forcing_module(
    grid: grids.Grid,
    freq: int = 8,
    amp: float = 1.0,
    offset=(0, 0)
) -> types.Spectral2DForcingFn_vort:
    """
    Constructs a divergence-free velocity forcing function with a checkerboard pattern.
    :param grid: The grids.Grid object defining the simulation domain.
    :param Nm:   Number of modes for the checkerboard pattern.
    :param amp:  Amplitude scaling factor for the velocity.
    :param offset: Offset for the grid coordinates (default (0,0)).
    :return: A function that returns spectral vorticity forcing.
    """
    # Define spatial grid
    x, y = grid.axes(offset=offset)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
 
    # Compute wave numbers
    Lx, Ly = grid.domain[0][1], grid.domain[1][1]
    kx = (2.0 * jnp.pi * freq) / Lx
    ky = (2.0 * jnp.pi * freq) / Ly
    denom = (kx**2 + ky**2)  # To normalize the stream function
 
    """
    Computes the time independant curl of divergence-free velocity 
    forcing (fx, fy).
    """
    # Compute curl of corcing from the stream function
    curl_f_values = -amp * jnp.sin(kx * X)* jnp.sin(ky * Y)
    curl_f_hat = jnp.fft.rfft2(curl_f_values)

    def forcing_fn():
        return curl_f_hat
 
    return forcing_fn

def checkerboard_Q2D(
    grid: grids.Grid,
    freq: int = 8,
    amp: float = 1.0,
    offset=(0, 0)
) -> types.Spectral2DForcingFn:
    """
    Constructs a divergence-free velocity forcing function with a checkerboard pattern.
    :param grid: The grids.Grid object defining the simulation domain.
    :param Nm:   Number of modes for the checkerboard pattern.
    :param amp:  Amplitude scaling factor for the velocity.
    :param offset: Offset for the grid coordinates (default (0,0)).
    :return: A function that returns spectral vorticity forcing.
    """
    # Define spatial grid
    x, y = grid.axes(offset=offset)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
 
    # Compute wave numbers
    Lx, Ly = grid.domain[0][1], grid.domain[1][1]
    kx = (2.0 * jnp.pi * freq) / Lx
    ky = (2.0 * jnp.pi * freq) / Ly
    denom = (kx**2 + ky**2)  # To normalize the stream function
 
    """
    Computes the time independant curl of divergence-free velocity 
    forcing (fx, fy).
    """
    # Compute curl of corcing from the stream function
    f_u_values =   amp * jnp.sin(kx * X)* jnp.cos(ky * Y)
    f_v_values = - amp * jnp.cos(kx * X)* jnp.sin(ky * Y)
    f_u_values = jnp.fft.rfftn(f_u_values)
    f_v_values = jnp.fft.rfftn(f_v_values)


    def forcing_fn():
        return f_u_values, f_v_values
 
    return forcing_fn


def forcing_3D(grid: grids.Grid,
    freq: int = 1,
    amp: float = 1.0,
    offset=(0, 0, 0)
    ) -> types.Array:
    """
    Constructs a divergence-free velocity forcing function with a checkerboard pattern.
    :param grid: The grids.Grid object defining the simulation domain.
    :param Nm:   Number of modes for the checkerboard pattern.
    :param amp:  Amplitude scaling factor for the velocity.
    :param offset: Offset for the grid coordinates (default (0,0)).
    :return: A function that returns spectral vorticity forcing.
    """
    if grid.ndim != 3:
       KeyError("Grid is not 3 dimensional")

    x, y, z = grid.axes(offset=offset)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    Lx, Ly, Lz = grid.domain[0][1] - grid.domain[0][0],\
                 grid.domain[1][1] - grid.domain[1][0], \
                 grid.domain[2][1] - grid.domain[2][0]

    kx = (2.0 * jnp.pi * freq) / Lx
    ky = (2.0 * jnp.pi * freq) / Ly

    fx =   amp * jnp.sin(kx * X)* jnp.cos(ky * Y)
    fy = - amp * jnp.cos(kx * X)* jnp.sin(ky * Y)
    fz = jnp.zeros_like(fx)
    fx = jnp.fft.rfftn( fx )
    fy = jnp.fft.rfftn( fy )
    fz = jnp.fft.rfftn( fz )

    def forcing_fn():
        return jnp.stack( [fx, fy, fz], axis = -1)
 
    return forcing_fn

def kolmogorov_forcing_module(
    grid: grids.Grid,
    k: int = 2,
    amp: float = 1.0,
    offset: Tuple[float, float] = (0, 0)
) -> types.Spectral2DForcingFn_vort:
    """
    Constructs a Kolmogorov-type vorticity forcing function in spectral space.

    :param grid: The grids.Grid object defining the simulation domain.
    :param k: Wavenumber for the forcing (e.g. number of oscillations).
    :param amp: Amplitude scaling factor for the forcing.
    :param offset: Offset for grid coordinates (default (0, 0)).
    :return: A function that returns spectral vorticity forcing.
    """
    # Get meshgrid
    x, y = grid.axes(offset=offset)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Stream function ψ that generates Kolmogorov forcing via curl(ψ)
    #psi = amp * jnp.sin(k * y)

    # Compute vorticity forcing: ∇²ψ = d²ψ/dy² = -k² * sin(k * y)
    curl_f_values = -amp * k * jnp.sin(k * Y)

    # Compute 2D FFT of the scalar vorticity forcing
    curl_f_hat = jnp.fft.rfft2(curl_f_values)

    def forcing_fn():
        return curl_f_hat

    return forcing_fn
