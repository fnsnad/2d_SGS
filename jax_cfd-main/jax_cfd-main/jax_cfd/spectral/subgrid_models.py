
"""Pseudospectral equations for subrid models."""

import dataclasses
from typing import Callable, Optional, Tuple
from jax import debug, value_and_grad, lax, random
import jax.numpy as jnp
from jax_cfd.base import forcings
from jax_cfd.spectral import time_stepping
from jax_cfd.base import grids
from jax_cfd.spectral import utils as spectral_utils
from jax_cfd.spectral.equations import _get_grid_variable
from jax_cfd.spectral import types
import optax

@dataclasses.dataclass
class smagorinsky(time_stepping.ImplicitExplicitODE):
  """Breaks the Navier-Stokes equation into implicit and explicit parts.

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms.

  Attributes:
    viscosity: strength of the diffusion term
    grid: underlying grid of the process
    smooth: smooth the advection term using the 2/3-rule.
    forcing_fn: forcing function, if None then no forcing is used.
    C: smagorinsky constant.
  """

  viscosity: float
  grid: grids.Grid
  C: float = 0.2
  drag: float = 0.
  smooth: bool = True
  forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
  _forcing_fn_with_grid = None

  def __post_init__(self):
    self.kx, self.ky = self.grid.rfft_mesh()
    self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
    self.kx, self.ky = 2j * jnp.pi * self.kx, 2j * jnp.pi * self.ky
    self.laplace = (self.kx**2 + self.ky**2)
    self.inv_laplace = self.laplace.at[0, 0].set(1)
    self.inv_laplace = 1 / self.inv_laplace
    self.linear_term = self.viscosity * self.laplace - self.drag
    self.Delta = jnp.prod(jnp.array(self.grid.step)) # Cell volume
    # setup the forcing function with the caller-specified grid.
    if self.forcing_fn is not None:
      self._forcing_fn_with_grid = self.forcing_fn(self.grid)

  def explicit_terms(self, vorticity_hat):
    velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)
    vxhat, vyhat = velocity_solve(vorticity_hat)
    vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)

    grad_x_hat = self.kx * vorticity_hat
    grad_y_hat = self.ky * vorticity_hat
    grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

    advection = -(grad_x * vx + grad_y * vy)
    advection_hat = jnp.fft.rfftn(advection)

    # Smagorisnky viscosity
    S11, S21 = strain_field_trace(self,vorticity_hat)
    S11, S21 = jnp.fft.irfftn(S11), jnp.fft.irfftn(S21)
    S = jnp.sqrt(2*S11**2 + 2*S21**2)
    vis_sma = 2*self.C*self.Delta*S
    S11 =  jnp.fft.rfftn( vis_sma * S11 )
    S21 =  jnp.fft.rfftn( vis_sma * S21 )
    # Add stress tensor acceleration to advection term
    #debug.print(S11.shape, S21.shape, advection_hat.shape)
    advection_hat -= -2*self.kx *self.ky * S11 + (self.kx**2 - self.ky**2) * S21 

    if self.smooth is not None:
      advection_hat *= self.filter_

    terms = advection_hat
    

    if self.forcing_fn is not None:
      fx, fy = self._forcing_fn_with_grid((_get_grid_variable(vx, self.grid),
                                           _get_grid_variable(vy, self.grid)))
      fx_hat, fy_hat = jnp.fft.rfft2(fx.data), jnp.fft.rfft2(fy.data)
      terms += spectral_utils.spectral_curl_2d((self.kx, self.ky),
                                               (fx_hat, fy_hat))

    return terms

  def implicit_terms(self, vorticity_hat):
    return self.linear_term * vorticity_hat

  def implicit_solve(self, vorticity_hat, time_step):
    return 1 / (1 - time_step * self.linear_term) * vorticity_hat
  
  # pylint: disable=g-doc-args,g-doc-return-or-yield,invalid-name
def forced_smagorinsky(viscosity, grid, smooth):
  """ Set up SGS for Navier Stokes with kolgomorov forcing.
  """
  wave_number = 4
  offsets = ((0, 0), (0, 0))
  # pylint: disable=g-long-lambda
  forcing_fn = lambda grid: forcings.kolmogorov_forcing(
      grid, k=wave_number, offsets=offsets)
  return smagorinsky(
      viscosity,
      grid,
      C=0.2,
      smooth=smooth,
      forcing_fn=forcing_fn)


def strain_field_trace(self,vorticity_hat):
  """ Computes the sqrt(Sij*Sij) for the smagorinsky model.
      Used inside smargorsinky class equation
      Assumes incompresibility and 2D flow.
      S11 = - S22 and S12 = S21
      Sij = 1/2 (du_i/dx_j + du_j/dx_i)
  """
  S11 = self.kx*self.ky * vorticity_hat * self.inv_laplace / 2
  S12 = (self.ky**2 - self.kx**2) * self.inv_laplace * vorticity_hat / 2
  return S11,S12
  
@dataclasses.dataclass
class Similarity(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid: grids.Grid
    Delta: float
    drag: float = 0.0
    smooth: bool = True
    k_filter: float = None
    forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
    _forcing_fn_with_grid = None

    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        if self.k_filter is None:
            self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        else:
            self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid, self.k_filter)
        self.kx, self.ky = 2j * jnp.pi * self.kx, 2j * jnp.pi * self.ky
        self.laplace = (self.kx**2 + self.ky**2)
        self.inv_laplace = jnp.where(self.laplace == 0, 0.0, 1 / self.laplace)
        self.linear_term = self.viscosity * self.laplace - self.drag
        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)

    def explicit_terms(self, state):
        U_hat = state

        if self.smooth:
            U_hat = U_hat * self.filter_[..., None]

        U = jnp.fft.irfft2(U_hat, axes=(0, 1))

        dUdx = jnp.fft.irfft2(self.kx[..., None] * U_hat, axes=(0, 1))
        dUdy = jnp.fft.irfft2(self.ky[..., None] * U_hat, axes=(0, 1))

        advection = -(dUdx * U[..., 0:1] + dUdy * U[..., 1:2])
        terms_les = self.filter_[..., None] * jnp.fft.rfft2(advection, axes=(0, 1))

        U_bar = fourier_filter(U, self.Delta)
        tau_xx = fourier_filter(U[..., 0] * U[..., 0], self.Delta) - U_bar[..., 0] * U_bar[..., 0]
        tau_xy = fourier_filter(U[..., 0] * U[..., 1], self.Delta) - U_bar[..., 0] * U_bar[..., 1]
        tau_yy = fourier_filter(U[..., 1] * U[..., 1], self.Delta) - U_bar[..., 1] * U_bar[..., 1]

        tau_xx_hat = jnp.fft.rfft2(tau_xx, axes=(0, 1))
        tau_xy_hat = jnp.fft.rfft2(tau_xy, axes=(0, 1))
        tau_yy_hat = jnp.fft.rfft2(tau_yy, axes=(0, 1))

        div_tau = jnp.stack(
            [self.kx * tau_xx_hat + self.ky * tau_xy_hat,
             self.kx * tau_xy_hat + self.ky * tau_yy_hat],
            axis=-1
        )

        terms_les -= self.filter_[..., None] * div_tau

        div_rhs = (self.kx * terms_les[..., 0] + self.ky * terms_les[..., 1])
        leray = jnp.stack([self.kx, self.ky], axis=-1)
        leray *= div_rhs[..., None] * self.inv_laplace[..., None]
        terms_les -= leray

        if self.smooth:
            terms_les *= self.filter_[..., None]

        return terms_les

    def implicit_terms(self, state):
        U_les = state
        return self.linear_term[..., None] * U_les

    def implicit_solve(self, y_hat, dt):
        U_les = y_hat
        return U_les / (1 - dt * (self.linear_term[..., None]))

@dataclasses.dataclass
class NGM4(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid: grids.Grid
    Delta: float
    drag: float = 0.0
    smooth: bool = True
    k_filter: float = None
    forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
    _forcing_fn_with_grid = None

    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        if self.k_filter is None:
            self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        else:
            self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid, self.k_filter)
        self.kx, self.ky = 2j * jnp.pi * self.kx, 2j * jnp.pi * self.ky
        self.laplace = (self.kx**2 + self.ky**2)
        self.inv_laplace = jnp.where(self.laplace == 0, 0.0, 1 / self.laplace)
        self.linear_term = self.viscosity * self.laplace - self.drag
        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)

    def explicit_terms(self, state):
        U_hat = state

        if self.smooth:
            U_hat *= self.filter_[..., None]

        U = jnp.fft.irfft2(U_hat, axes=(0, 1))

        dUdx = jnp.fft.irfft2(self.kx[..., None] * U_hat, axes=(0, 1))
        dUdy = jnp.fft.irfft2(self.ky[..., None] * U_hat, axes=(0, 1))

        advection = -(dUdx * U[..., 0:1] + dUdy * U[..., 1:2])
        terms_les = self.filter_[..., None] * jnp.fft.rfft2(advection, axes=(0, 1))

        dUdxdx = jnp.fft.irfft2( self.kx[..., None]**2 * U_hat, axes=(0, 1))
        dUdydy = jnp.fft.irfft2( self.ky[..., None]**2 * U_hat, axes=(0, 1))
        dUdxdy = jnp.fft.irfft2( self.ky[..., None] * self.kx[..., None] * U_hat, axes=(0, 1))

        tau_xx = self.Delta**2 / 12 * (dUdx[..., 0]**2 + dUdy[..., 0]**2)
        tau_xy = self.Delta**2 / 12 * (dUdx[..., 0] * dUdx[..., 1] + dUdy[..., 0] * dUdy[..., 1])
        tau_yy = self.Delta**2 / 12 * (dUdx[..., 1]**2 + dUdy[..., 1]**2)

        tau_xx += self.Delta**4 / 288 * (dUdxdx[..., 0]**2 + dUdydy[..., 0]**2 + 2 * dUdxdy[..., 0]**2)
        tau_xy += self.Delta**4 / 288 * (dUdxdx[..., 0] * dUdxdx[..., 1] + 2 * dUdxdy[..., 0] * dUdxdy[..., 1] + dUdydy[..., 0] * dUdydy[..., 1])
        tau_yy += self.Delta**4 / 288 * (dUdxdx[..., 1]**2 + dUdydy[..., 1]**2 + 2 * dUdxdy[..., 1]**2)

        tau_xx_hat = jnp.fft.rfft2(tau_xx, axes=(0, 1))
        tau_xy_hat = jnp.fft.rfft2(tau_xy, axes=(0, 1))
        tau_yy_hat = jnp.fft.rfft2(tau_yy, axes=(0, 1))

        div_tau = jnp.stack(
            [self.kx * tau_xx_hat + self.ky * tau_xy_hat,
             self.kx * tau_xy_hat + self.ky * tau_yy_hat],
            axis=-1
        )

        terms_les -= self.filter_[..., None] * div_tau

        div_rhs = (self.kx * terms_les[..., 0] + self.ky * terms_les[..., 1])
        leray = jnp.stack([self.kx, self.ky], axis=-1)
        leray *= div_rhs[..., None] * self.inv_laplace[..., None]
        terms_les -= leray

        if self.smooth:
            terms_les *= self.filter_[..., None]

        return terms_les

    def implicit_terms(self, state):
        U_les = state
        return self.linear_term[..., None] * U_les

    def implicit_solve(self, y_hat, dt):
        U_les = y_hat
        return U_les / (1 - dt * (self.linear_term[..., None]))
      
@dataclasses.dataclass
class NavierStokes2D_LES(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
    diff_order: int = 1
    regularize: bool = True
    forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
    sgs_model_fn: Optional[Callable[[grids.Grid], types.Spectral2DForcingFn]] = None
    _forcing_fn_with_grid = None
 
    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        self.laplace = (jnp.pi * 2j)**2 * (self.kx**2 + self.ky**2)
        if self.k_filter is None:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        else:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid,self.k_filter)
        self.linear_term = - self.viscosity * (- self.laplace)**self.diff_order - self.drag
        self.w_to_v = spectral_utils.vorticity_to_velocity(self.grid)
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta*5e-4, self.kx, self.ky )
        self.Delta2 = self.Delta*1.2
        self.G = jnp.sqrt(jnp.exp(- ((self.kx*2*jnp.pi)**2 + (self.ky*2*jnp.pi)**2) * (self.Delta2**2) / 24.0) )

        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)
 
    def explicit_terms(self, vorticity_hat):
        # Solve velocity
        if self.smooth:
            #vorticity_hat = self.filter_fn(vorticity_hat)
            vorticity_hat *= self.filter_
        vx_hat, vy_hat = self.w_to_v(vorticity_hat)
        vx = jnp.fft.irfftn(vx_hat)
        vy = jnp.fft.irfftn(vy_hat)
        # Gradients
        two_i_pi = 2j * jnp.pi
        grad_x_hat = two_i_pi * self.kx * vorticity_hat
        grad_y_hat = two_i_pi * self.ky * vorticity_hat
        grad_x = jnp.fft.irfftn(grad_x_hat)
        grad_y = jnp.fft.irfftn(grad_y_hat)
 
        advection = -(grad_x * vx + grad_y * vy)
        advection_hat = jnp.fft.rfftn(advection)
 
        terms = advection_hat

        if self.regularize:
            sigma2 = self.Delta**2/12
            tau_xx_6th, tau_xy_6th, tau_yy_6th = tau_third_order( vx_hat, vy_hat, self.kx ,self.ky, sigma2**3/4)
            trR = tau_xx_6th**2 + 2*tau_xy_6th**2 + tau_yy_6th**2
            nabla6 = jnp.fft.irfft2( self.laplace**3 * vorticity_hat )
            diss =  trR**(1/4) * sigma2**4/2/self.Delta**3 * nabla6
            terms += jnp.fft.rfft2(diss)
            #debug.print("nuR: {}", jnp.mean(trR**(1/4) * sigma2**4/2/self.Delta**3))

        if self.sgs_model_fn is not None:
            tau_xx, tau_xy, tau_yy = self.sgs_model_fn(vx_hat, vy_hat, self.kx, self.ky, self.Delta)
            div_tau_x, div_tau_y = spectral_utils.divergence_of_sgs_stress_fft(tau_xx, tau_xy, tau_yy, self.kx, self.ky)
            div_tau_x_hat = jnp.fft.rfft2(div_tau_x)
            div_tau_y_hat = jnp.fft.rfft2(div_tau_y)
            sgs_curl_hat = spectral_utils.spectral_curl_2d((self.kx, self.ky), (div_tau_x_hat, div_tau_y_hat))
 
            terms -= sgs_curl_hat#*self.G

        #terms *= self.G

        if self.smooth:
            #terms = self.filter_fn(terms)
            terms *= self.filter_
 
        if self.forcing_fn is not None:
            fx_hat, fy_hat = self._forcing_fn_with_grid()
            terms += spectral_utils.spectral_curl_2d((self.kx, self.ky), (fx_hat, fy_hat))
 
        return terms
 
    def implicit_terms(self, vorticity_hat):
        return self.linear_term * vorticity_hat
 
    def implicit_solve(self, vorticity_hat, time_step):
        return 1 / (1 - time_step * self.linear_term) * vorticity_hat#self.filter_fn(  )

def NGM_4thOrder(u, v, kx, ky, dx, C2=1/12, C4=1/288):
    (du_dx, du_dy, dv_dx, dv_dy,d2u_dx2, d2u_dy2, 
     d2u_dxdy,d2v_dx2, d2v_dy2, d2v_dxdy) = spectral_utils.velocity_derivatives_fft(u, v, kx, ky)
    # 2nd order NGM
    tau_xx_2nd = C2 * dx**2 * (du_dx**2 + du_dy**2)
    tau_xy_2nd = C2 * dx**2 * (du_dx * dv_dx + du_dy * dv_dy)
    tau_yy_2nd = C2 * dx**2 * (dv_dx**2 + dv_dy**2)

    # 4th order NGM
    tau_xx_4th = C4 * dx**4 * (d2u_dx2**2 + d2u_dy2**2 + 2 * d2u_dxdy**2)
    tau_yy_4th = C4 * dx**4 * (d2v_dx2**2 + d2v_dy2**2 + 2 * d2v_dxdy**2)
    tau_xy_4th = C4 * dx**4 * (d2u_dx2 * d2v_dx2 + 2 * d2u_dxdy * d2v_dxdy + d2u_dy2 * d2v_dy2)

    tau_xx = tau_xx_2nd + tau_xx_4th
    tau_xy = tau_xy_2nd + tau_xy_4th
    tau_yy = tau_yy_2nd + tau_yy_4th

    return tau_xx, tau_xy, tau_yy

def tau_third_order( u, v, kx ,ky, e3):
    two_pi_i = 2j * jnp.pi
    Dx = two_pi_i * kx
    Dy = two_pi_i * ky

    dud3x   = jnp.fft.irfft2( Dx**3 * u )
    dud2xdy = jnp.fft.irfft2( Dx**2 * Dy * u )
    dudxd2y = jnp.fft.irfft2( Dx * Dy**2 * u )
    dud3y   = jnp.fft.irfft2( Dy**3 * u )

    dvd3x   = jnp.fft.irfft2( Dx**3 * v )
    dvd2xdy = jnp.fft.irfft2( Dx**2 * Dy * v )
    dvdxd2y = jnp.fft.irfft2( Dx * Dy**2 * v )
    dvd3y   = jnp.fft.irfft2( Dy**3 * v )

    tau_xx = dud3x**2 + dud3y**2 + 3*dud2xdy**2 + 3*dudxd2y
    tau_yy = dvd3x**2 + dvd3y**2 + 3*dvd2xdy**2 + 3*dvdxd2y
    tau_xy = dud3x*dvd3x + dud3y*dvd3y + 3*dud2xdy*dvd2xdy + 3*dudxd2y*dvdxd2y

    return e3*tau_xx , e3*tau_xy, e3*tau_yy



def NGM_6thOrder(u, v, kx, ky, dx, C2=1/12, C4=1/288, C6=1/6912):
    (du_dx, du_dy, dv_dx, dv_dy,d2u_dx2, d2u_dy2, 
     d2u_dxdy,d2v_dx2, d2v_dy2, d2v_dxdy) = spectral_utils.velocity_derivatives_fft(u, v, kx, ky)
    # 2nd order NGM
    tau_xx_2nd = C2 * dx**2 * (du_dx**2 + du_dy**2)
    tau_xy_2nd = C2 * dx**2 * (du_dx * dv_dx + du_dy * dv_dy)
    tau_yy_2nd = C2 * dx**2 * (dv_dx**2 + dv_dy**2)

    # 4th order NGM
    tau_xx_4th = C4 * dx**4 * (d2u_dx2**2 + d2u_dy2**2 + 2 * d2u_dxdy**2)
    tau_yy_4th = C4 * dx**4 * (d2v_dx2**2 + d2v_dy2**2 + 2 * d2v_dxdy**2)
    tau_xy_4th = C4 * dx**4 * (d2u_dx2 * d2v_dx2 + 2 * d2u_dxdy * d2v_dxdy + d2u_dy2 * d2v_dy2)

    e3 = C6 * dx**6

    tau_xx_6th, tau_xy_6th, tau_yy_6th = tau_third_order( u, v, kx ,ky, e3)


    tau_xx = tau_xx_2nd + tau_xx_4th + tau_xx_6th
    tau_xy = tau_xy_2nd + tau_xy_4th + tau_xy_6th
    tau_yy = tau_yy_2nd + tau_yy_4th + tau_yy_6th

    return tau_xx, tau_xy, tau_yy

def NGMF_6thOrder(u, v, kx, ky, dx, C2=1/12, C4=1/288, C6=1/6912):
    # Get derivatives
    (du_dx, du_dy, dv_dx, dv_dy,d2u_dx2, d2u_dy2, 
     d2u_dxdy,d2v_dx2, d2v_dy2, d2v_dxdy) = spectral_utils.velocity_derivatives_fft(u, v, kx, ky)
    
    # Get filter functions
    filter_fn = spectral_utils.get_filter_fn( dx, kx, ky )

    # 2nd order NGM
    tau_xx_2nd = C2 * dx**2 * (du_dx**2 + du_dy**2)
    tau_xy_2nd = C2 * dx**2 * (du_dx * dv_dx + du_dy * dv_dy)
    tau_yy_2nd = C2 * dx**2 * (dv_dx**2 + dv_dy**2)

    # 4th order NGM
    tau_xx_4th = C4 * dx**4 * (d2u_dx2**2 + d2u_dy2**2 + 2 * d2u_dxdy**2)
    tau_yy_4th = C4 * dx**4 * (d2v_dx2**2 + d2v_dy2**2 + 2 * d2v_dxdy**2)
    tau_xy_4th = C4 * dx**4 * (d2u_dx2 * d2v_dx2 + 2 * d2u_dxdy * d2v_dxdy + d2u_dy2 * d2v_dy2)

    # Compute R
    two_pi_i = 2j * jnp.pi
    Dx = two_pi_i * kx
    Dy = two_pi_i * ky
    L = Dx**2 + Dy**2
    dLudx = jnp.fft.irfft2( Dx* L * u )
    dLudy = jnp.fft.irfft2( Dy* L * u )
    dLvdx = jnp.fft.irfft2( Dx* L * u )
    dLvdy = jnp.fft.irfft2( Dy* L * u )
    # dLudx + dLudy = 0

    tau_xx_R = C6 * dx**6 * ( dLudx**2 + dLudy**2 )
    tau_yy_R = C6 * dx**6 * ( dLvdx**2 + dLvdy**2 )
    tau_xy_R = C6 * dx**6 * ( dLudx*dLvdx + dLudy*dLvdy )

    tau_xx_R = jnp.fft.irfft2( filter_fn( jnp.fft.rfft2(tau_xx_R) ) )
    tau_yy_R = jnp.fft.irfft2( filter_fn( jnp.fft.rfft2(tau_yy_R) ) )
    tau_xy_R = jnp.fft.irfft2( filter_fn( jnp.fft.rfft2(tau_xy_R) ) )

    tau_xx = tau_xx_2nd + tau_xx_4th + tau_xx_R
    tau_xy = tau_xy_2nd + tau_xy_4th + tau_yy_R
    tau_yy = tau_yy_2nd + tau_yy_4th + tau_xy_R

    return tau_xx, tau_xy, tau_yy

@dataclasses.dataclass
class NavierStokes2D_implicit_LES(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    regularize: bool = False
    forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
    sgs_model_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float], Tuple]] = None
    _forcing_fn_with_grid = None
    _sgs_curl_fn = None
    
 
    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        self.laplace = (jnp.pi * 2j)**2 * (self.kx**2 + self.ky**2)
        self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        self.linear_term = self.viscosity * self.laplace - self.drag
        self.w_to_v = spectral_utils.vorticity_to_velocity(self.grid)
        self.optimizer = optax.lbfgs()

        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)

        if self.sgs_model_fn is not None:
          def sgs_curl_hat_fn(w_hat):
            u, v = self.w_to_v(w_hat)
            tau_xx, tau_xy, tau_yy = self.sgs_model_fn(u, v, 
                                                       self.kx, self.ky, self.Delta)
            div_tau_x, div_tau_y = spectral_utils.divergence_of_sgs_stress_fft(
               tau_xx, tau_xy, tau_yy, self.kx, self.ky)
            div_tau_x_hat = jnp.fft.rfft2(div_tau_x)
            div_tau_y_hat = jnp.fft.rfft2(div_tau_y)
            sgs_curl_hat = spectral_utils.spectral_curl_2d((self.kx, self.ky), (div_tau_x_hat, div_tau_y_hat))
            return sgs_curl_hat
          self._sgs_curl_fn = sgs_curl_hat_fn
 
    def explicit_terms(self, vorticity_hat):
        # Solve velocity
        vx_hat, vy_hat = self.w_to_v(vorticity_hat)
        vx = jnp.fft.irfftn(vx_hat)
        vy = jnp.fft.irfftn(vy_hat)
        # Gradients
        two_i_pi = 2j * jnp.pi
        grad_x_hat = two_i_pi * self.kx * vorticity_hat
        grad_y_hat = two_i_pi * self.ky * vorticity_hat
        grad_x = jnp.fft.irfftn(grad_x_hat)
        grad_y = jnp.fft.irfftn(grad_y_hat)
 
        advection = -(grad_x * vx + grad_y * vy)
        advection_hat = jnp.fft.rfftn(advection)
 
        terms = advection_hat

        if self.regularize:
            sigma = self.Delta**2/12
            tau_xx_6th, tau_xy_6th, tau_yy_6th = tau_third_order( vx_hat, vy_hat, self.kx ,self.ky, sigma**3/4)
            trR = tau_xx_6th**2 + 2*tau_xy_6th**2 + tau_yy_6th**2
            nabla6 = jnp.fft.irfft2( self.laplace**3 * vorticity_hat )
            diss = trR**(1/4) * sigma**8/2/self.Delta**3 * nabla6
            terms += jnp.fft.rfft2(diss)

        if self.smooth:
            terms *= self.filter_
 
        if self.forcing_fn is not None:
            fx, fy = self._forcing_fn_with_grid((_get_grid_variable(vx, self.grid),
                                                 _get_grid_variable(vy, self.grid)))
            fx_hat, fy_hat = jnp.fft.rfft2(fx.data), jnp.fft.rfft2(fy.data)
            terms += spectral_utils.spectral_curl_2d((self.kx, self.ky), (fx_hat, fy_hat))

        

        return terms
 
    def implicit_terms(self, vorticity_hat):
        if self.sgs_model_fn is not None:
          diss = self.linear_term * vorticity_hat
          sgs_curl_hat = self._sgs_curl_fn(vorticity_hat)
          if self.smooth:
            sgs_curl_hat *= self.filter_
          return diss + sgs_curl_hat
        else:
          return self.linear_term * vorticity_hat
 
    def implicit_solve(self, y_hat, dt):
        """
        Solves 1 - dt G x = y, for x
        """
        def loss_fn(x):
           x_hat = jnp.fft.rfft2(x)
           residual = jnp.fft.irfft2(x_hat - dt*self.implicit_terms(x_hat) - y_hat)
           return jnp.sum( residual**2 )

        value_and_grad_fn = value_and_grad(loss_fn)

        # Setup optimizer
        x = jnp.fft.irfft2(y_hat)
        opt_state = self.optimizer.init(x)

        # One optimization step (pure function)
        def step_fn(state):
            x, opt_state, step, _ = state
            value, grad = value_and_grad_fn(x)
            updates, opt_state = self.optimizer.update(
               grad, opt_state, x, value = value, grad = grad, value_fn = loss_fn)
            x = optax.apply_updates(x, updates)
            return (x, opt_state, step + 1, value)
        def cond_fn(state):
          _, _, step, loss = state
          return jnp.logical_and(step < 20, loss > 1e-28)
        # Run fixed number of optimization steps with `lax.scan`
        init_loss = loss_fn(x)
        state = (x, opt_state, 0, init_loss)
        final_x, _, final_step, final_loss = lax.while_loop(cond_fn, step_fn, state)

        # Return the solution in Fourier space
        return jnp.fft.rfft2(final_x)*self.filter_


@dataclasses.dataclass
class DNS_w_LES(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
 
    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        self.two_i_pi = 2j * jnp.pi
        self.laplace = self.two_i_pi**2 * (self.kx**2 + self.ky**2)
        if self.k_filter is None:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        else:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid,self.k_filter)

        self.filter2_ = spectral_utils.brick_wall_filter_2d(self.grid,1/16)#1/2
        self.linear_term = self.viscosity * self.laplace - self.drag
        self.linear_term_les = self.linear_term 
        #(self.viscosity * self.laplace - self.drag + (1/256)**6*self.laplace**3) * self.filter2_
        self.w_to_v = spectral_utils.vorticity_to_velocity(self.grid)
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta, self.kx, self.ky )
        self.Delta2 = self.Delta/4
        self.G = jnp.exp(- ((self.kx*2*jnp.pi)**2 + (self.ky*2*jnp.pi)**2)**3 * (self.Delta2**2)**3 / 24.0)
 
    def explicit_terms(self, state):
        # Unpack the state
        w_dns, w_les = state # already in fourier space

        if self.smooth:
            #w_les = self.filter_fn(w_les)
            w_les *= self.filter2_
        # Solve velocity
        uhat_dns, vhat_dns = self.w_to_v(w_dns)
        uhat_les, vhat_les = self.w_to_v(w_les)
        # Compute advection LES
        u = jnp.fft.irfftn(uhat_les)
        v = jnp.fft.irfftn(vhat_les)
        w = jnp.fft.irfftn(w_les)
        #dwdx_hat = jnp.fft.irfftn( self.two_i_pi * self.kx * w_les )
        #dwdy_hat = jnp.fft.irfftn( self.two_i_pi * self.ky * w_les )
        #advection = -(dwdx_hat * u + dwdy_hat * v)
        #terms_les = jnp.fft.rfftn(advection)
        # Compute in conservative form
        dx_uw = self.two_i_pi * self.kx * jnp.fft.rfftn( u*w, axes = (0,1) )
        dy_vw = self.two_i_pi * self.ky * jnp.fft.rfftn( v*w, axes = (0,1) )

        terms_les = - ( dx_uw + dy_vw )

        # Compute advection DNS
        u = jnp.fft.irfftn(uhat_dns)
        v = jnp.fft.irfftn(vhat_dns)
        dwdx_hat = jnp.fft.irfftn( self.two_i_pi * self.kx * w_dns )
        dwdy_hat = jnp.fft.irfftn( self.two_i_pi * self.ky * w_dns )
        advection = -(dwdx_hat * u + dwdy_hat * v)
        terms_dns = jnp.fft.rfftn(advection)

        # FIXME: compute the tau from dns and downsample it
        uu_bar = jnp.fft.rfftn( jnp.fft.irfftn( self.filter_fn(uhat_dns) )**2 )
        vv_bar = jnp.fft.rfftn( jnp.fft.irfftn( self.filter_fn(vhat_dns) )**2 )
        uv_bar = jnp.fft.rfftn( jnp.fft.irfftn( self.filter_fn(uhat_dns) )*jnp.fft.irfftn( self.filter_fn(vhat_dns) ) )
        
        u = jnp.fft.irfftn(uhat_dns * self.filter_ )
        v = jnp.fft.irfftn(vhat_dns * self.filter_ )
        tau_xx = self.filter_fn( jnp.fft.rfftn(u*u) ) - uu_bar
        tau_xy = self.filter_fn( jnp.fft.rfftn(u*v) ) - uv_bar
        tau_yy = self.filter_fn( jnp.fft.rfftn(v*v) ) - vv_bar

        div_curl_tau = ( (self.two_i_pi*self.kx)**2 - (self.two_i_pi*self.ky)**2 ) * tau_xy \
                        + self.two_i_pi**2*self.kx*self.ky*(tau_yy - tau_xx)

        terms_les -= div_curl_tau#*self.G #+ (1/256)**6*self.laplace**3
        
        #self.linear_term_les = self.linear_term \
        #  + jnp.fft.rfftn(jnp.fft.irfftn(div_curl_tau)**2)*(2*jnp.pi/self.Delta*4)**2 * self.laplace*self.filter2_ 


        if self.smooth:
            terms_dns *= self.filter_
            terms_les *= self.filter2_
 
        return terms_dns, terms_les
 
    def implicit_terms(self, state):
        # Unpack the state
        w_dns, w_les = state
        return self.linear_term * w_dns , self.linear_term_les * w_les
 
    def implicit_solve(self, y_hat, dt):
        """
        Solves 1 - dt G x = y, for x
        """
        w_dns, w_les = y_hat
        return 1 / (1 - dt * self.linear_term) * w_dns, 1 / (1 - dt * self.linear_term_les) * w_les
    
    def compute_flux_fn(self):
        
        Nx, Ny = self.grid.shape
        R = jnp.sqrt(self.kx**2 + self.ky**2)*2*jnp.pi
        # Create the filter kernel
        G_fn = lambda Delta: jnp.exp(- (R**2) * (Delta**2) / 24.0)

        # Helper function to compute advection
        def adv_fn( w ):
            u_hat, v_hat = self.w_to_v(w)
            # Compute advection LES
            #u = jnp.fft.irfft2( u_hat, s = (Nx,Ny) )
            #v = jnp.fft.irfft2( v_hat, s = (Nx,Ny) )
            #dwdx_hat = jnp.fft.irfft2( self.two_i_pi * self.kx * w, s = (Nx,Ny) )
            # dwdy_hat = jnp.fft.irfft2( self.two_i_pi * self.ky * w, s = (Nx,Ny) )
            # advection = ( dwdx_hat * u + dwdy_hat * v )
            # Filter
            #rhs_adv = jnp.fft.rfft2(advection, s = (Nx,Ny) )

            u = jnp.fft.irfftn(u_hat)
            v = jnp.fft.irfftn(v_hat)
            w = jnp.fft.irfftn(w)
            #dwdx_hat = jnp.fft.irfftn( self.two_i_pi * self.kx * w_les )
            #dwdy_hat = jnp.fft.irfftn( self.two_i_pi * self.ky * w_les )
            #advection = -(dwdx_hat * u + dwdy_hat * v)
            #terms_les = jnp.fft.rfftn(advection)
            # Compute in conservative form
            dx_uw = self.two_i_pi * self.kx * jnp.fft.rfftn( u*w, axes = (0,1) )
            dy_vw = self.two_i_pi * self.ky * jnp.fft.rfftn( v*w, axes = (0,1) )

            rhs_adv = ( dx_uw + dy_vw )

            return rhs_adv

        def get_tau_dns( w_dns ):
            # Compute tau from DNS state
            u_hat, v_hat = self.w_to_v( w_dns )
            u = jnp.fft.irfftn(u_hat )
            v = jnp.fft.irfftn(v_hat )
            u_hat = self.filter_fn(u_hat)
            v_hat = self.filter_fn(v_hat)
            uu_bar = jnp.fft.rfftn( jnp.fft.irfftn( u_hat )**2 )
            vv_bar = jnp.fft.rfftn( jnp.fft.irfftn( v_hat )**2 )
            uv_bar = jnp.fft.rfftn( jnp.fft.irfftn( u_hat )*jnp.fft.irfftn( v_hat ) )
            
            tau_xx = self.filter_fn( jnp.fft.rfftn(u*u) ) - uu_bar
            tau_xy = self.filter_fn( jnp.fft.rfftn(u*v) ) - uv_bar
            tau_yy = self.filter_fn( jnp.fft.rfftn(v*v) ) - vv_bar

            div_curl_tau = ( (self.two_i_pi*self.kx)**2 - (self.two_i_pi*self.ky)**2 ) * tau_xy \
                            + self.two_i_pi**2*self.kx*self.ky*(tau_yy - tau_xx)
            return div_curl_tau * self.filter_

        def flux_fn_(state, Delta):
            
            # Unpack state, should be in spectral state
            w_dns, w_les = state 
            if self.smooth:
                w_dns *= self.filter_
                w_les *= self.filter2_
            # Project LES filter to DNS
            w_dns_filtered = self.filter_fn(w_dns)
            G = G_fn(Delta)

            # Use large grid to compute tau
            div_curl_tau = get_tau_dns( w_dns )

            # Calculate flux in LES
            rhs_tau_les = div_curl_tau * self.filter2_ * G
            rhs_tau_les = jnp.fft.irfft2( rhs_tau_les , s = (Nx,Ny) )
            
            # Compute advection LES
            rhs_adv_les = adv_fn( w_les )
            rhs_adv_f_les = adv_fn( G * w_les )
            # Filter
            rhs_adv_les = jnp.fft.irfft2( rhs_adv_les * self.filter2_ * G , s = (Nx,Ny) )
            rhs_adv_f_les = jnp.fft.irfft2( rhs_adv_f_les * self.filter2_ , s = (Nx,Ny) )

            ω_les = jnp.fft.irfft2( G * w_les, s = (Nx,Ny) )
            flux_adv = - ω_les * ( rhs_adv_les - rhs_adv_f_les )
            flux_tau = - ω_les * rhs_tau_les

            # Calculate flux in DNS
            ω_dns = jnp.fft.irfft2( G * w_dns_filtered, s = (Nx,Ny) )
            rhs_adv_filtered_dns = adv_fn( w_dns_filtered * G )
            rhs_adv_filtered_dns = jnp.fft.irfft2( rhs_adv_filtered_dns * self.filter2_ , s = (Nx,Ny) )

            double_fl_adv = adv_fn( w_dns_filtered ) * self.filter2_ * G
            double_fl_adv = jnp.fft.irfft2( double_fl_adv, s = (Nx,Ny) )
            flux_dns = - ω_dns * ( double_fl_adv - rhs_adv_filtered_dns )

            rhs_adv_dns = adv_fn( w_dns_filtered )
            rhs_adv_dns = jnp.fft.irfft2( rhs_adv_dns * self.filter2_ * G , s = (Nx,Ny) )

            flux_filt_adv = - ω_dns * ( rhs_adv_dns - rhs_adv_filtered_dns)

            

            if self.smooth:
               div_curl_tau *= self.filter2_ * G

            div_curl_tau = jnp.fft.irfft2( div_curl_tau, s = (Nx,Ny) )
            #ω = jnp.fft.irfft2( G * w_dns_filtered, s = (Nx,Ny) )
            flux_dns_tau = - ω_dns * div_curl_tau

            spatial_profile = flux_adv, flux_tau, flux_dns, flux_filt_adv, flux_dns_tau
            
            tot_flux_adv = jnp.mean(flux_adv)
            tot_flux_tau = jnp.mean(flux_tau)
            tot_flux_dns = jnp.mean(flux_dns)
            tot_flux_filt_adv = jnp.mean(flux_filt_adv)
            tot_flux_dns_tau = jnp.mean(flux_dns_tau)

            return tot_flux_adv, tot_flux_tau, tot_flux_dns, tot_flux_filt_adv, tot_flux_dns_tau, spatial_profile

        return flux_fn_

@dataclasses.dataclass
class LES_w_R(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        self.two_i_pi = 2j * jnp.pi
        self.laplace = self.two_i_pi**2 * (self.kx**2 + self.ky**2)
        self.inv_laplace = jnp.where(self.laplace == 0, 1, 1 / self.laplace)
 
        if self.k_filter is None:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        else:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid,self.k_filter)
 
        self.linear_term = self.viscosity * self.laplace - self.drag
        self.w_to_v = spectral_utils.vorticity_to_velocity(self.grid)
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta, self.kx, self.ky )
        self.U_size = self.grid.shape + (2,)
        self.key = random.key(42)

    def explicit_terms(self, state):
        # Unpack the state
        U_les, R_ij = state # already in fourier space
        U_space = jnp.fft.irfft2( U_les, axes = (0,1) )
        R_space = jnp.fft.irfft2( R_ij, axes = (0,1))
 
        if self.smooth:
            #w_les = self.filter_fn(w_les)
            U_les *= self.filter_[...,None]
            R_ij *= self.filter_[...,None]
            #U_les = self.filter_fn(U_les)
        # Gradient U
        dUdx = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None] * U_les, axes = (0,1) )
        dUdy = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None] * U_les, axes = (0,1) )
        # Compute advection LES
        advection = -( dUdx * U_space[...,0:1] + dUdy * U_space[...,1:2] )
        terms_les = jnp.fft.rfft2( advection, axes = (0,1) )
 
        # Compute tau ngm
        dUdxdx = jnp.fft.irfft2( self.two_i_pi**2 * self.kx[...,None]**2 * U_les, axes = (0,1) )
        dUdydy = jnp.fft.irfft2( self.two_i_pi**2 * self.ky[...,None]**2 * U_les, axes = (0,1) )
        dUdxdy = jnp.fft.irfft2( self.two_i_pi**2 * self.ky[...,None]*self.kx[...,None] * U_les, axes = (0,1) )
        # Array ngmR as tau_xx, tau_yy, tau_xy
        ngm2xx = dUdx[...,0]**2 + dUdy[...,0]**2
        ngm2yy = dUdx[...,1]**2 + dUdy[...,1]**2
        ngm2xy = dUdx[...,0]*dUdx[...,1] + dUdy[...,0]*dUdy[...,1]             
        ngmR = self.Delta**2/12 * jnp.stack([ngm2xx, ngm2yy, ngm2xy], axis=-1)
 
        
        # Calculate ngm4
        ngm4xx = (dUdxdx[...,0]**2 + dUdydy[...,0]**2 + 2 * dUdxdy[...,0]**2) 
        ngm4yy = (dUdxdx[...,1]**2 + dUdydy[...,1]**2 + 2 * dUdxdy[...,1]**2)
        ngm4xy = (dUdxdx[...,0] * dUdxdx[...,1] + 2 * dUdxdy[...,0] * dUdxdy[...,1] + dUdydy[...,0] * dUdydy[...,1])
        ngm4 = self.Delta**4/288 * jnp.stack([ngm4xx, ngm4yy, ngm4xy], axis=-1)
        ngmR += ngm4

        ngmR = jnp.fft.rfft2( ngmR, axes = (0,1) ) + R_ij
 
        div_tau = jnp.stack( [ self.two_i_pi * self.kx * ngmR[...,0] + self.two_i_pi * self.ky * ngmR[...,2],\
                               self.two_i_pi * self.kx * ngmR[...,2] + self.two_i_pi * self.ky * ngmR[...,1]],\
                               axis = -1)
        terms_les -= div_tau
        #terms_les = self.filter_fn(terms_les)R_ij
        # Compute R evolution equation
 
        #I_1 = jnp.sqrt( 2*dUdx[...,0]**2 + ( dUdx[...,1] + dUdy[...,0] )**2/2 )
        #dxdM = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None], axes = (0,1) )
        #dydM = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None], axes = (0,1) )
        # Advection of R
        dRdx_ij = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None] * R_ij, axes = (0,1) )
        dRdy_ij = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None] * R_ij, axes = (0,1) )
        R_adv = (U_space[...,0:1] * dRdx_ij + U_space[...,1:2] * dRdy_ij)
        rhs_R = - R_adv
        # Production terms of R
        prod_R = jnp.stack([2*dUdx[...,0] * R_space[...,0] + 2*dUdy[...,0] * R_space[...,2],
                            2*dUdx[...,1] * R_space[...,2] + 2*dUdy[...,1] * R_space[...,1],
                              dUdx[...,1] * R_space[...,0] +   dUdy[...,0] * R_space[...,1]
                              ], axis=-1)
        #rhs_R += prod_R
        #rhs_R += jnp.fft.irfft2(self.filter_fn( jnp.fft.rfft2( prod_R, axes = (0,1) ) ), axes = (0,1) )
        
        rhs_R += -  prod_R \
            + 2 * jnp.fft.irfft2(self.filter_fn( jnp.fft.rfft2( prod_R, axes = (0,1) ) ), axes = (0,1) )


        # Filter production?
        rhs_R = jnp.fft.irfft2(self.filter_fn(jnp.fft.rfft2(rhs_R,axes=(0,1))),axes=(0,1))

        dLUdx = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None] * self.laplace[...,None] * U_les, axes = (0,1) )
        dLUdy = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None] * self.laplace[...,None] * U_les, axes = (0,1) )
 
        # Compute invariants
        I1 = 4*dUdx[...,0]**2 + (dUdx[...,1] + dUdy[...,0])**2
        I2 = (dUdy[...,0] - dUdx[...,1])**2
 
        # Add f forcing
        F_space = jnp.stack([(dLUdx[...,0]**2 + dLUdy[...,0]**2),
                         (dLUdx[...,1]**2 + dLUdy[...,1]**2),
                         (dLUdx[...,0]*dLUdx[...,1] + dLUdy[...,0]*dLUdy[...,1])], axis=-1)
 
        F_space *= self.Delta**6/6912
        F_scale = F_space.mean()

 
        #self.key, subkey = random.split(self.key)
        #R_rand = random.uniform(subkey,shape=R_space.shape )
        #ngmR += jnp.fft.rfft2( 1e-1*F_scale*R_rand, axes = (0,1) )

        # if self.smooth:
        # I1 = jnp.fft.irfft2( self.filter_ * jnp.fft.rfft2( I1 ) )
        #F_ij
        I = jnp.sqrt(I1[...,None] + I2[...,None])
        F_space = jnp.fft.irfft2( self.filter_fn( jnp.fft.rfft2( F_space, axes = (0,1) ) ), axes = (0,1) )
        linear_term =  I * ( F_space - R_space ) - 0.1 *I * R_space
        rhs_R += linear_term# * (- jnp.sign(linear_term))
        rhs_R = jnp.fft.rfft2( rhs_R, axes = (0,1) )
        rhs_R = self.filter_fn(rhs_R)
        order = 3
        # LU_space = jnp.fft.irfft2( self.laplace[...,None]**order*U_les, axes = (0,1) )   
        nu_eddy = (R_space[...,0]**2+R_space[...,1]**2+2*R_space[...,2]**2)**(1/4)
        
        # nu_eddy = 10*self.Delta**6/6912*jnp.stack([(dLUdx[...,0]**2 + dLUdy[...,0]**2),
        #                  (dLUdy[...,1]**2 + dLUdy[...,1]**2),
        #                  (dLUdx[...,0]*dLUdx[...,1] + dLUdy[...,0]*dLUdy[...,1])], axis=-1)
        #nu_eddy.mean() * self.Delta**7/(2*12**5)
        #nu_eddy = (F_space[...,0]**2+F_space[...,1]**2+2*F_space[...,2]**2)**(1/4)
        self.nu_R = I1.mean() * self.Delta**6/12**3#0*I1.mean() * self.Delta**2/4#/12**5*12**2*4
        nu_eddy *= self.Delta**5/(2*12**4)
        self.nu_eddy = nu_eddy.mean()
        #terms_les += jnp.fft.rfft2(nu_eddy.mean() * LU_space, axes = (0,1) )   

        # Leray projection
        div_rhs = self.two_i_pi * ( self.kx * terms_les[...,0] + self.ky * terms_les[...,1] )
        leray = jnp.stack( [self.two_i_pi * self.kx, self.two_i_pi * self.ky], axis = -1 ) 
        leray *= div_rhs[...,None] * self.inv_laplace[...,None]
        terms_les -= leray

        # Add hyperdissipation to R
        # LR_space = jnp.fft.irfft2( self.laplace[...,None]**3*R_ij, axes = (0,1) )   
        # LR_space = jnp.fft.rfft2( nu_eddy.mean() * LR_space, axes = (0,1) )
        # rhs_R += LR_space

 
        if self.smooth:
            terms_les *= self.filter_[...,None]
            rhs_R *= self.filter_[...,None]

            
        return terms_les, rhs_R
    
    def implicit_terms(self, state):

        # Calculate dissipation and hyperdissipation
        U_les, R_ij = state 
        #R_space = jnp.fft.irfft2( R_ij, axes = (0,1))
        order = 3
        order_R = 3
        return (self.linear_term[...,None] + self.nu_eddy*self.laplace[...,None]**order) * U_les ,\
              ( self.linear_term[...,None] + self.nu_R*self.laplace[...,None]**order_R ) * R_ij
    #1e3*self.nu_eddy*self.laplace[...,None]**order
    def implicit_solve(self, y_hat, dt):


        U_les, R_ij = y_hat 
        order = 3
        order_R = 3
        # + nu_eddy*self.laplace[...,None]**order
        return 1 / (1 - dt * (self.linear_term[...,None] + self.nu_eddy*self.laplace[...,None]**order)) * U_les, \
          1 / (1 - dt * (self.linear_term[...,None] + self.nu_R*self.laplace[...,None]**order_R ) ) * R_ij  
    

    def init_R( self, u_hat, grid_aux: grids.Grid):
        """
        Calculates R from u_les
        """
        kx ,ky = grid_aux.rfft_mesh()
        o_filt_fn = spectral_utils.get_filter_fn( self.Delta, kx, ky )
        # Filter field
        up = jnp.fft.irfft2( u_hat - o_filt_fn( u_hat ), axes = (0,1) )
        upup_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]**2         , axes =(0,1) ) )
        vpvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,1]**2         , axes =(0,1) ) )
        upvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]*up[...,1], axes =(0,1) ) )

        bar_up = jnp.fft.irfft2( o_filt_fn( u_hat[...,0] - o_filt_fn( u_hat[...,0] )), axes = (0,1) )
        bar_vp = jnp.fft.irfft2( o_filt_fn( u_hat[...,1] - o_filt_fn( u_hat[...,1] ) ), axes = (0,1) )

        R = jnp.zeros( u_hat.shape[:2] + (3,), dtype= u_hat.dtype )

        R = R.at[...,0].set( upup_bar - jnp.fft.rfft2( bar_up**2, axes = (0,1) ) )
        R = R.at[...,1].set( vpvp_bar - jnp.fft.rfft2( bar_vp**2, axes = (0,1) ) )
        R = R.at[...,2].set( upvp_bar - jnp.fft.rfft2( bar_up*bar_vp, axes = (0,1) ) )


        R = spectral_utils.down_res(R, self.grid.shape)

        if self.smooth:
            R *= self.filter_[...,None]

        return R
    
    def flux_tensor(self, state):
        # Unpack the state
        U_les, R_ij = state # already in fourier space
        U_space = jnp.fft.irfft2( U_les, axes = (0,1) )
        R_space = jnp.fft.irfft2( R_ij, axes = (0,1))
 
        if self.smooth:
            U_les *= self.filter_[...,None]
            R_ij *= self.filter_[...,None]
        # Gradient U
        dUdx = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None] * U_les, axes = (0,1) )
        dUdy = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None] * U_les, axes = (0,1) )
        # Compute tau ngm
        dUdxdx = jnp.fft.irfft2( self.two_i_pi**2 * self.kx[...,None]**2 * U_les, axes = (0,1) )
        dUdydy = jnp.fft.irfft2( self.two_i_pi**2 * self.ky[...,None]**2 * U_les, axes = (0,1) )
        dUdxdy = jnp.fft.irfft2( self.two_i_pi**2 * self.ky[...,None]*self.kx[...,None] * U_les, axes = (0,1) )
        # Array ngmR as tau_xx, tau_yy, tau_xy
        ngm2xx = dUdx[...,0]**2 + dUdy[...,0]**2
        ngm2yy = dUdx[...,1]**2 + dUdy[...,1]**2
        ngm2xy = dUdx[...,0]*dUdx[...,1] + dUdy[...,0]*dUdy[...,1]             
        ngmR = self.Delta**2/12 * jnp.stack([ngm2xx, ngm2yy, ngm2xy], axis=-1)
        # Calculate ngm4
        ngm4xx = (dUdxdx[...,0]**2 + dUdydy[...,0]**2 + 2 * dUdxdy[...,0]**2) 
        ngm4yy = (dUdxdx[...,1]**2 + dUdydy[...,1]**2 + 2 * dUdxdy[...,1]**2)
        ngm4xy = (dUdxdx[...,0] * dUdxdx[...,1] + 2 * dUdxdy[...,0] * dUdxdy[...,1] + dUdydy[...,0] * dUdydy[...,1])
        ngm4 = self.Delta**4/288 * jnp.stack([ngm4xx, ngm4yy, ngm4xy], axis=-1)
        ngmR += ngm4
        ngmR =  ngmR + R_space
 
        return ngmR
    
    # TODO write flux function
    def flux_fn(self):
       
      R = jnp.sqrt(self.kx**2 + self.ky**2)*2*jnp.pi
      # Create the filter kernel
      G_fn = lambda Delta: jnp.exp(- (R**2) * (Delta**2) / 24.0)

      def _flux_fn(state,Delta):
        # Already in fourier space
        U_les, R_ij = state 
        U_space = jnp.fft.irfft2( U_les, axes = (0,1) )
        # Make filter
        G = G_fn(Delta)

        if self.smooth:
            U_les *= self.filter_[...,None]
            
        # Gradient U
        dUdx = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None] * U_les, axes = (0,1) )
        dUdy = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None] * U_les, axes = (0,1) )
        # Compute advection LES
        advection = -( dUdx * U_space[...,0:1] + dUdy * U_space[...,1:2] )
        advection = jnp.fft.irfft( advection, axes = (0,1) )
        T = G[...,None] * advection

        U_space = jnp.fft.irfft2( G[...,None] * U_les, axes = (0,1) )
        advection = -( dUdx * U_space[...,0:1] + dUdy * U_space[...,1:2] )
        advection = jnp.fft.irfft( advection, axes = (0,1) )
        T -= advection

        # Compute tau ngm
        dUdxdx = jnp.fft.irfft2( self.two_i_pi**2 * self.kx[...,None]**2 * U_les, axes = (0,1) )
        dUdydy = jnp.fft.irfft2( self.two_i_pi**2 * self.ky[...,None]**2 * U_les, axes = (0,1) )
        dUdxdy = jnp.fft.irfft2( self.two_i_pi**2 * self.ky[...,None]*self.kx[...,None] * U_les, axes = (0,1) )

        ngmR = jnp.fft.irfft2( R_ij, axes = (0,1))
        # Calculate ngm4
        ngmR = ngmR.at[...,0].add( self.Delta**2/12* jnp.sum( dUdx**2, axis = -1 ) )
        ngmR = ngmR.at[...,1].add( self.Delta**2/12* jnp.sum( dUdy**2, axis = -1 ) )
        ngmR = ngmR.at[...,2].add( self.Delta**2/12* jnp.sum( dUdx*dUdy, axis = -1 ) )
        ngmR = ngmR.at[...,0].add( self.Delta**4/288 * (dUdxdx[...,0]**2 + dUdydy[...,0]**2 + 2 * dUdxdy[...,0]**2) )
        ngmR = ngmR.at[...,1].add( self.Delta**4/288 * (dUdxdx[...,1]**2 + dUdydy[...,1]**2 + 2 * dUdxdy[...,1]**2) )
        ngmR = ngmR.at[...,2].add( self.Delta**4/288 * (dUdxdx[...,0] * dUdxdx[...,1] + 2 * dUdxdy[...,0] * dUdxdy[...,1] + dUdydy[...,0] * dUdydy[...,1]) )
        
        ngmR = jnp.fft.rfft2( ngmR, axes = (0,1) )



        return 0
       

      return _flux_fn
    

@dataclasses.dataclass
class NS2D_LES_NGM_W_DNS(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid_dns: grids.Grid
    grid_les: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
    diff_order: int = 1
    regularize: bool = True
    forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
    sgs_model_fn: Optional[Callable[[grids.Grid], types.Spectral2DForcingFn]] = None
    _forcing_fn_with_grid = None
    _hyper_diss_fn = None
 
    def __post_init__(self):
        
        self.kx_les, self.ky_les = self.grid_les.rfft_mesh()
        self.kx_dns, self.ky_dns = self.grid_dns.rfft_mesh()
        
        self.two_i_pi = jnp.pi * 2j
        self.laplace_les = ( self.two_i_pi )**2 * (self.kx_les**2 + self.ky_les**2)
        self.laplace_dns = ( self.two_i_pi )**2 * (self.kx_dns**2 + self.ky_dns**2)

        if self.k_filter is None:
          self.filter_les = spectral_utils.brick_wall_filter_2d(self.grid_les)
          self.filter_dns = spectral_utils.brick_wall_filter_2d(self.grid_dns)
        else:
          self.filter_les = spectral_utils.brick_wall_filter_2d(self.grid_les, self.k_filter)
          self.filter_dns = spectral_utils.brick_wall_filter_2d(self.grid_dns, self.k_filter)

        # Build linear terms  
        self.linear_term_les = - self.viscosity * (- self.laplace_les)**self.diff_order - self.drag
        self.linear_term_dns = - self.viscosity * (- self.laplace_dns)**self.diff_order - self.drag
        
        self.w_to_v_les = spectral_utils.vorticity_to_velocity(self.grid_les)
        self.w_to_v_dns = spectral_utils.vorticity_to_velocity(self.grid_dns)
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta, self.kx_dns, self.ky_dns )

        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)
        # Regularize solution
        #if self.regularize:
        
        def hyper_diss(w_hat):  
            sigma2 = self.Delta**2/12
            u, v = self.w_to_v_les(w_hat)
            tau_xx_6th, tau_xy_6th, tau_yy_6th = tau_third_order( u, v, self.kx_les, 
                                                                    self.ky_les, sigma2**3/4)
            trR = tau_xx_6th**2 + 2*tau_xy_6th**2 + tau_yy_6th**2
            nabla6 = jnp.fft.irfft2( self.laplace_les**3 * w_hat )
            diss = trR**(1/4) * sigma2**4/2/self.Delta**3 * nabla6
            return jnp.fft.rfft2(diss)
        self._hyper_diss_fn = hyper_diss

        if self.sgs_model_fn is not None:
          def sgs_curl_hat_fn(w_hat):
            u, v = self.w_to_v_les(w_hat)
            tau_xx, tau_xy, tau_yy = self.sgs_model_fn(u, v, 
                                                       self.kx_les, self.ky_les, self.Delta)
            div_tau_x, div_tau_y = spectral_utils.divergence_of_sgs_stress_fft(
               tau_xx, tau_xy, tau_yy, self.kx_les, self.ky_les)
            div_tau_x_hat = jnp.fft.rfft2( div_tau_x, axes =(0,1) )
            div_tau_y_hat = jnp.fft.rfft2( div_tau_y, axes =(0,1) )
            sgs_curl_hat = spectral_utils.spectral_curl_2d(( self.kx_les, self.ky_les), ( div_tau_x_hat, div_tau_y_hat))
            return sgs_curl_hat
          self._sgs_curl_fn = sgs_curl_hat_fn
 
    def explicit_terms(self, state):
        
        # Solve velocity
        w_les, w_dns = state

        if self.smooth:
            #vorticity_hat = self.filter_fn(vorticity_hat)
            w_les *= self.filter_les
            w_dns *= self.filter_dns

        # Advection les
        uhat_les, vhat_les = self.w_to_v_les( w_les )
        u = jnp.fft.irfftn( uhat_les, axes = (0,1) )
        v = jnp.fft.irfftn( vhat_les, axes = (0,1) )
        dwdx = jnp.fft.irfftn( self.two_i_pi * self.kx_les * w_les, axes = (0,1) )
        dwdy = jnp.fft.irfftn( self.two_i_pi * self.ky_les * w_les, axes = (0,1) )
        advection = -( dwdx * u + dwdy * v)
        advection_hat = jnp.fft.rfftn( advection, axes = (0,1) )
        terms_les = advection_hat
 
        # Advection dns
        uhat_dns, vhat_dns = self.w_to_v_dns( w_dns )
        u = jnp.fft.irfftn( uhat_dns, axes = (0,1) )
        v = jnp.fft.irfftn( vhat_dns, axes = (0,1) )
        dwdx = jnp.fft.irfftn( self.two_i_pi * self.kx_dns * w_dns, axes = (0,1) )
        dwdy = jnp.fft.irfftn( self.two_i_pi * self.ky_dns * w_dns, axes = (0,1) )
        advection = -( dwdx * u + dwdy * v)
        advection_hat = jnp.fft.rfftn( advection, axes = (0,1) )
        terms_dns = advection_hat

        if self.sgs_model_fn is not None:
          sgs_curl_hat = self._sgs_curl_fn(w_les)
          terms_les -= sgs_curl_hat#*self.G

        if self.regularize:
            diss_les = self._hyper_diss_fn(w_les)
            terms_les +=  diss_les   
        #terms *= self.G

        if self.smooth:
            #terms = self.filter_fn(terms)
            terms_les *= self.filter_les
            terms_dns *= self.filter_dns
 
        return terms_les, terms_dns
 
    def implicit_terms(self, state):
        w_les, w_dns = state
        return self.linear_term_les * w_les, self.linear_term_dns * w_dns
 
    def implicit_solve(self, state, time_step):
        w_les, w_dns = state
        return 1 / (1 - time_step * self.linear_term_les) * w_les, 1 / (1 - time_step * self.linear_term_dns) * w_dns
    #self.filter_fn(  )

    def compute_flux_fn(self):
        
        Nx, Ny = self.grid_les.shape
        # Create the filter kernel
        def G_fn( Delta, kx,  ky):
           R = jnp.sqrt( kx**2 + ky**2)*2*jnp.pi
           return jnp.exp(- (R**2) * (Delta**2) / 24.0)


        # Helper function to compute advection
        def adv_fn( w, w_to_v, Dx, Dy ):
            u_hat, v_hat = w_to_v(w)
            # Compute advection LES
            # Filter

            u = jnp.fft.irfftn(u_hat)
            v = jnp.fft.irfftn(v_hat)
            w = jnp.fft.irfftn(w)
            # Compute in conservative form
            dx_uw = Dx * jnp.fft.rfftn( u*w, axes = (0,1) )
            dy_vw = Dy * jnp.fft.rfftn( v*w, axes = (0,1) )
            rhs_adv = ( dx_uw + dy_vw )

            return rhs_adv

        def get_tau_dns( w_dns ):
            # Compute tau from DNS state
            u_hat, v_hat = self.w_to_v_dns( w_dns )
            u = jnp.fft.irfftn(u_hat )
            v = jnp.fft.irfftn(v_hat )
            u_hat = self.filter_fn(u_hat)
            v_hat = self.filter_fn(v_hat)
            uu_bar = jnp.fft.rfftn( jnp.fft.irfftn( u_hat )**2 )
            vv_bar = jnp.fft.rfftn( jnp.fft.irfftn( v_hat )**2 )
            uv_bar = jnp.fft.rfftn( jnp.fft.irfftn( u_hat )*jnp.fft.irfftn( v_hat ) )
            
            tau_xx = self.filter_fn( jnp.fft.rfftn(u*u) ) - uu_bar
            tau_xy = self.filter_fn( jnp.fft.rfftn(u*v) ) - uv_bar
            tau_yy = self.filter_fn( jnp.fft.rfftn(v*v) ) - vv_bar

            div_curl_tau = ( (self.two_i_pi*self.kx_dns)**2 - (self.two_i_pi*self.ky_dns)**2 ) * tau_xy \
                            + self.two_i_pi**2*self.kx_dns*self.ky_dns*(tau_yy - tau_xx)
            return div_curl_tau * self.filter_dns

        def flux_fn_(state, Delta):
            
            # Unpack state, should be in spectral state
            w_les, w_dns  = state 
            if self.smooth:
                w_dns *= self.filter_dns
                w_les *= self.filter_les
            # Project LES filter to DNS
            w_dns_filtered = self.filter_fn(w_dns)

            G =  G_fn( Delta, self.kx_les,  self.ky_les)

            # Use large grid to compute tau
            div_curl_tau = get_tau_dns( w_dns )
            # Calcualte ngm with LES field
            ngm_curl_hat = self._sgs_curl_fn(w_les)

            # Calculate flux in LES
            rhs_tau_les = ngm_curl_hat * self.filter_les * G
            rhs_tau_les = jnp.fft.irfft2( rhs_tau_les , s = (Nx,Ny) )
            
            # Compute advection LES
            rhs_adv_les = adv_fn( w_les, self.w_to_v_les,
                                 self.two_i_pi * self.kx_les, self.two_i_pi * self.ky_les )
            rhs_adv_f_les = adv_fn( G * w_les, self.w_to_v_les,
                                 self.two_i_pi * self.kx_les, self.two_i_pi * self.ky_les )
            # Filter
            rhs_adv_les = jnp.fft.irfft2( rhs_adv_les * self.filter_les * G , s = (Nx,Ny) )
            rhs_adv_f_les = jnp.fft.irfft2( rhs_adv_f_les * self.filter_les , s = (Nx,Ny) )

            ω_les = jnp.fft.irfft2( G * w_les, s = (Nx,Ny) )
            flux_adv = - ω_les * ( rhs_adv_les - rhs_adv_f_les )
            flux_tau = - ω_les * rhs_tau_les

            # Calculate flux in DNS
            G =  G_fn(Delta,self.kx_dns,self.ky_dns)

            ω_dns = jnp.fft.irfft2( G * w_dns_filtered )
            rhs_adv_filtered_dns = adv_fn( w_dns_filtered * G, self.w_to_v_dns,
                                 self.two_i_pi * self.kx_dns, self.two_i_pi * self.ky_dns )
            rhs_adv_filtered_dns = jnp.fft.irfft2( rhs_adv_filtered_dns * self.filter_dns )

            double_fl_adv = adv_fn( w_dns_filtered  * self.filter_dns * G, self.w_to_v_dns,
                                 self.two_i_pi * self.kx_dns, self.two_i_pi * self.ky_dns   )
            double_fl_adv = jnp.fft.irfft2( double_fl_adv )
            flux_dns = - ω_dns * ( double_fl_adv - rhs_adv_filtered_dns )

            rhs_adv_dns = adv_fn( w_dns_filtered, self.w_to_v_dns,
                                 self.two_i_pi * self.kx_dns, self.two_i_pi * self.ky_dns )
            rhs_adv_dns = jnp.fft.irfft2( rhs_adv_dns * self.filter_dns * G  )

            flux_filt_adv = - ω_dns * ( rhs_adv_dns - rhs_adv_filtered_dns)

            

            if self.smooth:
               div_curl_tau *= self.filter_dns * G

            div_curl_tau = jnp.fft.irfft2( div_curl_tau )
            #ω = jnp.fft.irfft2( G * w_dns_filtered, s = (Nx,Ny) )
            flux_dns_tau = - ω_dns * div_curl_tau

            tot_flux_adv = jnp.mean(flux_adv)
            tot_flux_tau = jnp.mean(flux_tau)
            tot_flux_dns = jnp.mean(flux_dns)
            tot_flux_filt_adv = jnp.mean(flux_filt_adv)
            tot_flux_dns_tau = jnp.mean(flux_dns_tau)

            return tot_flux_adv, tot_flux_tau, tot_flux_dns, tot_flux_filt_adv, tot_flux_dns_tau

        return flux_fn_
    
@dataclasses.dataclass
class LES_LC_w_DNS(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid_dns: grids.Grid
    grid_les: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
    diff_order: int = 1
    forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
    _forcing_fn_with_grid = None

    def __post_init__(self):

        self.kx_les, self.ky_les = self.grid_les.rfft_mesh()
        self.kx_dns, self.ky_dns = self.grid_dns.rfft_mesh()
        
        self.two_i_pi = jnp.pi * 2j
        self.laplace_les = ( self.two_i_pi )**2 * (self.kx_les**2 + self.ky_les**2)
        self.laplace_dns = ( self.two_i_pi )**2 * (self.kx_dns**2 + self.ky_dns**2)

        if self.k_filter is None:
          self.filter_les = spectral_utils.brick_wall_filter_2d(self.grid_les)
          self.filter_dns = spectral_utils.brick_wall_filter_2d(self.grid_dns)
        else:
          self.filter_les = spectral_utils.brick_wall_filter_2d(self.grid_les, self.k_filter)
          self.filter_dns = spectral_utils.brick_wall_filter_2d(self.grid_dns, self.k_filter)

        # Build linear terms  
        self.linear_term_les = - self.viscosity * (- self.laplace_les)**self.diff_order - self.drag
        self.linear_term_dns = - self.viscosity * (- self.laplace_dns)**self.diff_order - self.drag
        
        self.w_to_v_les = spectral_utils.vorticity_to_velocity(self.grid_les)
        self.w_to_v_dns = spectral_utils.vorticity_to_velocity(self.grid_dns)
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta, self.kx_les, self.ky_les )

        self.les_vis_order = 2

        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid_les)

        def bracket( psi_hat, phi_hat ):
           
          psi = jnp.fft.irfftn( psi_hat )
          phi = jnp.fft.irfftn( phi_hat )
           
          psi_bar = jnp.fft.irfftn( self.filter_fn( psi_hat ) )
          phi_bar = jnp.fft.irfftn( self.filter_fn( phi_hat ) )

          B = self.filter_fn(jnp.fft.rfftn( psi*phi )) - jnp.fft.rfftn(psi_bar * phi_bar)
          return B

        self.bracket_op = bracket

           

    def explicit_terms(self, state):
       
        # Solve velocity
        w_les, w_dns = state

        if self.smooth:
            #vorticity_hat = self.filter_fn(vorticity_hat)
            w_les *= self.filter_les
            w_dns *= self.filter_dns

        # Advection les
        u_hat_les, v_hat_les = self.w_to_v_les( w_les )
        u = jnp.fft.irfftn( u_hat_les, axes = (0,1) )
        v = jnp.fft.irfftn( v_hat_les, axes = (0,1) )
        dwdx = jnp.fft.irfftn( self.two_i_pi * self.kx_les * w_les, axes = (0,1) )
        dwdy = jnp.fft.irfftn( self.two_i_pi * self.ky_les * w_les, axes = (0,1) )
        advection = -( dwdx * u + dwdy * v)
        advection_hat = jnp.fft.rfftn( advection, axes = (0,1) )
        terms_les = advection_hat
 
        # Compute tau
        tau_xx = self.bracket_op( u_hat_les, u_hat_les )
        tau_yy = self.bracket_op( v_hat_les, v_hat_les )
        tau_xy = self.bracket_op( u_hat_les, v_hat_les )

        u_pr = self.filter_fn( u_hat_les ) - u_hat_les
        v_pr = self.filter_fn( v_hat_les ) - v_hat_les

        tau_xx += 2*self.bracket_op( u_hat_les, u_pr )
        tau_yy += 2*self.bracket_op( v_hat_les, v_pr )
        tau_xy += self.bracket_op( u_pr, v_hat_les ) + self.bracket_op( v_pr, u_hat_les )

        tau_xx += self.bracket_op( u_pr, u_pr )
        tau_yy += self.bracket_op( v_pr, v_pr )
        tau_xy += self.bracket_op( u_pr, v_pr )
        
        div_curl_tau = ( (self.two_i_pi*self.kx_les)**2 - (self.two_i_pi*self.ky_les)**2 ) * tau_xy \
                            + self.two_i_pi**2*self.kx_les*self.ky_les*(tau_yy - tau_xx)
            

        terms_les -= div_curl_tau
        # Compute strain invariant
        trR = self.bracket_op( u_pr, u_pr ) + self.bracket_op( v_pr, v_pr )
        if self.smooth is True:
           trR *= self.filter_les
        IR = jnp.sqrt( jnp.abs( jnp.fft.irfftn( trR ) ) )
        Lw = self.Delta**(2*self.les_vis_order)*(-self.laplace_les)**self.les_vis_order*w_les
        Lw = jnp.fft.rfftn( IR * jnp.fft.irfftn(Lw) )

        terms_les -= Lw

        # Advection dns
        uhat_dns, vhat_dns = self.w_to_v_dns( w_dns )
        u = jnp.fft.irfftn( uhat_dns, axes = (0,1) )
        v = jnp.fft.irfftn( vhat_dns, axes = (0,1) )
        dwdx = jnp.fft.irfftn( self.two_i_pi * self.kx_dns * w_dns, axes = (0,1) )
        dwdy = jnp.fft.irfftn( self.two_i_pi * self.ky_dns * w_dns, axes = (0,1) )
        advection = -( dwdx * u + dwdy * v)
        advection_hat = jnp.fft.rfftn( advection, axes = (0,1) )
        terms_dns = advection_hat


        if self.smooth:
            #terms = self.filter_fn(terms)
            terms_les *= self.filter_les
            terms_dns *= self.filter_dns
 
        return terms_les, terms_dns

    def implicit_terms(self, state):
        w_les, w_dns = state
        return self.linear_term_les * w_les, self.linear_term_dns * w_dns
 
    def implicit_solve(self, state, time_step):
        w_les, w_dns = state
        return 1 / (1 - time_step * self.linear_term_les) * w_les, 1 / (1 - time_step * self.linear_term_dns) * w_dns
    

@dataclasses.dataclass
class LES_w_RM(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
 
    def __post_init__(self):
        # FFT functions and mesh
        self.rfft2 = jnp.fft.rfft2
        self.irfft2 = jnp.fft.irfft2
        self.kx, self.ky = self.grid.rfft_mesh()
 
        # Spectral constants
        self.two_i_pi = 2j * jnp.pi
        # Differentiation multipliers, expanded for vector components
        self.two_ip_kx = (self.two_i_pi * self.kx)[..., None]  # (Nx, Ny//2+1, 1)
        self.two_ip_ky = (self.two_i_pi * self.ky)[..., None]
 
        # Laplacian and its inverse for projection
        self.laplace = (self.two_i_pi**2) * (self.kx**2 + self.ky**2)  # (Nx, Ny//2+1)
        self.laplace_n = self.laplace[..., None]  # (Nx, Ny//2+1, 1)
        self.inv_laplace = jnp.where(self.laplace == 0, 1.0, 1.0/self.laplace)[..., None]
 
        # Projection vector (kx, ky)
        self.proj = jnp.stack([self.two_i_pi*self.kx,
                               self.two_i_pi*self.ky], axis=-1)  # (Nx, Ny//2+1, 2)
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta, self.kx, self.ky )
        # Spatial filter
        filt = (spectral_utils.brick_wall_filter_2d(self.grid)
                if self.k_filter is None
                else spectral_utils.brick_wall_filter_2d(self.grid, self.k_filter))
        self.filter_spatial = filt[..., None]  # (Nx, Ny//2+1, 1)
        # Linear operators for implicit solve
        self.linear  = (self.viscosity * self.laplace - self.drag)[..., None]
        self.linearR = (self.viscosity * (self.laplace - 48.0/self.Delta**2))[..., None]
 
    def explicit_terms(self, state):
        U_hat, R_hat = state
        fft, ifft = self.rfft2, self.irfft2
 
        # Real-space velocity and R
        U = ifft(U_hat, axes=(0, 1))
        R = ifft(R_hat, axes=(0, 1))
 
        # Filtered or raw velocity in spectral space
        U_f = U_hat * self.filter_spatial if self.smooth else U_hat
 
        # First derivatives
        dUdx    = ifft(self.two_ip_kx * U_f,      axes=(0, 1))
        dUdy    = ifft(self.two_ip_ky * U_f,      axes=(0, 1))
        dUdxlap = ifft(self.two_ip_kx * self.laplace_n * U_f, axes=(0, 1))
        dUdylap = ifft(self.two_ip_ky * self.laplace_n * U_f, axes=(0, 1))
 
        # Nonlinear advection term
        adv = -(dUdx * U[..., 0:1] + dUdy * U[..., 1:2])
        term_u = fft(adv, axes=(0, 1))
 
        # Compute NGM2, NGM4, Forcing, Eddy-viscosity...
        c6 = self.Delta**6 / 6912
        I1 = jnp.sqrt(4*dUdx[...,0]**2 + (dUdx[...,1] + dUdy[...,0])**2)
        I2 = jnp.sqrt((dUdy[...,0] - dUdx[...,1])**2)
 
        d2xx = ifft((self.two_ip_kx**2) * U_f, axes=(0, 1))
        d2yy = ifft((self.two_ip_ky**2) * U_f, axes=(0, 1))
        d2xy = ifft((self.two_ip_kx * self.two_ip_ky) * U_f, axes=(0, 1))
 
        sq  = self.Delta**2 / 12.0
        q4  = self.Delta**4 / 288.0
        ngm2 = sq * jnp.stack([(dUdx[...,0]**2 + dUdy[...,0]**2),(dUdx[...,1]**2 + dUdy[...,1]**2),(dUdx[...,0]*dUdx[...,1] + dUdy[...,0]*dUdy[...,1])], axis=-1)
        ngm4 = q4 * jnp.stack([(d2xx[...,0]**2 + d2yy[...,0]**2 + 2*d2xy[...,0]**2),(d2xx[...,1]**2 + d2yy[...,1]**2 + 2*d2xy[...,1]**2),(d2xx[...,0]*d2xx[...,1] + 2*d2xy[...,0]*d2xy[...,1] + d2yy[...,0]*d2yy[...,1])], axis=-1)
        Forcing = c6  * jnp.stack([dUdxlap[...,0]**2 + dUdylap[...,0]**2,dUdxlap[...,1]**2 + dUdylap[...,1]**2,dUdxlap[...,0]*dUdxlap[...,1] + dUdylap[...,0]*dUdylap[...,1]], axis=-1)
 
        Eddy_viscosity = 10*(self.Delta**3 / (1728) *jnp.sqrt(ngm4[...,0] + ngm4[...,1] + Forcing[...,0] + Forcing[...,1])[..., None] *jnp.stack([dUdxlap[...,0],dUdylap[...,1],(dUdxlap[...,1] + dUdylap[...,0]) / 2], axis=-1))
 
        # Total tau in spectral
        tau_hat = fft(ngm2 + ngm4 + Eddy_viscosity, axes=(0,1)) + R_hat
 
        # Divergence of subgrid-stress
        tau_u = self.two_i_pi * jnp.stack([
            self.kx * tau_hat[...,0] + self.ky * tau_hat[...,2],
            self.kx * tau_hat[...,2] + self.ky * tau_hat[...,1]
        ], axis=-1)
        term_u = term_u - tau_u
 
        # Enforce incompressibility via projection
        div_u  = self.two_i_pi * (self.kx * term_u[...,0] + self.ky * term_u[...,1])
        term_u = term_u - self.proj * (div_u[..., None] * self.inv_laplace)
 
        # R-equation RHS
        adv_R = -(U[...,0:1] * ifft(self.two_ip_kx * R_hat, axes=(0,1)) +
                  U[...,1:2] * ifft(self.two_ip_ky * R_hat, axes=(0,1)))
        rhs_R = (adv_R +jnp.stack([2*(dUdx[...,0]*R[...,0] + dUdy[...,0]*R[...,2]),2*(dUdx[...,1]*R[...,2] + dUdy[...,1]*R[...,1]),(dUdx[...,1]*R[...,0] + dUdy[...,0]*R[...,2])], axis=-1) +Forcing*I2[..., None]/5 - (I1 + I2)[..., None] * R)
        rhs_R = fft(rhs_R, axes=(0,1))
 
        # Optional filtering
        if self.smooth:
            term_u *= self.filter_spatial
            rhs_R   *= self.filter_spatial
 
        return term_u, rhs_R
 
    def implicit_terms(self, state):
        U_hat, R_hat = state
        return self.linear * U_hat, self.linearR * R_hat
 
    def implicit_solve(self, y_hat, dt):
        U_hat, R_hat = y_hat
        invU = 1/(1 - dt*self.linear)
        invR = 1/(1 - dt*self.linearR)
        return U_hat * invU, R_hat * invR
    
    def init_R( self, u_les, grid_aux: grids.Grid):
      """
      Calculates R from u_les
      """
      kx ,ky = grid_aux.rfft_mesh()
      o_filt_fn = spectral_utils.get_filter_fn( self.Delta, kx, ky )
      # Filter field
      up_sp = jnp.fft.irfft2( u_les - o_filt_fn( u_les ), axes = (0,1) )
      upup_bar = o_filt_fn( jnp.fft.rfft2( up_sp[...,0]**2         , axes =(0,1) ) )
      vpvp_bar = o_filt_fn( jnp.fft.rfft2( up_sp[...,1]**2         , axes =(0,1) ) )
      upvp_bar = o_filt_fn( jnp.fft.rfft2( up_sp[...,0]*up_sp[...,1], axes =(0,1) ) )

      bar_up = jnp.fft.irfft2( o_filt_fn( u_les[...,0] - o_filt_fn( u_les[...,0] )), axes = (0,1) )
      bar_vp = jnp.fft.irfft2( o_filt_fn( u_les[...,1] - o_filt_fn( u_les[...,1] ) ), axes = (0,1) )

      R = jnp.zeros( u_les.shape[:2] + (3,), dtype= u_les.dtype )

      R = R.at[...,0].set( upup_bar - jnp.fft.rfft2( bar_up**2, axes = (0,1) ) )
      R = R.at[...,1].set( vpvp_bar - jnp.fft.rfft2( bar_vp**2, axes = (0,1) ) )
      R = R.at[...,2].set( upvp_bar - jnp.fft.rfft2( bar_up*bar_vp, axes = (0,1) ) )


      R = spectral_utils.down_res(R, self.grid.shape)

      if self.smooth:
         R *= self.filter_spatial

      return R
    

@dataclasses.dataclass
class DNS_R(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
 
    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        self.two_i_pi = 2j * jnp.pi
        self.laplace = self.two_i_pi**2 * (self.kx**2 + self.ky**2)
        if self.k_filter is None:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        else:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid,self.k_filter)

        self.linear_term = self.viscosity * self.laplace - self.drag
        #(self.viscosity * self.laplace - self.drag + (1/256)**6*self.laplace**3) * self.filter2_
        self.linear_R = self.laplace[...,None]
        self.w_to_v = spectral_utils.vorticity_to_velocity(self.grid)
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta, self.kx, self.ky )
 
    def explicit_terms(self, state):
        # Unpack the state
        w_hat, R_hat = state # already in fourier space
        # R will be as [{xx}, {yy}, {xy}]

        if self.smooth:
            R_hat *= self.filter_[...,None]
            w_hat *= self.filter_

        # Solve velocity
        u_hat, v_hat = self.w_to_v(w_hat)

        # Real space variables
        u = jnp.fft.irfft2(u_hat)
        v = jnp.fft.irfft2(v_hat)
        w = jnp.fft.irfft2(w_hat)
        R_ij = jnp.fft.irfft2( R_hat, axes = (0,1) )

        ### vorticirt evolution ###

        # Compute advection in conservative form
        dx_uw = self.two_i_pi * self.kx * jnp.fft.rfftn( u*w, axes = (0,1) )
        dy_vw = self.two_i_pi * self.ky * jnp.fft.rfftn( v*w, axes = (0,1) )
        adv = ( dx_uw + dy_vw )
        terms_w = - adv

        ### stress tensor evolutoin ###

        # Velocity gradients
        dudx = jnp.fft.irfft2( self.two_i_pi * self.kx * self.filter_fn( u_hat ) )
        dudy = jnp.fft.irfft2( self.two_i_pi * self.ky * self.filter_fn( u_hat ) )
        dvdx = jnp.fft.irfft2( self.two_i_pi * self.kx * self.filter_fn( v_hat ) )
        dvdy = jnp.fft.irfft2( self.two_i_pi * self.ky * self.filter_fn( v_hat ) )

        # Advection for R
        dRdx = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None] * R_hat, axes = (0,1) )
        dRdy = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None] * R_hat, axes = (0,1) )
        adv_R = jnp.fft.rfft2( u[...,None] * dRdx + v[...,None] * dRdy, axes = (0,1) )
        # Ru_hat = jnp.fft.rfft2( u[...,None] * R_ij, axes =(0,1) )
        # Rv_hat = jnp.fft.rfft2( v[...,None] * R_ij, axes =(0,1) )
        # dRudx = self.two_i_pi * self.kx[...,None] * Ru_hat
        # dRvdx = self.two_i_pi * self.ky[...,None] * Rv_hat
        # adv_R = dRudx + dRvdx
        terms_R = - adv_R
        # Production for R
        prod_R = jnp.stack([2*R_ij[...,0] * dudx + 2*R_ij[...,2] * dudy,
                            2*R_ij[...,2] * dvdx + 2*R_ij[...,1] * dvdy,
                            R_ij[...,0] * dvdx + R_ij[...,1] * dvdy, ], axis = -1)
        terms_R += jnp.fft.rfft2( prod_R, axes = (0,1) )
        #terms_R += - jnp.fft.rfft2( prod_R, axes = (0,1) ) \
        #    + 2 * self.filter_fn( jnp.fft.rfft2( prod_R, axes = (0,1) ) )
 
        # Compute ngm R
        dLudx = jnp.fft.irfft2( self.two_i_pi * self.kx * self.laplace * self.filter_fn( u_hat ) )
        dLudy = jnp.fft.irfft2( self.two_i_pi * self.ky * self.laplace * self.filter_fn( u_hat ) )
        dLvdx = jnp.fft.irfft2( self.two_i_pi * self.kx * self.laplace * self.filter_fn( v_hat ) )
        dLvdy = jnp.fft.irfft2( self.two_i_pi * self.ky * self.laplace * self.filter_fn( v_hat ) )

        F_ij = jnp.stack([ ( dLudx**2 + dLudy**2 ),
                           ( dLvdx**2 + dLvdy**2 ),
                           ( dLudx*dLvdx + dLudy*dLvdy )], axis=-1)
 
        F_ij *= self.Delta**6/6912
        # Compute invariants
        I1 = jnp.sqrt(4*dudx**2 + (dvdx + dudy)**2)
        I2 = jnp.sqrt(( dudy - dvdx )**2) #( )
        I = (I1[...,None] + I2[...,None])
        if self.smooth:
           I1 = jnp.fft.irfft2( self.filter_ * jnp.fft.rfft2( I1 ) )
        #F_ij
        F_ij = jnp.fft.irfft2( self.filter_fn( jnp.fft.rfft2( F_ij, axes = (0,1) ) ), axes = (0,1) )
        linear_term = jnp.fft.rfft2( I * ( F_ij - R_ij ), axes = (0,1) )
        terms_R -= jnp.fft.rfft2( 1/2 * I1[...,None] * R_ij, axes = (0,1) )
        #terms_R += linear_term# * (- jnp.sign(linear_term))
        self.nu_R = 0*I1.mean() * self.Delta**6/12**3/12

        if self.smooth:
            terms_w *= self.filter_
            terms_R *= self.filter_[...,None]
 
        return terms_w, terms_R
 
    def implicit_terms(self, state):
        # Unpack the state
        w_hat, R_hat = state
        return self.linear_term * w_hat, \
              (self.nu_R*self.linear_R**3 + self.viscosity*self.linear_R) * R_hat
 
    def implicit_solve(self, y_hat, dt):
        """
        Solves 1 - dt G x = y, for x
        """
        w_hat, R_hat = y_hat
        return 1 / (1 - dt * self.linear_term) * w_hat, \
              1 / (1 - dt * (self.nu_R*self.linear_R**3  + self.viscosity * self.linear_R)) * R_hat
    
    def init_R( self, u_hat, grid_aux: grids.Grid):
        """
        Calculates R from u_les
        """
        kx ,ky = grid_aux.rfft_mesh()
        o_filt_fn = spectral_utils.get_filter_fn( self.Delta, kx, ky )
        # Filter field
        up = jnp.fft.irfft2( u_hat - o_filt_fn( u_hat ), axes = (0,1) )
        upup_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]**2         , axes =(0,1) ) )
        vpvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,1]**2         , axes =(0,1) ) )
        upvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]*up[...,1], axes =(0,1) ) )

        bar_up = jnp.fft.irfft2( o_filt_fn( u_hat[...,0] - o_filt_fn( u_hat[...,0] )), axes = (0,1) )
        bar_vp = jnp.fft.irfft2( o_filt_fn( u_hat[...,1] - o_filt_fn( u_hat[...,1] ) ), axes = (0,1) )

        R = jnp.zeros( u_hat.shape[:2] + (3,), dtype= u_hat.dtype )

        R = R.at[...,0].set( upup_bar - jnp.fft.rfft2( bar_up**2, axes = (0,1) ) )
        R = R.at[...,1].set( vpvp_bar - jnp.fft.rfft2( bar_vp**2, axes = (0,1) ) )
        R = R.at[...,2].set( upvp_bar - jnp.fft.rfft2( bar_up*bar_vp, axes = (0,1) ) )


        R = spectral_utils.down_res(R, self.grid.shape)

        if self.smooth:
            R *= self.filter_[...,None]

        return R



@dataclasses.dataclass
class DNS_coarse_R(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid_dns: grids.Grid
    grid_les: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
 
    def __post_init__(self):
        self.two_i_pi = 2j * jnp.pi
        # Initialize the fine grid
        self.kx_dns, self.ky_dns = self.grid_dns.rfft_mesh()
        self.laplace_dns = self.two_i_pi**2 * (self.kx_dns**2 + self.ky_dns**2)
        # Initialize the coarse grid
        self.kx_les, self.ky_les = self.grid_les.rfft_mesh()
        self.laplace_les = self.two_i_pi**2 * (self.kx_les**2 + self.ky_les**2)
        # Define filtering
        if self.k_filter is None:
          self.filter_dns = spectral_utils.brick_wall_filter_2d(self.grid_dns)
          self.filter_les = spectral_utils.brick_wall_filter_2d(self.grid_les)
        else:
          self.filter_dns = spectral_utils.brick_wall_filter_2d(self.grid_dns,2/3)
          self.filter_les = spectral_utils.brick_wall_filter_2d(self.grid_les,self.k_filter)[...,None]

        # Viscosity and linear damping for dns
        self.linear_term_dns = self.viscosity * self.laplace_dns - self.drag
        # Laplace for dns
        self.linear_R = self.laplace_les[...,None]
        # Helper functions
        self.w_to_v_dns = spectral_utils.vorticity_to_velocity(self.grid_dns)
        self.w_to_v_les = spectral_utils.vorticity_to_velocity(self.grid_les)
        #
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta, self.kx_dns, self.ky_dns )
        self.filter_fn_les = spectral_utils.get_filter_fn( self.Delta, self.kx_les, self.ky_les )
        
    def explicit_terms(self, state):
        # Unpack the state
        w_hat, R_hat = state # already in fourier space
        # R will be as [{xx}, {yy}, {xy}]
        # R should be on the downsampled grid

        if self.smooth:
            R_hat *= self.filter_les
            w_hat *= self.filter_dns

        # Solve velocity
        u_hat, v_hat = self.w_to_v_dns(w_hat)

        # Real space variables
        u = jnp.fft.irfft2(u_hat)
        v = jnp.fft.irfft2(v_hat)
        w = jnp.fft.irfft2(w_hat)
        R_ij = jnp.fft.irfft2( R_hat, axes = (0,1) )

        ### Vorticity evolution ###

        # Compute advection in conservative form
        dx_uw = self.two_i_pi * self.kx_dns * jnp.fft.rfftn( u*w, axes = (0,1) )
        dy_vw = self.two_i_pi * self.ky_dns * jnp.fft.rfftn( v*w, axes = (0,1) )
        adv = ( dx_uw + dy_vw )
        terms_w = - adv

        ### Streess tensor evolution ###
    
        # Velocity gradients
        dudx = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.kx_dns * self.filter_fn( u_hat ), 
                                                       self.grid_les.shape ) )
        dudy = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.ky_dns * self.filter_fn( u_hat ),
                                                        self.grid_les.shape ))
        dvdx = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.kx_dns * self.filter_fn( v_hat ), 
                                                       self.grid_les.shape ) )
        dvdy = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.ky_dns * self.filter_fn( v_hat ),
                                                        self.grid_les.shape ))

        # Velocity downsampled
        u = jnp.fft.irfft2( spectral_utils.down_res( self.filter_fn( u_hat ), self.grid_les.shape ) )
        v = jnp.fft.irfft2( spectral_utils.down_res( self.filter_fn( v_hat ), self.grid_les.shape ) )


        # Advection for R
        dRdx = jnp.fft.irfft2( self.two_i_pi * self.kx_les[...,None] * R_hat, axes = (0,1) )
        dRdy = jnp.fft.irfft2( self.two_i_pi * self.ky_les[...,None] * R_hat, axes = (0,1) )
        adv_R = jnp.fft.rfft2( u[...,None] * dRdx + v[...,None] * dRdy, axes = (0,1) )
        terms_R = - adv_R
        # Production for R
        prod_R = jnp.stack([2*R_ij[...,0] * dudx + 2*R_ij[...,2] * dudy,
                            2*R_ij[...,2] * dvdx + 2*R_ij[...,1] * dvdy,
                            R_ij[...,0] * dvdx + R_ij[...,1] * dvdy, ], axis = -1)
        terms_R += jnp.fft.rfft2( prod_R, axes = (0,1) )
        # Uncomment for brandons method
        #terms_R += - jnp.fft.rfft2( prod_R, axes = (0,1) ) \
        #    + 2 * self.filter_fn( jnp.fft.rfft2( prod_R, axes = (0,1) ) )
 
        # Compute ngm R
        dLudx = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.kx_dns * self.laplace_dns * \
                                                         self.filter_fn( u_hat ), self.grid_les.shape ) )
        dLudy = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.ky_dns * self.laplace_dns * \
                                                         self.filter_fn( u_hat ), self.grid_les.shape ) )
        dLvdx = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.kx_dns * self.laplace_dns * \
                                                        self.filter_fn( v_hat ), self.grid_les.shape ) )
        dLvdy = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.ky_dns * self.laplace_dns * \
                                                        self.filter_fn( v_hat ), self.grid_les.shape ) )

        F_ij = jnp.stack([ ( dLudx**2 + dLudy**2 ),
                           ( dLvdx**2 + dLvdy**2 ),
                           ( dLudx*dLvdx + dLudy*dLvdy )], axis=-1)
 
        F_ij *= self.Delta**6/6912
        # Compute invariants
        I1 = jnp.sqrt(4*dudx**2 + (dvdx + dudy)**2)
        I2 = jnp.sqrt(( dudy - dvdx )**2) #( )
        I = I1[...,None] + I2[...,None]
        #F_ij
        #F_ij = jnp.fft.irfft2( self.filter_fn( jnp.fft.rfft2( F_ij, axes = (0,1) ) ), axes = (0,1) )
        linear_term = jnp.fft.rfft2( (I) * ( F_ij - R_ij ) - 0.1 * I * R_ij, axes = (0,1) )
        terms_R += linear_term# * (- jnp.sign(linear_term))
        self.nu_R = I1.mean() * self.Delta**6/12**3/12

        if self.smooth:
            terms_w *= self.filter_dns
            terms_R *= self.filter_les#[...,None]
 
        return terms_w, terms_R
 
    def implicit_terms(self, state):
        # Unpack the state
        w_hat, R_hat = state
        return self.linear_term_dns * w_hat, \
              (self.nu_R*self.linear_R**3 + self.viscosity*self.linear_R) * R_hat
 
    def implicit_solve(self, y_hat, dt):
        """
        Solves 1 - dt G x = y, for x
        """
        w_hat, R_hat = y_hat
        return 1 / (1 - dt * self.linear_term_dns) * w_hat, \
              1 / (1 - dt * (self.nu_R*self.linear_R**3  + self.viscosity * self.linear_R)) * R_hat
    
    def init_R( self, u_hat, grid_aux: grids.Grid):
        """
        Calculates R from u_les
        """
        kx ,ky = grid_aux.rfft_mesh()
        o_filt_fn = spectral_utils.get_filter_fn( self.Delta, kx, ky )
        # Filter field
        up = jnp.fft.irfft2( u_hat - o_filt_fn( u_hat ), axes = (0,1) )
        upup_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]**2         , axes =(0,1) ) )
        vpvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,1]**2         , axes =(0,1) ) )
        upvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]*up[...,1], axes =(0,1) ) )

        bar_up = jnp.fft.irfft2( o_filt_fn( u_hat[...,0] - o_filt_fn( u_hat[...,0] )), axes = (0,1) )
        bar_vp = jnp.fft.irfft2( o_filt_fn( u_hat[...,1] - o_filt_fn( u_hat[...,1] ) ), axes = (0,1) )

        R = jnp.zeros( u_hat.shape[:2] + (3,), dtype= u_hat.dtype )

        R = R.at[...,0].set( upup_bar - jnp.fft.rfft2( bar_up**2, axes = (0,1) ) )
        R = R.at[...,1].set( vpvp_bar - jnp.fft.rfft2( bar_vp**2, axes = (0,1) ) )
        R = R.at[...,2].set( upvp_bar - jnp.fft.rfft2( bar_up*bar_vp, axes = (0,1) ) )


        R = spectral_utils.down_res(R, self.grid_les.shape)

        if self.smooth:
            R *= self.filter_les

        return R
    
    
@dataclasses.dataclass
class DNS_coarse_TR(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid_dns: grids.Grid
    grid_les: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
 
    def __post_init__(self):
        self.two_i_pi = 2j * jnp.pi
        # Initialize the fine grid
        self.kx_dns, self.ky_dns = self.grid_dns.rfft_mesh()
        self.laplace_dns = self.two_i_pi**2 * (self.kx_dns**2 + self.ky_dns**2)
        # Initialize the coarse grid
        self.kx_les, self.ky_les = self.grid_les.rfft_mesh()
        self.laplace_les = self.two_i_pi**2 * (self.kx_les**2 + self.ky_les**2)
        # Define filtering
        if self.k_filter is None:
          self.filter_dns = spectral_utils.brick_wall_filter_2d(self.grid_dns)
          self.filter_les = spectral_utils.brick_wall_filter_2d(self.grid_les)
        else:
          self.filter_dns = spectral_utils.brick_wall_filter_2d(self.grid_dns,2/3)
          self.filter_les = spectral_utils.brick_wall_filter_2d(self.grid_les,self.k_filter)[...,None]

        # Viscosity and linear damping for dns
        self.linear_term_dns = self.viscosity * self.laplace_dns - self.drag
        # Laplace for dns
        self.linear_R = self.laplace_les[...,None]
        # Helper functions
        self.w_to_v_dns = spectral_utils.vorticity_to_velocity(self.grid_dns)
        self.w_to_v_les = spectral_utils.vorticity_to_velocity(self.grid_les)
        #
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta, self.kx_dns, self.ky_dns )
        self.filter_fn_les = spectral_utils.get_filter_fn( self.Delta, self.kx_les, self.ky_les )
        
    def explicit_terms(self, state):
        # Unpack the state
        w_hat, R_hat = state # already in fourier space
        # R will be as [{xx}, {xy}], R is assumes to be traceless
        # R should be on the downsampled grid

        if self.smooth:
            R_hat *= self.filter_les
            w_hat *= self.filter_dns

        # Solve velocity
        u_hat, v_hat = self.w_to_v_dns(w_hat)

        # Real space variables
        u = jnp.fft.irfft2(u_hat)
        v = jnp.fft.irfft2(v_hat)
        w = jnp.fft.irfft2(w_hat)
        R_ij = jnp.fft.irfft2( R_hat, axes = (0,1) )

        ### Vorticity evolution ###

        # Compute advection in conservative form
        dx_uw = self.two_i_pi * self.kx_dns * jnp.fft.rfftn( u*w, axes = (0,1) )
        dy_vw = self.two_i_pi * self.ky_dns * jnp.fft.rfftn( v*w, axes = (0,1) )
        adv = ( dx_uw + dy_vw )
        terms_w = - adv

        ### Streess tensor evolution ###
    
        # Velocity gradients
        dudx = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.kx_dns * self.filter_fn( u_hat ), 
                                                       self.grid_les.shape ) )
        dudy = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.ky_dns * self.filter_fn( u_hat ),
                                                        self.grid_les.shape ))
        dvdx = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.kx_dns * self.filter_fn( v_hat ), 
                                                       self.grid_les.shape ) )
        dvdy = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.ky_dns * self.filter_fn( v_hat ),
                                                        self.grid_les.shape ))

        # Velocity downsampled
        u = jnp.fft.irfft2( spectral_utils.down_res( self.filter_fn( u_hat ), self.grid_les.shape ) )
        v = jnp.fft.irfft2( spectral_utils.down_res( self.filter_fn( v_hat ), self.grid_les.shape ) )


        # Advection for R
        dRdx = jnp.fft.irfft2( self.two_i_pi * self.kx_les[...,None] * R_hat, axes = (0,1) )
        dRdy = jnp.fft.irfft2( self.two_i_pi * self.ky_les[...,None] * R_hat, axes = (0,1) )
        adv_R = jnp.fft.rfft2( u[...,None] * dRdx + v[...,None] * dRdy, axes = (0,1) )
        terms_R = - adv_R
        # Production for R
        prod_R = jnp.stack([0.5*R_ij[...,1] * ( dudy - dvdx ),
                            0.5*R_ij[...,0] * ( dvdx - dudy ),
                            ], axis = -1)
        terms_R += jnp.fft.rfft2( 2*prod_R, axes = (0,1) )
        # Uncomment for Brandons method
        #terms_R += - jnp.fft.rfft2( prod_R, axes = (0,1) ) \
        #    + 2 * self.filter_fn( jnp.fft.rfft2( prod_R, axes = (0,1) ) )
 
        # Compute ngm R
        dLudx = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.kx_dns * self.laplace_dns * \
                                                         self.filter_fn( u_hat ), self.grid_les.shape ) )
        dLudy = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.ky_dns * self.laplace_dns * \
                                                         self.filter_fn( u_hat ), self.grid_les.shape ) )
        dLvdx = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.kx_dns * self.laplace_dns * \
                                                        self.filter_fn( v_hat ), self.grid_les.shape ) )
        dLvdy = jnp.fft.irfft2( spectral_utils.down_res( self.two_i_pi * self.ky_dns * self.laplace_dns * \
                                                        self.filter_fn( v_hat ), self.grid_les.shape ) )

        trNGMK = dLudx**2 + dLudy**2 + dLvdx**2 + dLvdy**2
        trNGMK *= self.Delta**6/6912
        
        forcing = jnp.stack([ trNGMK * dudx ,
                              0.5 * trNGMK * ( dudy + dvdx ),
                            ], axis = -1)

        # Compute invariants
        I1 = 4*dudx**2 + (dvdx + dudy)**2
        I2 = ( dudy - dvdx )**2
        I_scale = jnp.sqrt( I1[...,None] + I2[...,None] )

        # Build linear terms
        linear_term = jnp.fft.rfft2( I_scale * ( forcing - 0.1 * R_ij ), axes = (0,1) )
        terms_R += linear_term# * (- jnp.sign(linear_term))
        self.nu_R = I1.mean() * self.Delta**6/12**3/12

        if self.smooth:
            terms_w *= self.filter_dns
            terms_R *= self.filter_les#[...,None]
 
        return terms_w, terms_R
 
    def implicit_terms(self, state):
        # Unpack the state
        w_hat, R_hat = state
        return self.linear_term_dns * w_hat, \
              (self.nu_R*self.linear_R**3 + self.viscosity*self.linear_R) * R_hat
 
    def implicit_solve(self, y_hat, dt):
        """
        Solves 1 - dt G x = y, for x
        """
        w_hat, R_hat = y_hat
        return 1 / (1 - dt * self.linear_term_dns) * w_hat, \
              1 / (1 - dt * (self.nu_R*self.linear_R**3  + self.viscosity * self.linear_R)) * R_hat
    
    def init_R( self, u_hat, grid_aux: grids.Grid):
        """
        Calculates R from u_les
        """
        kx ,ky = grid_aux.rfft_mesh()
        o_filt_fn = spectral_utils.get_filter_fn( self.Delta, kx, ky )
        # Filter field
        up = jnp.fft.irfft2( u_hat - o_filt_fn( u_hat ), axes = (0,1) )
        upup_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]**2         , axes =(0,1) ) )
        vpvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,1]**2         , axes =(0,1) ) )
        upvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]*up[...,1], axes =(0,1) ) )

        bar_up = jnp.fft.irfft2( o_filt_fn( u_hat[...,0] - o_filt_fn( u_hat[...,0] )), axes = (0,1) )
        bar_vp = jnp.fft.irfft2( o_filt_fn( u_hat[...,1] - o_filt_fn( u_hat[...,1] ) ), axes = (0,1) )


        # Not symmetized R
        Rxx = upup_bar - jnp.fft.rfft2( bar_up**2, axes = (0,1) )
        Ryy = vpvp_bar - jnp.fft.rfft2( bar_vp**2, axes = (0,1) )
        Rxy = upvp_bar - jnp.fft.rfft2( bar_up*bar_vp, axes = (0,1) )
        
        # Symmetrize traceless R
        Rij = jnp.stack([ 0.5 * Rxx - 0.5 * Ryy,
                          Rxy,
                         ],axis = -1)
                         
        Rij = spectral_utils.down_res(Rij, self.grid_les.shape)

        if self.smooth:
            Rij *= self.filter_les

        return Rij
    

@dataclasses.dataclass
class LES_w_TR(time_stepping.ImplicitExplicitODE):
    viscosity: float
    grid: grids.Grid
    Delta: float
    drag: float = 0.
    smooth: bool = True
    k_filter: float = None
    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        self.two_i_pi = 2j * jnp.pi
        self.laplace = self.two_i_pi**2 * (self.kx**2 + self.ky**2)
        self.inv_laplace = jnp.where(self.laplace == 0, 1, 1 / self.laplace)
 
        if self.k_filter is None:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        else:
          self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid,self.k_filter)
 
        self.linear_term = self.viscosity * self.laplace - self.drag
        self.w_to_v = spectral_utils.vorticity_to_velocity(self.grid)
        self.filter_fn = spectral_utils.get_filter_fn( self.Delta*1.5, self.kx, self.ky )
        self.U_size = self.grid.shape + (2,)
        self.sigma_sq = self.Delta**2/12

    def explicit_terms(self, state):
        # Unpack the state
        U_les, R_ij = state # already in fourier space
        U_space = jnp.fft.irfft2( U_les, axes = (0,1) )
        R_space = jnp.fft.irfft2( R_ij, axes = (0,1))
 
        if self.smooth:
            U_les *= self.filter_[...,None]
            R_ij *= self.filter_[...,None]
        # Gradient U
        dUdx = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None] * U_les, axes = (0,1) )
        dUdy = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None] * U_les, axes = (0,1) )
        # Compute advection LES
        advection = -( dUdx * U_space[...,0:1] + dUdy * U_space[...,1:2] )
        terms_les = jnp.fft.rfft2( advection, axes = (0,1) )
 
        # Compute tau ngm
        dUdxdx = jnp.fft.irfft2( self.two_i_pi**2 * self.kx[...,None]**2 * U_les, axes = (0,1) )
        dUdydy = jnp.fft.irfft2( self.two_i_pi**2 * self.ky[...,None]**2 * U_les, axes = (0,1) )
        dUdxdy = jnp.fft.irfft2( self.two_i_pi**2 * self.ky[...,None]*self.kx[...,None] * U_les, axes = (0,1) )
        # Array ngmR as tau_xx, tau_yy, tau_xy
        ngm2xx = dUdx[...,0]**2 + dUdy[...,0]**2
        ngm2yy = dUdx[...,1]**2 + dUdy[...,1]**2
        ngm2xy = dUdx[...,0]*dUdx[...,1] + dUdy[...,0]*dUdy[...,1]             
        ngmR = self.sigma_sq * jnp.stack( [ ngm2xx/2-ngm2yy/2, ngm2xy], axis=-1 )
 
        
        # Calculate ngm4
        ngm4xx = (dUdxdx[...,0]**2 + dUdydy[...,0]**2 + 2 * dUdxdy[...,0]**2) 
        ngm4yy = (dUdxdx[...,1]**2 + dUdydy[...,1]**2 + 2 * dUdxdy[...,1]**2)
        ngm4xy = (dUdxdx[...,0] * dUdxdx[...,1] + 2 * dUdxdy[...,0] * dUdxdy[...,1] + dUdydy[...,0] * dUdydy[...,1])
        ngm4 = self.sigma_sq**2/2 * jnp.stack([ ngm4xx/2 - ngm4yy/2, ngm4xy], axis=-1)
        ngmR += ngm4

        ngmR = jnp.fft.rfft2( ngmR, axes = (0,1) ) + R_ij
 
        div_tau = jnp.stack( [ self.two_i_pi * self.kx * ngmR[...,0] + self.two_i_pi * self.ky * ngmR[...,2],\
                               self.two_i_pi * self.kx * ngmR[...,2] - self.two_i_pi * self.ky * ngmR[...,0]],\
                               axis = -1)
        terms_les -= div_tau

        # ----------------------------
        # Compute R evolution equation
        # ----------------------------
        # Advection of R
        dRdx_ij = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None] * R_ij, axes = (0,1) )
        dRdy_ij = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None] * R_ij, axes = (0,1) )
        R_adv = (U_space[...,0:1] * dRdx_ij + U_space[...,1:2] * dRdy_ij)
        rhs_R = - R_adv
        # Production terms of R
        prod_R = jnp.stack([0.5*R_space[...,1] * ( dUdy[...,0] - dUdx[...,1] ),
                            0.5*R_space[...,0] * ( dUdx[...,1] - dUdy[...,0] ),
                            ], axis=-1)
        prod_R = jnp.fft.irfft2( self.filter_fn(jnp.fft.rfft2(prod_R, axes = (0,1))), axes = (0,1) )

        rhs_R += 2*prod_R

        #rhs_R = jnp.fft.irfft2( self.filter_fn(jnp.fft.rfft2(rhs_R, axes = (0,1))), axes = (0,1) )
        #rhs_R -= 2*prod_R
        # Filter production?
        # rhs_R = jnp.fft.irfft2(self.filter_fn(jnp.fft.rfft2(rhs_R,axes=(0,1))),axes=(0,1))

        dLUdx = jnp.fft.irfft2( self.two_i_pi * self.kx[...,None] * self.laplace[...,None] * U_les, axes = (0,1) )
        dLUdy = jnp.fft.irfft2( self.two_i_pi * self.ky[...,None] * self.laplace[...,None] * U_les, axes = (0,1) )
 
 
        
        #F_space = jnp.stack([(dLUdx[...,0]**2 + dLUdy[...,0]**2),
        #                 (dLUdx[...,1]**2 + dLUdy[...,1]**2),
        #                 (dLUdx[...,0]*dLUdx[...,1] + dLUdy[...,0]*dLUdy[...,1])], axis=-1)
 
        #F_space *= self.sigma_sq**3/4
        #F_scale = F_space.mean()
        # Add f forcing
        trNGMK = dLUdx[...,0]**2 + dLUdy[...,0]**2 + dLUdx[...,1]**2 + dLUdy[...,1]**2
        trNGMK *= self.sigma_sq**3/4

        #F_ij
        S_space = jnp.stack([dUdx[...,0],
                             dUdy[...,0]/2+dUdx[...,1]/2,
                            ], axis=-1)
        linear_term = trNGMK[...,None] * S_space
        linear_term = jnp.fft.irfft2( self.filter_fn(jnp.fft.rfft2(linear_term, axes = (0,1))), axes = (0,1) )
        
        rhs_R += linear_term
        
        # Compute invariants
        I1 = 4*dUdx[...,0]**2 + (dUdx[...,1] + dUdy[...,0])**2
        I2 = (dUdy[...,0] - dUdx[...,1])**2
        I = jnp.sqrt(I1[...,None] + I2[...,None])

        linear_drag =  - 0.15 * I * R_space
        rhs_R += linear_drag

        # In fourier space
        rhs_R = jnp.fft.rfft2( rhs_R, axes = (0,1) )

        # Set regularization scales
        nu_eddy = (2*R_space[...,0]**2+2*R_space[...,1]**2)**(1/4)
        
        # nu_eddy = 10*self.Delta**6/6912*jnp.stack([(dLUdx[...,0]**2 + dLUdy[...,0]**2),
        #                  (dLUdy[...,1]**2 + dLUdy[...,1]**2),
        #                  (dLUdx[...,0]*dLUdx[...,1] + dLUdy[...,0]*dLUdy[...,1])], axis=-1)
        #nu_eddy.mean() * self.Delta**7/(2*12**5)
        #nu_eddy = (F_space[...,0]**2+F_space[...,1]**2+2*F_space[...,2]**2)**(1/4)
        self.nu_R = I1.mean() * self.sigma_sq**3/12/12/12
        #0*I1.mean() * self.Delta**2/4#/12**5*12**2*4
        nu_eddy *= self.sigma_sq**4/(self.Delta**3)
        self.nu_eddy = nu_eddy.mean()
        #terms_les += jnp.fft.rfft2(nu_eddy.mean() * LU_space, axes = (0,1) )   

        # Leray projection
        div_rhs = self.two_i_pi * ( self.kx * terms_les[...,0] + self.ky * terms_les[...,1] )
        leray = jnp.stack( [self.two_i_pi * self.kx, self.two_i_pi * self.ky], axis = -1 ) 
        leray *= div_rhs[...,None] * self.inv_laplace[...,None]
        terms_les -= leray

 
        if self.smooth:
            terms_les *= self.filter_[...,None]
            rhs_R *= self.filter_[...,None]

            
        return terms_les, rhs_R
    
    def implicit_terms(self, state):

        # Calculate dissipation and hyperdissipation
        U_les, R_ij = state 
        #R_space = jnp.fft.irfft2( R_ij, axes = (0,1))
        order = 3
        order_R = 3
        return (self.linear_term[...,None] + self.nu_eddy*self.laplace[...,None]**order) * U_les ,\
              ( self.linear_term[...,None] + self.nu_R*self.laplace[...,None]**order_R ) * R_ij
    #1e3*self.nu_eddy*self.laplace[...,None]**order
    def implicit_solve(self, y_hat, dt):


        U_les, R_ij = y_hat 
        order = 3
        order_R = 3
        # + nu_eddy*self.laplace[...,None]**order
        return 1 / (1 - dt * (self.linear_term[...,None] + self.nu_eddy*self.laplace[...,None]**order)) * U_les, \
          1 / (1 - dt * (self.linear_term[...,None] + self.nu_R*self.laplace[...,None]**order_R ) ) * R_ij  
    

    def init_R( self, u_hat, grid_aux: grids.Grid):
        """
        Calculates R from u_les
        """
        kx ,ky = grid_aux.rfft_mesh()
        o_filt_fn = spectral_utils.get_filter_fn( self.Delta, kx, ky )
        # Filter field
        up = jnp.fft.irfft2( u_hat - o_filt_fn( u_hat ), axes = (0,1) )
        upup_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]**2         , axes =(0,1) ) )
        vpvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,1]**2         , axes =(0,1) ) )
        upvp_bar = o_filt_fn( jnp.fft.rfft2( up[...,0]*up[...,1], axes =(0,1) ) )

        bar_up = jnp.fft.irfft2( o_filt_fn( u_hat[...,0] - o_filt_fn( u_hat[...,0] )), axes = (0,1) )
        bar_vp = jnp.fft.irfft2( o_filt_fn( u_hat[...,1] - o_filt_fn( u_hat[...,1] ) ), axes = (0,1) )

        #R = jnp.zeros( u_hat.shape[:2] + (3,), dtype= u_hat.dtype )

        Rxx = upup_bar - jnp.fft.rfft2( bar_up**2, axes = (0,1) )
        Ryy = vpvp_bar - jnp.fft.rfft2( bar_vp**2, axes = (0,1) )
        Rxy = upvp_bar - jnp.fft.rfft2( bar_up*bar_vp, axes = (0,1) )

        Rij = jnp.stack([Rxx/2-Ryy/2,Rxy],axis=-1)


        Rij = spectral_utils.down_res(Rij, self.grid.shape)

        if self.smooth:
            Rij *= self.filter_[...,None]

        return Rij
