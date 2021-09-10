import numpy as np
from DVR_derivatives import Derivatives

# constants and conversion factors
me = 9.10938356e-31
har2wave = 219474.6
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_OH = (m_H*m_O)/(m_H+m_O)
omega_asym = 3815.044564/har2wave


small_grid_points = 100
Roo_grid = np.linspace(3.9, 5.8, small_grid_points)
sp_grid = np.linspace(-1.5, 1.5, small_grid_points)
wvfns = np.load('small_grid_2d_h3o2_no_cutoff.npz')['wvfns']


# test = Derivatives(wvfns[:, 0].reshape((small_grid_points, small_grid_points)), sp_grid, Roo_grid)
# z_ground_no_der = wvfns[:, 0].reshape((small_grid_points, small_grid_points))
# z_ground_dx1 = test.compute_derivative(dx=1)/z_ground_no_der
# z_ground_dy1 = test.compute_derivative(dy=1)/z_ground_no_der
# z_ground_dx2 = test.compute_derivative(dx=2)/z_ground_no_der
# z_ground_dy2 = test.compute_derivative(dy=2)/z_ground_no_der
# z_ground_dx1_dy1 = test.compute_derivative(dx=1, dy=1)/z_ground_no_der
#
# test = Derivatives(wvfns[:, 0].reshape((small_grid_points, small_grid_points)), sp_grid, Roo_grid, fd=True)
# z_ground_no_der = wvfns[:, 0].reshape((small_grid_points, small_grid_points))
# z_ground_dx1_fd = test.compute_derivative(dx=1)/z_ground_no_der
# z_ground_dy1_fd = test.compute_derivative(dy=1)/z_ground_no_der
# z_ground_dx2_fd = test.compute_derivative(dx=2)/z_ground_no_der
# z_ground_dy2_fd = test.compute_derivative(dy=2)/z_ground_no_der
# z_ground_dx1_dy1_fd = test.compute_derivative(dx=1, dy=1)/z_ground_no_der

ground_der = Derivatives(wvfns[:, 0], sp_grid, Roo_grid)
z_ground_no_der = wvfns[:, 0].reshape((small_grid_points, small_grid_points))
z_ground_dx1 = ground_der.compute_derivative(dx=1)/z_ground_no_der
z_ground_dy1 = ground_der.compute_derivative(dy=1)/z_ground_no_der
z_ground_dx2 = ground_der.compute_derivative(dx=2)/z_ground_no_der
z_ground_dy2 = ground_der.compute_derivative(dy=2)/z_ground_no_der
z_ground_dx1_dy1 = ground_der.compute_derivative(dx=1, dy=1)/z_ground_no_der
#
excite_der = Derivatives(wvfns[:, 2], sp_grid, Roo_grid)
z_excite_no_der = wvfns[:, 2].reshape((small_grid_points, small_grid_points))
z_excite_dx1 = excite_der.compute_derivative(dx=1)/z_excite_no_der
z_excite_dy1 = excite_der.compute_derivative(dy=1)/z_excite_no_der
z_excite_dx2 = excite_der.compute_derivative(dx=2)/z_excite_no_der
z_excite_dy2 = excite_der.compute_derivative(dy=2)/z_excite_no_der
z_excite_dx1_dy1 = excite_der.compute_derivative(dx=1, dy=1)/z_excite_no_der

np.save('z_ground_dx1_new_derivative_method', z_ground_dx1)
np.save('z_ground_dx2_new_derivative_method', z_ground_dx2)
np.save('z_ground_dy1_new_derivative_method', z_ground_dy1)
np.save('z_ground_dy2_new_derivative_method', z_ground_dy2)
np.save('z_ground_dx1_dy1_new_derivative_method', z_ground_dx1_dy1)
# #
np.save('z_excite_dx1_new_derivative_method', z_excite_dx1)
np.save('z_excite_dx2_new_derivative_method', z_excite_dx2)
np.save('z_excite_dy1_new_derivative_method', z_excite_dy1)
np.save('z_excite_dy2_new_derivative_method', z_excite_dy2)
np.save('z_excite_dx1_dy1_new_derivative_method', z_excite_dx1_dy1)

z_ground_dx1 = np.load('z_ground_dx1_new_derivative_method.npy')
z_ground_dy1 = np.load('z_ground_dy1_new_derivative_method.npy')
z_ground_dx2 = np.load('z_ground_dx2_new_derivative_method.npy')
z_ground_dy2 = np.load('z_ground_dy2_new_derivative_method.npy')
z_ground_dx1_dy1 = np.load('z_ground_dx1_dy1_new_derivative_method.npy')

z_excite_dx1 = np.load('z_excite_dx1_new_derivative_method.npy')
z_excite_dy1 = np.load('z_excite_dy1_new_derivative_method.npy')
z_excite_dx2 = np.load('z_excite_dx2_new_derivative_method.npy')
z_excite_dy2 = np.load('z_excite_dy2_new_derivative_method.npy')
z_excite_dx1_dy1 = np.load('z_excite_dx1_dy1_new_derivative_method.npy')

from scipy import interpolate

X, Y = np.meshgrid(sp_grid, Roo_grid, indexing='ij')
import matplotlib.pyplot as plt
# fig, axes = plt.subplots(3, 2)
# axes[0, 0].contourf(X, Y, z_ground_no_der)
# axes[0, 1].contourf(X, Y, z_ground_dx1_dy1)
# axes[1, 0].contourf(X, Y, z_ground_dx1)
# axes[1, 1].contourf(X, Y, z_ground_dy1)
# axes[2, 0].contourf(X, Y, z_ground_dx2)
# axes[2, 1].contourf(X, Y, z_ground_dy2)
# fig.colorbar(im00, ax=axes[0, 0])
# fig.colorbar(im01, ax=axes[0, 1])
# fig.colorbar(im10, ax=axes[1, 0])
# fig.colorbar(im11, ax=axes[1, 1])
# fig.colorbar(im20, ax=axes[2, 0])
# fig.colorbar(im21, ax=axes[2, 1])
# plt.show()
# fig, axes = plt.subplots(3, 2)
# im00 = axes[0, 0].contourf(X, Y, z_excite_no_der)
# im01 = axes[0, 1].contourf(X, Y, z_excite_dx1_dy1)
# im10 = axes[1, 0].contourf(X, Y, z_excite_dx1)
# im11 = axes[1, 1].contourf(X, Y, z_excite_dy1)
# im20 = axes[2, 0].contourf(X, Y, z_excite_dx2)
# im21 = axes[2, 1].contourf(X, Y, z_excite_dy2)
# fig.colorbar(im00, ax=axes[0, 0])
# fig.colorbar(im01, ax=axes[0, 1])
# fig.colorbar(im10, ax=axes[1, 0])
# fig.colorbar(im11, ax=axes[1, 1])
# fig.colorbar(im20, ax=axes[2, 0])
# fig.colorbar(im21, ax=axes[2, 1])
# plt.show()

# ground_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_ground_no_der.flatten())

ground_no_der = interpolate.interp2d(sp_grid, Roo_grid, z_ground_no_der.T, kind='cubic')
ground_dx1 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dx1.T, kind='cubic')
ground_dx2 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dx2.T, kind='cubic')
ground_dy1 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dy1.T, kind='cubic')
ground_dy2 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dy2.T, kind='cubic')
ground_dx1_dy1 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dx1_dy1.T, kind='cubic')


def interp(x, y, poiuy):
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = poiuy(x[i], y[i])
    return out


# a = np.unravel_index(np.argmax(z_ground_no_der), z_ground_no_der.shape)
# dx = (sp_grid[-1] - sp_grid[0])/(len(sp_grid)-1)
# xh = np.array([sp_grid[a[0]] - 2*dx, sp_grid[a[0]] - 1*dx, sp_grid[a[0]], sp_grid[a[0]] + 1*dx, sp_grid[a[0]] + 2*dx])
# dx = (Roo_grid[-1] - Roo_grid[0])/(len(Roo_grid)-1)
# roo = np.array([Roo_grid[a[1]] - 2*dx, Roo_grid[a[1]] - 1*dx, Roo_grid[a[1]], Roo_grid[a[1]] + 1*dx, Roo_grid[a[1]] + 2*dx])
#
# z = ground_no_der(xh, roo).T
# der = Derivatives(z, xh, roo, fd=True)
# dx1 = der.compute_derivative(dx=1)
# dx1oz = dx1/z
# print(dx1oz[2, 2]/z_ground_dx1[a])

# import matplotlib.pyplot as plt
# x = np.linspace(-1.3, 1.3, 600)
# dx = 1e-3
# dx = 0.0043
# dx = 0.0303
# x = np.array([sp_grid[49] - 2*dx, sp_grid[49] - dx, sp_grid[49], sp_grid[49] + dx, sp_grid[49] +2*dx])
# # x = sp_grid
# Roo = np.linspace(4, 5.2, 600)
# dx = 0.0191919
# Roo = np.array([Roo_grid[40] - 2*dx, Roo_grid[40] - dx, Roo_grid[40], Roo_grid[40] + dx, Roo_grid[40] + 2*dx])
# # Roo = Roo_grid
# z = ground_no_der(x, Roo)
# der = Derivatives(z, x, Roo, fd=True)
# dx1 = der.compute_derivative(dx=1)
# dx1y = dx1/z
# print(np.average(dx1y/ground_dx1(x, Roo)))
# plt.contourf(X, Y, z_ground_dx1)
# X, Y = np.meshgrid(x, Roo)
# fig, axes = plt.subplots(3, 5)
# im1 = axes[0, 0].contourf(X[80:520, 80:520], Y[80:520, 80:520], dx1y[80:520, 80:520])
# im2 = axes[1, 0].contourf(X[80:520, 80:520], Y[80:520, 80:520], ground_dx1(x, Roo)[80:520, 80:520])
# im3 = axes[2, 0].contourf(X[80:520, 80:520], Y[80:520, 80:520], (dx1y/ground_dx1(x, Roo))[80:520, 80:520])
# fig.colorbar(im1, ax=axes[0, 0])
# fig.colorbar(im2, ax=axes[1, 0])
# fig.colorbar(im3, ax=axes[2, 0])
# dx1 = der.compute_derivative(dy=1)
# dx1y = dx1/z
# im1 = axes[0, 1].contourf(X[80:520, 80:520], Y[80:520, 80:520], dx1y[80:520, 80:520])
# im2 = axes[1, 1].contourf(X[80:520, 80:520], Y[80:520, 80:520], ground_dy1(x, Roo)[80:520, 80:520])
# im3 = axes[2, 1].contourf(X[80:520, 80:520], Y[80:520, 80:520], (dx1y/ground_dy1(x, Roo))[80:520, 80:520])
# fig.colorbar(im1, ax=axes[0, 1])
# fig.colorbar(im2, ax=axes[1, 1])
# fig.colorbar(im3, ax=axes[2, 1])
# dx1 = der.compute_derivative(dx=1, dy=1)
# dx1y = dx1/z
# im1 = axes[0, 2].contourf(X[80:520, 80:520], Y[80:520, 80:520], dx1y[80:520, 80:520])
# im2 = axes[1, 2].contourf(X[80:520, 80:520], Y[80:520, 80:520], ground_dx1_dy1(x, Roo)[80:520, 80:520])
# im3 = axes[2, 2].contourf(X[80:520, 80:520], Y[80:520, 80:520], (dx1y/ground_dx1_dy1(x, Roo))[80:520, 80:520])
# fig.colorbar(im1, ax=axes[0, 2])
# fig.colorbar(im2, ax=axes[1, 2])
# fig.colorbar(im3, ax=axes[2, 2])
# dx1 = der.compute_derivative(dx=2)
# dx1y = dx1/z
# im1 = axes[0, 3].contourf(X[80:520, 80:520], Y[80:520, 80:520], dx1y[80:520, 80:520])
# im2 = axes[1, 3].contourf(X[80:520, 80:520], Y[80:520, 80:520], ground_dx2(x, Roo)[80:520, 80:520])
# im3 = axes[2, 3].contourf(X[80:520, 80:520], Y[80:520, 80:520], (dx1y/ground_dx2(x, Roo))[80:520, 80:520])
# fig.colorbar(im1, ax=axes[0, 3])
# fig.colorbar(im2, ax=axes[1, 3])
# fig.colorbar(im3, ax=axes[2, 3])
# dx1 = der.compute_derivative(dy=2)
# dx1y = dx1/z
# im1 = axes[0, 4].contourf(X[80:520, 80:520], Y[80:520, 80:520], dx1y[80:520, 80:520])
# im2 = axes[1, 4].contourf(X[80:520, 80:520], Y[80:520, 80:520], ground_dy2(x, Roo)[80:520, 80:520])
# im3 = axes[2, 4].contourf(X[80:520, 80:520], Y[80:520, 80:520], (dx1y/ground_dy2(x, Roo))[80:520, 80:520])
# fig.colorbar(im1, ax=axes[0, 4])
# fig.colorbar(im2, ax=axes[1, 4])
# fig.colorbar(im3, ax=axes[2, 4])
# plt.tight_layout()
# plt.show()
# y = ground_no_der(x, Roo)
# y = wvfns[:, 0].reshape((600, 600))[224, :]
# y = wvfns[:, 0].reshape((600, 600))[:, 300]
# plt.plot(sp_grid, y, label='wvfn')
# plt.show()
# der = Derivatives(y, Roo_grid)
# ground_no_der = interpolate.interp1d(x, y, kind='cubic')
# interp = interpolate.splrep(x, ground_no_der(x), s=0)
# interp_ground_no_der = interpolate.splev(x, interp, der=0)
# dx1 = interpolate.splev(x, interp, der=1)
# dx1 = der.compute_derivative(dx=1)
# dx2 = der.compute_derivative(dx=2)
# dx2 = interpolate.splev(x, interp, der=2)
# y = ground_dx1(x, Roo)
# # import matplotlib.pyplot as plt
# # plt.plot(x, y, label='chain rule stuff')
# # plt.plot(x, dx1/interp_ground_no_der, label='scipy')
# # plt.legend()
# # plt.show()
# test = ground_no_der.ev(x, Roo)
# ground_no_der = interpolate.interp1d(Roo_grid, y, kind='cubic')
# ground_dx1 = interpolate.interp1d(Roo_grid, dx1/y, kind='cubic')
# y = ground_dx2(x, Roo)
# ground_dx2 = interpolate.interp1d(Roo_grid, dx2/y, kind='cubic')

# x = np.linspace(-0.7, 0.7, 1000)
# plt.plot(sp_grid, dx1/y, label='1st derivative')
# plt.plot(sp_grid, dx2/y, label='2nd derivative')
# plt.plot(x, ground_dx1(x), label='1st derivative interp')
# plt.plot(x, ground_dx2(x), label='2nd derivative interp')
# plt.ylim(0, 200)
# plt.legend()
# plt.show()

def psi_t(coords, excite):
    psi = np.ones((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    if excite == 'sp':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = interp(dists[:, -1], dists[:, -2], ground_no_der)
        # psi[:, 1] = ground_no_der(dists[:, -2])
    return psi


def dpsidasp(coords, excite, dists):
    collect = np.zeros((len(coords), 3))
    mw_h = m_OH * omega_asym
    if excite == 'sp':
        collect[:, 0] = -mw_h*dists[:, 0]
        collect[:, 2] = excite_xh_dx1(dists[:, -1], dists[:, -2])
        collect[:, 1] = excite_xh_dy1(dists[:, -1], dists[:, -2])
    elif excite == 'a':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = ground_dx1(dists[:, -1], dists[:, -2])
        collect[:, 1] = ground_dy1(dists[:, -1], dists[:, -2])
    else:
        collect[:, 0] = -mw_h*dists[:, 0]
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], ground_dx1)
        # collect[:, 2] = ground_dx1(dists[:, -1])
        # collect[:, 1] = ground_dx1(dists[:, -2])
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], ground_dy1)
    return collect


def d2psidasp(coords, excite, dists):
    collect = np.zeros((len(coords), 4))
    mw_h = m_OH * omega_asym
    if excite == 'sp':
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 2] = excite_xh_dx2(dists[:, -1], dists[:, -2])
        collect[:, 1] = excite_xh_dy2(dists[:, -1], dists[:, -2])
        collect[:, 3] = excite_xh_dx1_dy1(dists[:, -1], dists[:, -2])
    elif excite == 'a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 0]**2 - 3)
        collect[:, 2] = ground_dx2(dists[:, -1], dists[:, -2])
        collect[:, 1] = ground_dy2(dists[:, -1], dists[:, -2])
        collect[:, 3] = ground_dx1_dy1(dists[:, -1], dists[:, -2])
    else:
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], ground_dx2)
        # collect[:, 2] = ground_dx2(dists[:, -1])
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], ground_dy2)
        # collect[:, 1] = ground_dx2(dists[:, -2])
        collect[:, 3] = interp(dists[:, -1], dists[:, -2], ground_dx1_dy1)
    return collect


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    mid = (coords[:, 1] + coords[:, 3])/2
    rxh = coords[:, 0] - mid
    rxh_dist = np.linalg.norm(rxh, axis=-1)
    sp = rxh_dist*np.cos(roh_roo_angle(coords, rxh, dis[:, -2], rxh_dist))
    return np.vstack((a_oh, dis[:, 0], dis[:, 1], dis[:, -2], sp)).T


def roh_roo_angle(coords, rxh, roo_dist, rxh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (rxh)/np.broadcast_to(rxh_dist[:, None], (len(rxh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


def daroodx(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    for bond in range(2):
        chain[:, 2*bond + 1, :, 0] = (-1)**bond*1/np.sqrt(2)*(
                (coords[:, 2*bond + 1] - coords[:, 2*bond + 2]) / dists[:, bond, None])
        chain[:, 2*bond + 2, :, 0] = (-1)**bond*1/np.sqrt(2)*(
                (coords[:, 2*bond + 2] - coords[:, 2*bond + 1]) / dists[:, bond, None])
    chain[:, 1, :, 1] = ((coords[:, 1] - coords[:, 3]) / dists[:, -1, None])
    chain[:, 3, :, 1] = ((coords[:, 3] - coords[:, 1]) / dists[:, -1, None])
    return chain


def daroodx2(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    for bond in range(2):
        chain[:, 2*bond + 1, :, 0] = (-1)**bond*1/np.sqrt(2)*(1./dists[:, bond, None]
                                                              - (coords[:, 2*bond + 1]-coords[:, 2*bond + 2])**2
                                                              /dists[:, bond, None]**3)
        chain[:, 2*bond + 2, :, 0] = (-1)**bond*1/np.sqrt(2)*(1./dists[:, bond, None]
                                                              - (coords[:, 2*bond + 1]-coords[:, 2*bond + 2])**2
                                                              /dists[:, bond, None]**3)
    chain[:, 1, :, 1] = (1./dists[:, -1, None] - (coords[:, 1]-coords[:, 3])**2/dists[:, -1, None]**3)
    chain[:, 3, :, 1] = (1./dists[:, -1, None] - (coords[:, 1]-coords[:, 3])**2/dists[:, -1, None]**3)
    return chain


def dspdx(coords):
    chain = np.zeros((len(coords), 5, 3, 4))
    dx = 1e-3  #Bohr
    coeffs = np.array([1/12, -2/3, 2/3, -1/12])/dx
    atoms = [0, 1, 3]  # the only atoms that affect the derivative of sp
    for atom in atoms:
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 2] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 3] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)


def d2spdx2(coords, sp):
    chain = np.zeros((len(coords), 5, 3, 5))
    chain[:, :, :, 2] = np.broadcast_to(sp[..., None, None], (len(coords), 5, 3))
    dx = 1e-3  #Bohr
    coeffs = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])/(dx**2)
    atoms = [0, 1, 3]  # the only atoms that affect the derivative of sp
    for atom in atoms:
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 3] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 4] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] -= 2*dx
    chain[:, [2, 4]] = np.zeros((len(coords), 2, 3, 5))
    return np.dot(chain, coeffs)


def sp_calc_for_fd(coords):
    bonds = [[1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    mid = (coords[:, 1] + coords[:, 3]) / 2
    rxh = coords[:, 0] - mid
    rxh_dist = np.linalg.norm(rxh, axis=1)
    sp = rxh_dist * np.cos(roh_roo_angle(coords, rxh, dis[:, -2], rxh_dist))
    return sp


def dpsidx(coords, excite):
    dists = all_dists(coords)
    droox = daroodx(coords, dists[:, [1, 2, -2]])
    dspx = dspdx(coords)
    dr = np.concatenate((droox, dspx[..., None]), axis=-1)
    collect = dpsidasp(coords, excite, dists)
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def d2psidx2(coords, excite):
    dists = all_dists(coords)
    droox = daroodx(coords, dists[:, [1, 2, -2]])
    dspx = dspdx(coords)
    dr1 = np.concatenate((droox, dspx[..., None]), axis=-1)
    droox2 = daroodx2(coords, dists[:, [1, 2, -2]])
    dspx2 = d2spdx2(coords, dists[:, -1])
    dr2 = np.concatenate((droox2, dspx2[..., None]), axis=-1)
    first_dir = dpsidasp(coords, excite, dists)
    second_dir = d2psidasp(coords, excite, dists)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1 ** 2, second_dir[:, None, 0:3, None]).squeeze()
    part3 = dr1[..., 0] * dr1[..., 1] * np.broadcast_to(second_dir[:, -1, None, None], (len(dr1), 5, 3)).squeeze()
    part4 = np.matmul(np.broadcast_to(dr1[..., 0, None], droox.shape)*dr1[..., [1, 2]],
                      (np.broadcast_to(first_dir[:, 0, None],
                       first_dir[:, [1, 2]].shape)*first_dir[:, [1, 2]])[:, None, :, None]).squeeze()
    return part1 + part2 + 2*part3 + 2*part4


def all_da_psi(coords, excite):
    dx = 1e-3
    psi = np.zeros((len(coords), 3, 5, 3))
    psi[:, 1] = np.broadcast_to(np.prod(psi_t(coords, excite), axis=1)[:, None, None], (len(coords), 5, 3))
    for atom in range(5):
        for xyz in range(3):
            coords[:, atom, xyz] -= dx
            psi[:, 0, atom, xyz] = np.prod(psi_t(coords, excite), axis=1)
            coords[:, atom, xyz] += 2*dx
            psi[:, 2, atom, xyz] = np.prod(psi_t(coords, excite), axis=1)
            coords[:, atom, xyz] -= dx
    return psi


def local_kinetic(psit):
    dtau = 1
    sigmaH = np.sqrt(dtau / m_H)
    sigmaO = np.sqrt(dtau / m_O)
    sigma = np.broadcast_to(np.array([sigmaH, sigmaO, sigmaH, sigmaO, sigmaH])[:, None], (5, 3))
    dx = 1e-3
    d2psidx2 = ((psit[:, 0] - 2. * psit[:, 1] + psit[:, 2]) / dx ** 2) / psit[:, 1]
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return kin, d2psidx2


def drift(psit):
    dx = 1e-3
    der = (psit[:, 2] - psit[:, 0])/(2*dx)/psit[:, 1]
    return der

new_struct = np.array([
    [0.000000000000000, 0.000000000000000, 0.000000000000000],
    [-2.304566686034061, 0.000000000000001, 0.000000000000000],
    [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
    [2.304566686034061, 0.000000000000001, 0.000000000000000],
    [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233]
])
new_struct[:, 0] = new_struct[:, 0] + 2.304566686034061

grid_sp = np.linspace(-1, 1, 1000)
# grid_oo = np.linspace(4, 4.9, 1000)


def shared_prot_grid(coords, sp):
    mid = (coords[:, 3, 0] - coords[:, 1, 0])/2
    coords[:, 0, 0] = mid+sp
    return coords


def oo_grid(coords, Roo):
    equil_roo_roh_x = coords[0, 3, 0] - coords[0, 4, 0]
    coords[:, 3, 0] = Roo
    coords[:, 4, 0] = Roo - equil_roo_roh_x
    return coords


coords = shared_prot_grid(np.array([new_struct]*1000), grid_sp)
# coords = oo_grid(np.array([new_struct]*1000), grid_oo)
# coords = shared_prot_grid(coords, np.zeros(1000))

def new_local_kinetic(coords, excite):
    dtau = 1
    sigmaH = np.sqrt(dtau / m_H)
    sigmaO = np.sqrt(dtau / m_O)
    sigma = np.broadcast_to(np.array([sigmaH, sigmaO, sigmaH, sigmaO, sigmaH])[:, None], (5, 3))
    der = d2psidx2(coords, excite)
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * der, axis=1), axis=1)
    return kin

from ProtWaterPES import *
import multiprocessing as mp
class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot


def pot(coordz):
    coords = np.array_split(coordz, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    V = np.concatenate(V)
    return V


pool = mp.Pool(mp.cpu_count()-1)

v = pot(coords)

fd_psi = all_da_psi(coords, None)
fd_kin, fd_second_der = local_kinetic(fd_psi)
second_der = d2psidx2(coords, None)
kin = new_local_kinetic(coords, None)
fd_first = drift(fd_psi)
grid = grid_sp
first = dpsidx(coords, None)
import matplotlib.pyplot as plt
plt.plot(grid, fd_first[:, 0, 0], label='fd H1 x')
plt.plot(grid, first[:, 0, 0], label='H1 x')
plt.plot(grid, fd_first[:, 0, 1], label='fd H1 y')
plt.plot(grid, first[:, 0, 1], label='H1 y')
plt.plot(grid, fd_first[:, 0, 2], label='fd H1 z')
plt.plot(grid, first[:, 0, 2], label='H1 z')
plt.legend()
plt.show()
plt.plot(grid, fd_first[:, 1, 0], label='fd O1 x')
plt.plot(grid, first[:, 1, 0], label='O1 x')
plt.plot(grid, fd_first[:, 1, 1], label='fd O1 y')
plt.plot(grid, first[:, 1, 1], label='O1 y')
plt.plot(grid, fd_first[:, 1, 2], label='fd O1 z')
plt.plot(grid, first[:, 1, 2], label='O1 z')
plt.legend()
plt.show()
plt.plot(grid, fd_first[:, 3, 0], label='fd O2 x')
plt.plot(grid, first[:, 3, 0], label='O2 x')
plt.plot(grid, fd_first[:, 3, 1], label='fd O2 y')
plt.plot(grid, first[:, 3, 1], label='O2 y')
plt.plot(grid, fd_first[:, 3, 2], label='fd O2 z')
plt.plot(grid, first[:, 3, 2], label='O2 z')
plt.legend()
plt.show()

plt.plot(grid, v*har2wave, label='pot')
plt.plot(grid, kin*har2wave, label='chain rule')
plt.plot(grid, (kin + v)*har2wave, label='chain rule')
plt.plot(grid, fd_kin*har2wave, label='finite diff')
plt.plot(grid, (fd_kin + v)*har2wave, label='finite diff')
plt.legend()
plt.ylim(0, 4500)
plt.xlabel('XH (Bohr)')
plt.ylabel(r'Energy (cm$^{-1}$)')
plt.savefig('comparison_of_finite_diff_vs_chain_rule')
plt.show()
