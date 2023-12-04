# %%
from nmler import core
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import navis
# %%
#core.rewriter("data/ss_axons_9_7_23.nml", "outputs/remapped_ss_axons_9_7_23.nml")

# %% Aquiring neuronlists for annotators
n_br = core.get_neurons("outputs/remapped_br_axons_8_15_23.nml")
n_km = core.get_neurons("outputs/remapped_km_axons_8_15_23.nml")
n_ko = core.get_neurons("outputs/remapped_ko_axons_8_14_23.nml")
n_ss = core.get_neurons("outputs/remapped_ss_axons_9_7_23.nml")

# %% acquiring first principal components and datameans
pcs_br, datameans_br = core.calculate_pc(n_br)
pcs_km, datameans_km = core.calculate_pc(n_km)
pcs_ko, datameans_ko = core.calculate_pc(n_ko)
pcs_ss, datameans_ss = core.calculate_pc(n_ss)

# %% function to get 3D and 2D scatter plots for individual annotators
# fig_pc_3D, fig_pc_xy, fig_pc_yz, fig_pc_zx = core.fig_pcs(pcs_br)
# fig_pc_xy.write_image(file="plts/pc_plts/ss/pcs_xy_ss_axons_9_7_23.png", format = "png")
# fig_pc_yz.write_image(file="plts/pc_plts/ss/pcs_yz_ss_axons_9_7_23.png", format = "png")
# fig_pc_zx.write_image(file="plts/pc_plts/ss/pcs_zx_ss_axons_9_7_23.png", format = "png")

# %% function to get combined 3D scatter plot of all annotations
pcs_all, dm_all, fig_pc_all_3d, fig_pc_all_xy, fig_pc_all_yz, fig_pc_all_zx, dm_al, pc_al = core.fig_pcs_all(n_br, n_km, n_ko, n_ss)
# fig_pc_all_xy.write_image(file="plts/pc_plts/all/pcs_xy_al_axons.png", format = "png")
# fig_pc_all_yz.write_image(file="plts/pc_plts/all/pcs_yz_al_axons.png", format = "png")
# fig_pc_all_zx.write_image(file="plts/pc_plts/all/pcs_zx_al_axons.png", format = "png")
# %% function to get all best fit lines for individual
best_fit_all = core.best_fit_all(pc_al, dm_al, 'N/A')
# %% function to get singular best fit line for a first principal component
# best_fit_0 = core.best_fit(np.array(pcs_br.loc[0,:]), np.array(datameans_br.loc[0,:]), "Brian", "0")
# %% function to get plots for all best fit lines (bfl) from individual annotations
fig_bfl_3d, fig_bfl_xy, fig_bfl_yz, fig_bfl_zx = core.fig_best_fit(best_fit_all)
# fg_bfl_xy.write_image(file="plts/best_fit_line_plts/ko/bfl_xy_ko_axons_8_14_23.png", format = "png")
# fig_bfl_yz.write_image(file="plts/best_fit_line_plts/ko/bfl_yz_ko_axons_8_14_23.png", format = "png")
# fig_bfl_zx.write_image(file="plts/best_fit_line_plts/ko/bfl_zx_ko_axons_8_14_23.png", format = "png")

# %% function to get histogram plot for cable lengths from individual annotations
cable_len_histo = core.histo_cable_len(n_br)
# plt.savefig("plts/cable_len_plts/histograms/ss/histo_cable_len_ss_axons_9_7_23.png")

# %% function to get violin plot from individual annotations
cable_len_vio = core.vio_cable_len(n_br)
#plt.savefig("plts/cable_len_plts/violins/ko/vio_cable_len_ko_axons_8_14_23.png")

# %% function to get figure for combined violin plots from all annotations
cable_len_all, cable_len_vio_all = core.vio_cable_len_all(n_br, n_km, n_ko, n_ss)
cable_len_vio_all.write_image(file="plts/cable_len_plts/violins/all/vio_cable_len_all_axons_10_18_23.png", format = "png")

# %% Plot of actual annotations
navis.plot3d(n_ko)
# %%
