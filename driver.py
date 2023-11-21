# %%
from nmler import core
import numpy as np
# %%
core.rewriter("data/ko_axons_8_14_23.nml", "outputs/remapped_ko_axons_8_14_23.nml")
# %%
neurons = core.get_neurons("outputs/remapped_ko_axons_8_14_23.nml")
# %%
pcs, means = core.calculate_pc(neurons)
# %%
best_fit_0 = core.best_fit(np.array(pcs.loc[0,:]), np.array(means.loc[0,:]))
# %%
