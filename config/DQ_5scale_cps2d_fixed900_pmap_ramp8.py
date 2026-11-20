_base_ = ['DQ_5scale_cps2d.py']

num_queries = 900
set_cost_pmap = 0.05
cost_pmap_warmup_epochs = 0
cost_pmap_ramp_epochs = 8

cgfe_spatial_density_enable = True
cgfe_spatial_density_alpha_warmup_epochs = 0
cgfe_spatial_density_alpha_ramp_epochs = 8
