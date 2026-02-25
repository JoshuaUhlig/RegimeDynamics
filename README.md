## Data Analysis

1. Create a folder called `all_trajectories/` and a folder called `figures/´
2. Run `create_figs.py`

## Simulation of CTRW

1. Create a folder called `sim_trajectories/`
2. Run `sim.jl` to generate simulation trajectories
3. Analyse the results by calling `create_tamsd_plot()` in `sim_functions.py`

### Robustness Test
For the robustness test, repeat the above steps with the following changes:
- Create `sim_trajectories_lb/` and `sim_trajectories_ub/` instead
- Run `sim_lb.jl` and `sim_ub.jl` instead
- Call `create_tamsd_comparison_plot()` in `sim_functions.py` instead

