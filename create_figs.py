from functions import *

## data preparation
vdem = pd.read_csv("vdem-dm-data.csv")
extract_all_trajectories(vdem, ["pc1", "pc2"], "all_trajectories/", min_length=10)
create_country_table(df=vdem, pathname="country_table.dat")


### Visualisation plot
Autocracies = [
    ("Albania", 1970),
    ("North Korea", 2010),
    ("Germany", 1940),
    ("Vietnam", 2020),
]
Hybrid = [("India", 1946), ("Japan", 1920), ("Zimbabwe", 1980), ("Turkey", 1910)]
Democracies = [
    ("Switzerland", 1910),
    ("Germany", 2020),
    ("USA", 1990),
    ("Sweden", 1950),
]
plot_space_examples(vdem, Auto=Autocracies, Hybrid=Hybrid, Demo=Democracies)
#### country specific plots
countries = list(
    reversed(
        [
            "Switzerland",
            "South Africa",
            "USA",
            "Colombia",
            "Japan",
            "Hungary",
        ]
    )
)
trajectory_with_tamsd_inset(
    path="all_trajectories/",
    countries=countries,
    comp1=1,
    comp2=2,
    df=vdem,
    features=["pc1", "pc2"],
    inset_width=0.3,
    inset_height=0.18,
    inset_left=0.6,
    inset_bottom=0.7,
)
####
###### FPT plot
plot_fpt_years(
    df=vdem,
    bin_size=2.5,
    symbsize=70,
    max_fpt=7.334841628959276,
    mode="smooth",
    smooth_sigma=0.1,
    coverage_sigma=1.0,
)

##### histogram creation
test_step_distr()
create_hist_one_fig(
    "all_trajectories/",
    eps=0,
    inset_width=0.4,
    inset_height=0.4,
    inset_left=0.15,
    inset_bottom=0.19,
)
create_symmetry_hist("all_trajectories/", eps=0)
create_pc_correlation("all_trajectories/", eps=0, absolute=False)
plot_ergodicity_breaking()
### extreme events
create_extreme_stepsize_composite(
    df=vdem,
    features=["pc1", "pc2"],
    selections_list=[
        # Top-left panel
        [("Japan", 1945), ("Spain", 1977), ("Hungary", 1989), ("Indonesia", 1998)],
        # Top-right panel
        [("Germany", 1932), ("Tanzania", 1961), ("Poland", 2015), ("Romania", 1947)],
        # Bottom-left panel
        [("Brazil", 1963), ("Greece", 1966), ("Chile", 1972), ("Thailand", 2013)],
        # Bottom-right panel
        [("Cuba", 1958), ("Portugal", 1974), ("Cambodia", 1974), ("Iran", 1978)],
    ],
    titles=[
        "Democratisation",
        "Party-based change",
        "Military Coup",
        "Revolutionary Change",
    ],
    savepath="figures/four_panel_extreme.pdf",
)
