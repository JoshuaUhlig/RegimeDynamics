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

### country specific plots
countries = list(reversed(["Switzerland", "USA", "Colombia", "Japan", "Hungary"]))
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

### FPT plot
plot_fpt_years(df=vdem, bin_size=2.5, symbsize=90, max_fpt=7.334841628959276)

### histogram creation
test_step_distr()
create_hist_one_fig(
    "all_trajectories/",
    eps=0,
    inset_width=0.4,
    inset_height=0.4,
    inset_left=0.15,
    inset_bottom=0.19,
)

## extreme events
create_extreme_stepsize_composite(
    df=vdem,
    features=["pc1", "pc2"],
    selections_list=[
        # Top-left panel
        [("Japan", 1945), ("Spain", 1977), ("Hungary", 1989), ("Indonesia", 1998)],
        # Top-right panel
        [
            ("Germany", 1932),
            ("Philippines", 1971),
            ("Chile", 1972),
            ("Poland", 2015),
        ],
        # Bottom-left panel
        [("Brazil", 1963), ("Greece", 1966), ("Turkey", 1979), ("Thailand", 2013)],
        # Bottom-right panel
        [("Cuba", 1958), ("Portugal", 1974), ("Cambodia", 1974), ("Iran", 1978)],
    ],
    titles=[
        "Democratisation",
        "Autocratisation",
        "Military Coup",
        "Revolutionary Change",
    ],
    savepath="figures/four_panel_extreme.pdf",
)
