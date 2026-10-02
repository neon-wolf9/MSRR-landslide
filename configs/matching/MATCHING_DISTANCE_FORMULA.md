# Frozen matching distance

For active field `f`, `z_f(x)=[T_f(x)-median_f]/IQR_f`, with parameters fitted jointly to the event positives and candidate pool. `T_f` is identity, `log(1+x)`, or an already-frozen log transform as recorded in `MATCHING_NORMALIZATION_PROTOCOL.csv`.

For each feature group `g`, `D_g(i,j)` is the mean absolute fieldwise difference `mean_f |z_f(i)-z_f(j)|`. Soil distance first averages depths within each of six properties and then averages properties. The composite cost is

`d(i,j)=0.30D_terrain+0.15D_soil+0.15D_landcover+0.15D_hydrology+0.05D_accessibility+0.20D_rainfall`.

When both units are in the frozen all-soil-missing block, the soil term is omitted and the remaining weighted sum is divided by 0.85. Spatial distance and geology compatibility are hard feasibility constraints, not soft cost terms; no extra geology penalty enters the composite cost.
