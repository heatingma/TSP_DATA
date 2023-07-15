# TSP_DATA
Generating, Reading, Writing, and Solving TSP Data Files

example:

```python

from solver.gurobi_solver import gurobi_solver
from tsp_data.utils import generate_opt_tour_file

filename = "tsp_data/examples/tsp_data_calculate/tsp/burma14.tsp"
save_path = "tsp_data/examples/tsp_data_calculate/tour_by_gurobi/burma14.opt.tour"
tours = gurobi_solver(filename)
generate_opt_tour_file(tours,save_path)

```