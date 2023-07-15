
import warnings
from utils import generate_tsp_data,tab_printer,parameter_parser,generate_tsp_file
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    node_coords = generate_tsp_data(args.batch_size,args.nodes_num,args.low,args.high,args.random_mode)
    generate_tsp_file(node_coords,args.tsp_path)


        



