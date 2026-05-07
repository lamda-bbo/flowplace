import torch as th
import os
import sys
import yaml
from types import SimpleNamespace
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DREAMPLACE_PARENT_DIR = os.path.join(ROOT_DIR, "DREAMPlace", "install")
DREAMPLACE_DIR = os.path.join(DREAMPLACE_PARENT_DIR, "dreamplace")
sys.path.extend(
    [ROOT_DIR, DREAMPLACE_PARENT_DIR, DREAMPLACE_DIR]
)

from problem_instance import ProblemInstance

CONFIG_DIR = os.path.join(ROOT_DIR, "config")


def process_args():
    # Command line config
    params = [arg.lstrip("--") for arg in sys.argv if arg.startswith("--")]

    cmd_config_dict = {}
    for arg in params:
        key, value = arg.split('=')
        try:
            cmd_config_dict[key] = eval(value)
        except:
            cmd_config_dict[key] = value

        if key in ["benchmark"]:
            benchmark = value

    # Default config
    config_path = os.path.join(CONFIG_DIR, f"{cmd_config_dict['config']}.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in cmd_config_dict.items():
        config_dict[key] = value

    # Debug mode
    if 'debug' not in config_dict:
        config_dict['debug'] = False

    args = SimpleNamespace(**config_dict)
    args.design_name = benchmark
    print(f"benchmark:\t{args.design_name}")

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if th.cuda.is_available() and args.use_cuda:
        args.device = 'cuda'
    else:
        args.use_cuda = False
        args.device = 'cpu'
    print(f'using device: {args.device}')

    # Set unique token
    unique_token = "seed_{}_{}".format(args.seed, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    assert args.grid % 32 == 0, 'grid should be a multiple of 32'

    return args, config_path

if __name__ == "__main__":
    args, config_path = process_args()
    seed = args.seed
    macro_pos = th.load(args.macro_pos_path)
    result_path = args.result_path
    plot_path = args.plot_path
    problem =ProblemInstance(args, args.benchmark)
    gp_hpwl, tns, wns = problem.evaluate(macro_pos) # 
    problem.save_placement(result_path)
    problem.plot(gp_hpwl, plot_path)