import os
import sys
import numpy as np

from itertools import combinations
import pickle
from PIL import Image

import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from itertools import combinations
import pickle
from PIL import Image
from flow_matching.utils import visualize_placement, hpwl 

import numpy as np
import os
from operator import itemgetter
import pickle
# Macro dict (macro id -> name, x, y)
BENCHMARK_DIR = 'benchmarks/ispd2005/'

CHIP_BOUNDARY = {
    "adaptec1": (22.0, 22.0, 11589.0, 11589.0),
    "adaptec2": (29.0, 29.0, 15244.0, 15244.0),
    "adaptec3": (36.0, 82.0, 23190.0, 23386.0),
    "adaptec4": (36.0, 58.0, 23226.0, 23386.0),
    "bigblue1": (22.0, 22.0, 11589.0, 11589.0),
    "bigblue2": (46.0, 76.0, 18721.0, 18796.0),
    "bigblue3": (36.0, 100.0, 27693.0, 27868.0),
    "bigblue4": (78.0, 58.0, 32223.0, 32386.0),
}

def read_node_file(fopen, benchmark):
    node_info = {}
    node_info_raw_id_name ={}
    port_info = {}
    fixed_node_info = {}
    node_cnt = 0
    node_range_list = {}
    if benchmark == "bigblue2" or benchmark == "bigblue4" or "ibm" in benchmark:
        node_range_list_path = os.path.join(BENCHMARK_DIR, benchmark, 'node_list_{}_1024.pkl'.format(benchmark))
        assert os.path.exists(node_range_list_path), "node_range_list_path not exists: {}".format(node_range_list_path)
        with open(node_range_list_path, 'rb') as f:
            node_range_list = pickle.load(f)
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith(" "):
            continue
        line = line.strip().split()
        if "ibm" in benchmark:
            node_name = line[0]
            x = int(line[1])
            y = int(line[2])
            if node_name.startswith("p"):
                port_info[node_name] = {"x": x, "y": y}
                continue
            if len(node_range_list) > 0 and node_name not in node_range_list:
                continue
            node_info[node_name] = {"id": node_cnt, "x": x , "y": y, 'fixed': False }
            node_info_raw_id_name[node_cnt] = node_name
            node_cnt += 1
        else:
            if line[-1] != "terminal":
                continue
            node_name = line[0]
            if (benchmark == "bigblue2" or benchmark == "bigblue4") and node_name not in node_range_list:
                continue
            x = int(line[1])
            y = int(line[2])
            node_info[node_name] = {"id": node_cnt, "x": x , "y": y, 'fixed': False}
            if benchmark == "adaptec1" or benchmark == "bigblue1": # adaptec1 bigblue1 adaptec2，这几个位置，是特殊的边缘IO的macro，必须得固定，然后提前放置好
                if (x == 432 and y == 72) or (x == 72 and y == 432):
                    node_info[node_name]['fixed'] = True
                    continue
            elif benchmark == "adaptec2":
                if (x == 576 and y == 192) or (x == 192 and y == 576) or (x == 96 and y == 576) or (x == 576 and y == 96):
                    node_info[node_name]['fixed'] = True
                    continue
            node_info_raw_id_name[node_cnt] = node_name
            node_cnt += 1
    return node_info, node_info_raw_id_name, port_info


def read_net_file(fopen, node_info, port_info):
    net_info = {}
    net_name = None
    net_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith("  ") and \
            not line.startswith("NetDegree"):
            continue
        line = line.strip().split()
        if line[0] == "NetDegree":
            net_name = line[-1]
        else:
            node_name = line[0]
            if node_name in node_info or node_name in port_info:
                if not net_name in net_info:
                    net_info[net_name] = {}
                    net_info[net_name]["nodes"] = {}
                    net_info[net_name]["ports"] = {}
                if node_name in node_info:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["nodes"][node_name] = {}
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
                else:
                    if len(line) >= 3:
                        x_offset = float(line[-2])
                        y_offset = float(line[-1])
                    else:
                        x_offset = 0.0
                        y_offset = 0.0
                    net_info[net_name]["ports"][node_name] = {}
                    net_info[net_name]["ports"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    return net_info



def get_comp_hpwl_dict(node_info, net_info):
    # node_name
    comp_hpwl_dict = {}
    for net_name in net_info:
        max_idx = 0
        for node_name in net_info[net_name]["nodes"]:
            max_idx = max(max_idx, node_info[node_name]["id"])
        if not max_idx in comp_hpwl_dict:
            comp_hpwl_dict[max_idx] = []
        comp_hpwl_dict[max_idx].append(net_name)
    return comp_hpwl_dict


# node_to_net_set[node_name] = {'net_name_1', 'net_name_2', ..., 'net_name_n'}
def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict


def get_port_to_net_dict(port_info, net_info):
    port_to_net_dict = {}
    for port_name in port_info:
        port_to_net_dict[port_name] = set()
    for net_name in net_info:
        for port_name in net_info[net_name]["ports"]:
            port_to_net_dict[port_name].add(net_name)
    return port_to_net_dict


def read_pl_file(fopen, node_info, port_info):
    max_height = 0
    max_width = 0
    for line in fopen.readlines():
        
        line = line.strip()
        if not line.startswith('o') and not line.startswith('p'):
            continue
        line = line.strip().split()
        node_name = line[0]
        if node_name not in node_info and node_name not in port_info:
            continue
        place_x = int(line[1])
        place_y = int(line[2])
        if node_name in node_info:
            max_height = max(max_height, node_info[node_name]["x"] + place_x)
            max_width = max(max_width, node_info[node_name]["y"] + place_y)
        elif node_name in port_info:
            max_height = max(max_height, port_info[node_name]["x"] + place_x)
            max_width = max(max_width, port_info[node_name]["y"] + place_y)

        if node_name in node_info:
            node_info[node_name]["raw_x"] = place_x
            node_info[node_name]["raw_y"] = place_y
        else:
            port_info[node_name]["raw_x"] = place_x
            port_info[node_name]["raw_y"] = place_y
    return max(max_height, max_width), max(max_height, max_width)



def read_scl_file(fopen, benchmark):
    assert "ibm" in benchmark
    for line in fopen.readlines():
        if not "Numsites" in line:
            continue
        line = line.strip().split()
        max_height = int(line[-1])
        break
    return max_height, max_height


def get_node_id_to_name(node_info, node_to_net_dict):
    node_name_and_num = []
    for node_name in node_info:
        node_name_and_num.append((node_name, len(node_to_net_dict[node_name])))
    node_name_and_num = sorted(node_name_and_num, key=itemgetter(1), reverse = True)
    print("node_name_and_num", node_name_and_num)
    node_id_to_name = [node_name for node_name, _ in node_name_and_num]
    for i, node_name in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
    return node_id_to_name


def get_node_id_to_name_topology(node_info, node_to_net_dict, net_info, benchmark):
    node_id_to_name = []
    adjacency = {}
    for net_name in net_info:
        for node_name_1, node_name_2 in list(combinations(net_info[net_name]['nodes'],2)):
            if node_name_1 not in node_info or node_name_2 not in node_info:
                continue
            if node_name_1 not in adjacency:
                adjacency[node_name_1] = set()
            if node_name_2 not in adjacency:
                adjacency[node_name_2] = set()
            adjacency[node_name_1].add(node_name_2)
            adjacency[node_name_2].add(node_name_1)

    visited_node = set()

    node_net_num = {}
    print("[get_node_id_to_name]: node_info len", len(node_info))
    for node_name in node_info:
        node_net_num[node_name] = len(node_to_net_dict[node_name])
    
    node_net_num_fea = {}
    node_net_num_max = max(node_net_num.values())
    print("[get_node_id_to_name]: node_net_num_max", node_net_num_max)
    for node_name in node_info:
        node_net_num_fea[node_name] = node_net_num[node_name]/node_net_num_max
    
    node_area_fea = {}
    node_area_max_node = max(node_info, key = lambda x : node_info[x]['x'] * node_info[x]['y'])
    node_area_max = node_info[node_area_max_node]['x'] * node_info[node_area_max_node]['y']
    print("node_area_max = {}".format(node_area_max))
    for node_name in node_info:
        node_area_fea[node_name] = node_info[node_name]['x'] * node_info[node_name]['y'] / node_area_max
    
    if "V" in node_info:
        add_node = "V"
        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node]))
        node_net_num.pop(add_node)
    
    add_node = max(node_net_num, key = lambda v: node_net_num[v])
    visited_node.add(add_node)
    node_id_to_name.append((add_node, node_net_num[add_node]))
    node_net_num.pop(add_node)
    while len(node_id_to_name) < len(node_info):
        candidates = {}
        for node_name in visited_node:
            if node_name not in adjacency:
                continue
            for node_name_2 in adjacency[node_name]:
                if node_name_2 in visited_node:
                    continue
                if node_name_2 not in candidates:
                    candidates[node_name_2] = 0
                candidates[node_name_2] += 1
        # for remove all uncertain macros
        if True:
            for node_name in node_info:
                if node_name not in candidates and node_name not in visited_node:
                    candidates[node_name] = 0
        if len(candidates) > 0:
            if benchmark == "bigblue3":
                add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*100000 +\
                    node_info[v]['x']*node_info[v]['y'] * 1 + int(v[1:])*1e-6)
            else:
                add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*1000 +\
                    node_info[v]['x']*node_info[v]['y'] * 1 + int(v[1:])*1e-8)
        else:
            if benchmark == "bigblue3" or "ibm" in benchmark:
                add_node = max(node_net_num, key = lambda v: node_net_num[v]*100000 + node_info[v]['x']*node_info[v]['y']*1+ int(v[1:])*1e-8)
            else:
                add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1+ int(v[1:])*1e-8)

        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node]))
        node_net_num.pop(add_node)
    for i, (node_name, _) in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
    print("[get_node_id_to_name]: node_id_to_name", node_id_to_name)
    node_id_to_name_res = [x for x, _ in node_id_to_name]
    return node_id_to_name_res


def get_pin_cnt(net_info):
    pin_cnt = 0
    for net_name in net_info:
        pin_cnt += len(net_info[net_name]["nodes"])
    return pin_cnt


def get_total_area(node_info):
    area = 0
    for node_name in node_info:
        area += node_info[node_name]["x"] * node_info[node_name]["y"]
    return area

def divide_node(node_info):
    new_node_info = {}
    fixed_node_info = {}
    for node_name in node_info:
        if not node_info[node_name]['fixed']:
            new_node_info[node_name] = node_info[node_name]
        else:
            fixed_node_info[node_name] = node_info[node_name]
    return new_node_info, fixed_node_info




class PlaceDB():

    def __init__(self, benchmark_dir = '', benchmark = 'adaptec1'):

        self.benchmark = benchmark

        benchmark_path = os.path.join(benchmark_dir, benchmark) # 需要加进去
        assert os.path.exists(benchmark_path)
        node_file_path = os.path.join(benchmark_path, benchmark + ".nodes")
        with open(node_file_path, "r") as node_file:
            self.node_info, self.node_info_raw_id_name, self.port_info = \
            read_node_file(node_file, benchmark)

        self.original_node_info = self.node_info.copy() # full node info (include fixed and movable)

        net_file_path = os.path.join(benchmark_path, benchmark + ".nets")
        with open(net_file_path, "r") as net_file:
            self.net_info = read_net_file(net_file, self.node_info, self.port_info)
            self.net_cnt = len(self.net_info)

        pl_file_path = os.path.join(benchmark_path, benchmark + ".pl")
        with open(pl_file_path, "r") as pl_file:
            if benchmark == "adaptec1" or benchmark == "bigblue1":
                read_pl_file(pl_file, self.node_info, self.port_info)
            elif benchmark == "adaptec2":
                read_pl_file(pl_file, self.node_info, self.port_info)
            else:
                read_pl_file(pl_file, self.node_info, self.port_info)

        if not "ibm" in benchmark:
            self.port_to_net_dict = {}
            if benchmark in CHIP_BOUNDARY:
                llx, lly, urx, ury = CHIP_BOUNDARY[benchmark]
                self.max_width = urx - llx
                self.max_height = ury - lly
                self.chip_llx = llx
                self.chip_lly = lly
            else:
                self.chip_llx = 0.0
                self.chip_lly = 0.0
        else:
            self.port_to_net_dict = get_port_to_net_dict(self.port_info, self.net_info)
            scl_file_path = os.path.join(benchmark_path, benchmark + ".scl")
            with open(scl_file_path, "r") as scl_file:
                self.max_height, self.max_width = read_scl_file(scl_file, benchmark)

        print(f"[PlaceDB]: max_height = {self.max_height}, max_width = {self.max_width}")
        
        print(f"[PlaceDB]: CHIP BOUNDARY = {CHIP_BOUNDARY[self.benchmark]}")

        self.node_to_net_dict = get_node_to_net_dict(self.node_info, self.net_info)
        if benchmark in ["adaptec1", 'adaptec2','bigblue1'] :
            self.node_info, self.fixed_node_info = divide_node(self.node_info)
        else:
            self.fixed_node_info = {}
        self.node_id_to_name = get_node_id_to_name_topology(self.node_info, self.node_to_net_dict, self.net_info, self.benchmark)
        self.node_name_to_id =  dict((t, i) for i, t in enumerate(self.node_id_to_name))
        self.node_cnt = len(self.node_info)
        self.debug_str()
    
    def debug_str(self):
        print("node_cnt = {}".format(len(self.node_info)))
        print("fixed_node_cnt = {}".format(len(self.fixed_node_info)))
        print("net_cnt = {}".format(len(self.net_info)))
        print("max_height = {}".format(self.max_height))
        print("max_width = {}".format(self.max_width))
        print("pin_cnt = {}".format(get_pin_cnt(self.net_info)))
        print("port_cnt = {}".format(len(self.port_info)))
        print("area ratio = {}".format(get_total_area(self.node_info)/(self.max_height*self.max_height)))




class ExtendedPlaceDB(PlaceDB):
    def __init__(self, args, benchmark_dir, benchmark):
        super().__init__(benchmark_dir=benchmark_dir, benchmark=benchmark)
        self.args = args

        if benchmark in CHIP_BOUNDARY:
            llx, lly, urx, ury = CHIP_BOUNDARY[benchmark]
            self.max_width = urx - llx
            self.max_height = ury - lly
            self.chip_llx = llx
            self.chip_lly = lly
        else:
            self.chip_llx = 0.0
            self.chip_lly = 0.0

        self.all_node_info = {**self.node_info, **self.fixed_node_info}
        self.all_node_names = list(self.all_node_info.keys())
        self.all_node_cnt = len(self.all_node_names)

        movable_names = [n for n in self.node_id_to_name if n in self.node_info]
        fixed_names = [n for n in self.all_node_names if n in self.fixed_node_info]
        self.node_order = movable_names + fixed_names
        self.node_name_to_id = {name: i for i, name in enumerate(self.node_order)}

        self.x, self.cond = self._build_pyg_data()

    def _build_pyg_data(self):
        V = self.all_node_cnt
        name_to_id = self.node_name_to_id

        # === 1. Sizes: Normalize [0, 2] (chip -> [-1,1]) ===
        sizes = []
        for name in self.node_order:
            node = self.all_node_info[name]
            sx = 2 * node['x'] / self.max_width
            sy = 2 * node['y'] / self.max_height
            sizes.append([sx, sy])
        cond_x = torch.tensor(sizes, dtype=torch.float32)

        # === 2. is_ports:  IO macro ===
        is_ports = torch.zeros((V, 1), dtype=torch.float32)
        for name in self.fixed_node_info:
            idx = name_to_id[name]
            is_ports[idx] = 1.0

        # === 3. Initial positions: Normalize center to [-1, 1] ===
        positions = []
        for name in self.node_order:
            node = self.all_node_info[name]
            if "raw_x" not in node or "raw_y" not in node:
                raise ValueError(f"Node {name} is missing placement information (raw_x/raw_y). Value: {node}")
            raw_x = node["raw_x"]
            raw_y = node["raw_y"]

            abs_x = raw_x + self.chip_llx
            abs_y = raw_y + self.chip_lly
            center_x = abs_x + node['x'] / 2
            center_y = abs_y + node['y'] / 2
            norm_x = 2 * center_x / self.max_width - 1
            norm_y = 2 * center_y / self.max_height - 1
            positions.append([norm_x, norm_y])
        x = torch.tensor(positions, dtype=torch.float32)

        edge_list = []
        attr_list = []
        for net_name in self.net_info:
            macro_to_off = {}
            for macro in self.net_info[net_name]["nodes"]:
                if macro not in self.all_node_info:
                    continue
                off_x = self.net_info[net_name]["nodes"][macro].get("x_offset", 0)
                off_y = self.net_info[net_name]["nodes"][macro].get("y_offset", 0)
                norm_off_x = 2 * off_x / self.max_width
                norm_off_y = 2 * off_y / self.max_height
                macro_to_off[macro] = (norm_off_x, norm_off_y)

            connected_macros = [m for m in macro_to_off.keys() if m in name_to_id]
            if len(connected_macros) < 2:
                continue

            for a, b in combinations(connected_macros, 2):
                off_a = macro_to_off[a]
                off_b = macro_to_off[b]
                i = name_to_id[a]
                j = name_to_id[b]
                edge_list.append([i, j])
                attr_list.append([off_a[0], off_a[1], off_b[0], off_b[1]])
                edge_list.append([j, i])
                attr_list.append([off_b[0], off_b[1], off_a[0], off_a[1]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(attr_list, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)

        # === 5. Construct Data ===
        cond = Data(
            x=cond_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_ports=is_ports  # 1 = 不可移动
        )
        cond.chip_size = np.array([0.0, 0.0, 2.0, 2.0])
        cond.benchmark_name = self.benchmark
        cond.max_width = self.max_width
        cond.max_height = self.max_height
        cond.chip_llx = self.chip_llx
        cond.chip_lly = self.chip_lly
        cond.movable_mask = (is_ports.squeeze() == 0)  # 用于训练 mask
        # name to index mapping
        cond.name_index_mapping = self.node_name_to_id
        return x, cond

    def get_macro_pos(self, in_grid_space=False):
        macro_pos = {}
        grid_size_x = self.max_width / self.args.grid
        grid_size_y = self.max_height / self.args.grid
        for name in self.node_order:
            node = self.all_node_info[name]
            raw_x = node.get("raw_x", 0)
            raw_y = node.get("raw_y", 0)
            size_x = node['x']
            size_y = node['y']
            if in_grid_space:
                gx = int((raw_x + self.chip_llx) / grid_size_x)
                gy = int((raw_y + self.chip_lly) / grid_size_y)
                gw = int(np.ceil(size_x / grid_size_x))
                gh = int(np.ceil(size_y / grid_size_y))
                macro_pos[name] = (gx, gy, gw, gh)
            else:
                macro_pos[name] = (raw_x, raw_y, size_x, size_y)
        return macro_pos
    

def convert_to_ispd2005(position, cond, benchmark_name = None, output_path: str = 'placements'):
    os.makedirs(output_path, exist_ok=True)

    if benchmark_name is None:
        benchmark_name = cond.benchmark_name
    
    placement_file = os.path.join(output_path, f"{benchmark_name}.pl")

    benchmark_pl_path = os.path.join(BENCHMARK_DIR, benchmark_name, benchmark_name + ".pl")
    assert os.path.exists(benchmark_pl_path), f"benchmark_pl_path not exists: {benchmark_pl_path}"
    with open(benchmark_pl_path, 'r') as f:
        lines = f.readlines()


    if hasattr(cond, 'max_width'):
        max_width = cond.max_width
        max_height = cond.max_height
        chip_llx = cond.chip_llx if hasattr(cond, 'chip_llx') else CHIP_BOUNDARY[benchmark_name][0]
        chip_lly = cond.chip_lly if hasattr(cond, 'chip_lly') else CHIP_BOUNDARY[benchmark_name][1]
    else:
        llx, lly, urx, ury = CHIP_BOUNDARY[benchmark_name]
        max_width = urx - llx
        max_height = ury - lly
        chip_llx = llx
        chip_lly = lly
    
    # check  attribute in cond DEBUG
    
    
    with open(placement_file, 'w') as out_f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not line.startswith('o') and not line.startswith('p'):
                out_f.write(line)
                continue
            
            parts = line.split()
            node_name = parts[0]
            orig_x = float(parts[1])
            orig_y = float(parts[2])
            # Check for flags
            flags = ' '.join(parts[3:]) if len(parts) > 3 else ''
            has_fixed = '/FIXED' in flags

            if node_name not in cond.name_index_mapping:
                # Not in cond: keep position, ensure no /FIXED
                new_flags = ': N' if has_fixed else flags
                out_f.write(f"{node_name}\t{int(orig_x)}\t{int(orig_y)}\t{new_flags}\n")
            else:
                i = cond.name_index_mapping[node_name]
                if float(cond.is_ports[i]) == 1.0:
                    # Fixed: keep position, add /FIXED if not present
                    new_flags = ': N /FIXED' if not has_fixed else flags
                    out_f.write(f"{node_name}\t{int(orig_x)}\t{int(orig_y)}\t{new_flags}\n")
                else:
                    # Movable: compute new position, : N
                    norm_cx, norm_cy = position[i]
                    cx = (norm_cx.item() + 1) / 2 * max_width + chip_llx
                    cy = (norm_cy.item() + 1) / 2 * max_height + chip_lly
                    size_x = cond.x[i, 0].item() / 2 * max_width
                    size_y = cond.x[i, 1].item() / 2 * max_height
                    ll_x = cx - size_x / 2
                    ll_y = cy - size_y / 2
                    out_f.write(f"{node_name}\t{int(ll_x)}\t{int(ll_y)}\t: N /FIXED\n")

        



