import numpy as np
import os
import copy
import math
import re
import pickle
from pathlib import Path
import time
import logging
import torch as th

from DREAMPlace.dreamplace.Params import Params
from DREAMPlace.dreamplace.PlaceDB import PlaceDB
from DREAMPlace.dreamplace.NonLinearPlace import NonLinearPlace
import DREAMPlace.dreamplace.Timer as Timer
import DREAMPlace.dreamplace.ops.place_io.place_io as place_io

from itertools import combinations
import pdb
from PIL import Image


def to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)

class ProblemInstance():
    def __init__(self, args, benchmark, init=True):
        self.benchmark = benchmark
        self.args = args
        self.gp_hpwl = None
        self.regularity = None
        self.database = {}
        self.dmp_params = self._load_dmp_params()
        self.dmp_placedb = PlaceDB()
        self.dmp_placedb(self.dmp_params)
        if args.use_timer_for_evaluation:
            self.timer = Timer.Timer()
            self.timer(self.dmp_params, self.dmp_placedb)
            # This must be done to explicitly execute the parser builders.
            # The parsers in OpenTimer are all in lazy mode.
            self.timer.update_timing()
        else:
            self.timer = None
        self.dmp_placer = NonLinearPlace(self.dmp_params, self.dmp_placedb, timer=self.timer)
        self.results = {}

        self.max_width = self.dmp_placedb.xh - self.dmp_placedb.xl
        self.max_height = self.dmp_placedb.yh - self.dmp_placedb.yl
        self.num_movable_nodes = self.dmp_placedb.num_movable_nodes

        self.grid = self.args.grid # grid : 224

        self.ratio_x = self.max_width / self.grid
        self.ratio_y = self.max_height / self.grid

        # if not init, only load the dmp_params and dmp_placedb
        if not init:
            return

        self.macro_names = []
        self.macros = []
        self.macro_x = []
        self.macro_y = []
        self.macro_size_x = []
        self.macro_size_y = []
        # Calculate average node area
        self.port_indices = []     
        total_area = 0

        for node_name in self.dmp_placedb.node_names:
            node = self.dmp_placedb.node_name2id_map[node_name.decode('utf-8')]
            if node < (self.dmp_placedb.num_physical_nodes - self.dmp_placedb.num_terminal_NIs):  # exclude IO ports
                total_area += self.dmp_placedb.node_size_x[node] * self.dmp_placedb.node_size_y[node]
            else:
                self.port_indices.append(node)  # store the port indices
        avg_area = total_area / len(self.dmp_placedb.node_names)
        
        # Identify macros based on area and height criteria
        ratio = 0.0001 if self.benchmark == "superblue10" or self.benchmark == "superblue7" else 0.001
        for node_name in self.dmp_placedb.node_names:
            node = self.dmp_placedb.node_name2id_map[node_name.decode('utf-8')]
            if node < (self.dmp_placedb.num_physical_nodes - self.dmp_placedb.num_terminal_NIs):  # exclude IO ports
                area = self.dmp_placedb.node_size_x[node] * self.dmp_placedb.node_size_y[node]
                height = self.dmp_placedb.node_size_y[node]
                # if area > ratio * total_area:   # do not include too small "macros"
                if area > 10 * avg_area or height > 2 * self.dmp_placedb.row_height:
                    self.macros.append(node)
                    self.macro_names.append(node_name.decode('utf-8'))
                    self.macro_x.append(self.dmp_placedb.node_x[node])
                    self.macro_y.append(self.dmp_placedb.node_y[node])
                    self.macro_size_x.append(self.dmp_placedb.node_size_x[node])
                    self.macro_size_y.append(self.dmp_placedb.node_size_y[node])
        
        self.macro_names = np.array(self.macro_names).astype(np.str_)
        self.n_macro = len(self.macro_names)
        self.node_names = self.dmp_placedb.node_names.astype(np.str_)
        return

    
    def _load_dmp_params(self):
        params = Params()
        if "superblue" in self.benchmark:
            params.load(os.path.join("DREAMPlace/test/iccad2015.ot", f'{self.benchmark}.json'))
        else:
            params.load(os.path.join("DREAMPlace/test/or_cases", f'{self.benchmark}.json'))
        return params
    
        
    def init_dmp_db(self):
        if self.dmp_caller is None:
            assert0
        else:
            return self.dmp_caller.init_db()

    def evaluate(self, macro_pos):
        if len(macro_pos) == 0:
            return np.inf

        self.apply(macro_pos=macro_pos)

        metric = self.dmp_placer(self.dmp_params, self.dmp_placedb)[-1]
        if isinstance(metric, list):
            gp_hpwl = metric[0][0].hpwl.item()
        else:
            gp_hpwl = metric.hpwl.item()

        self.results['placement'] = (self.dmp_placedb.node_x.copy(),
                                    self.dmp_placedb.node_y.copy())
        self.results['figure'] = copy.copy(self.dmp_placer.pos[0].data.clone().cpu().numpy())
        
        # Evaluate timing metrics (TNS and WNS) if timing optimization is enabled
        if self.args.use_timer_for_evaluation:
            tns, wns = self.evaluate_timing()
        else:
            tns, wns = 0, 0
        # tns, wns = 0, 0
       
        return gp_hpwl, tns, wns

    def evaluate_timing(self): # opentimer
        """
        Evaluate timing metrics (TNS and WNS) using the timing operator.
        Returns:
            tuple: (tns, wns) timing metrics
        """ 
        # Get timing operator from the placer's op_collections
        timing_op = self.dmp_placer.op_collections.timing_op
        time_unit = timing_op.timer.time_unit()
        
        # Perform timing analysis on current placement
        # The timing operator takes the current position as input
        timing_op(self.dmp_placer.pos[0].data.clone().cpu())
        timing_op.timer.update_timing()
        
        # Report TNS and WNS
        # Note: OpenTimer considers early,late,rise,fall for tns/wns
        # The following values are normalized by time units
        tns = timing_op.timer.report_tns_elw(split=1) / (time_unit * 1e17)
        wns = timing_op.timer.report_wns(split=1) / (time_unit * 1e15)
        
        return tns, wns

    def apply(self, macro_pos):
        for node_id in macro_pos:
            pos_x, pos_y, _, _ = macro_pos[node_id]
            print(f"Placing macro node {node_id} at position ({pos_x}, {pos_y})")
            pos_x += self.args.halo
            pos_y += self.args.halo
            pos_x = round(pos_x * self.ratio_x + self.ratio_x)
            pos_y = round(pos_y * self.ratio_y + self.ratio_y)
            self.dmp_placedb.node_x[node_id] = pos_x
            self.dmp_placedb.node_y[node_id] = pos_y

        node_x, node_y = self.dmp_placedb.unscale_pl(self.dmp_params.shift_factor, 
                                                     self.dmp_params.scale_factor)
        place_io.PlaceIOFunction.apply(self.dmp_placedb.rawdb, node_x, node_y)

        with th.no_grad():
            self.dmp_placer.pos[0].data.copy_(
                th.from_numpy(self.dmp_placer._initialize_position(self.dmp_params, self.dmp_placedb)).to(self.dmp_placer.device) )
    
    def save_placement(self, path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.dmp_placedb.node_x[:] = self.results['placement'][0].copy()
        self.dmp_placedb.node_y[:] = self.results['placement'][1].copy()
        # unscale locations
        node_x, node_y = self.dmp_placedb.unscale_pl(self.dmp_params.shift_factor, 
                                                     self.dmp_params.scale_factor)
        # update raw database
        place_io.PlaceIOFunction.apply(self.dmp_placedb.rawdb, node_x, node_y)
        self.dmp_placedb.write(
            self.dmp_params,
            path
        )

    def plot(self, hpwl, figure_name):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(figure_name), exist_ok=True)
        pos = self.results['figure']
        self.dmp_placer.plot(
            self.dmp_params,
            None,
            None,
            pos,
            figure_name, 
        )

        img = Image.open(figure_name)
        out = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.close()
        out.save(figure_name)
        
    def set_mp_hpwl(self, mp_hpwl):
        self.mp_hpwl = mp_hpwl
    
    def set_gp_hpwl(self, gp_hpwl):
        self.gp_hpwl = gp_hpwl
    
    def set_regularity(self, regularity):
        self.regularity = regularity

    def get_node_info(self):
        node_info = {}
        node_info_raw_id_name ={}
        for id, (macro_name, size_x, size_y, raw_x, raw_y) in enumerate(zip(self.macro_names, self.macro_size_x, self.macro_size_y, self.macro_x, self.macro_y)):
            node_info[macro_name] = {"id": id, "x": size_x, "y": size_y, "raw_x": raw_x, "raw_y": raw_y}
            node_info_raw_id_name[id] = macro_name
        
        return node_info, node_info_raw_id_name
    
    def get_net_info(self):
        net_info = {}
        for net_id, net_name in enumerate(self.net_names):
            net_info[net_name] = {}
            net_info[net_name]["nodes"] = {}
            net_info[net_name]["ports"] = {}

            pins = self.net2pin_map[net_id]
            nodes = self.pin2node_map[pins]
            offset_x = self.pin_offset_x[pins] - self.node_size_x[nodes]/2
            offset_y = self.pin_offset_y[pins] - self.node_size_y[nodes]/2

            for node, o_x, o_y in zip(nodes, offset_x, offset_y):
                if node in self.macros:
                    net_info[net_name]["nodes"][self.node_names[node]] = {"x_offset": o_x, "y_offset": o_y}

        for net_name in list(net_info.keys()):
            if len(net_info[net_name]["nodes"]) <= 1:
                net_info.pop(net_name)
        
        net_cnt = 0
        for net_name in net_info:
            net_info[net_name]['id'] = net_cnt
            net_cnt += 1
        print("adjust net size = {}".format(len(net_info)))
        return net_info
    

       

    
def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict

def get_node_id_to_name_topology(node_info, node_to_net_dict, net_info, benchmark):
    node_id_to_name = []
    adjacency = {}

    for net_name in net_info:
        for node_name_1, node_name_2 in list(combinations(net_info[net_name]['nodes'],2)):
            if node_name_1 not in adjacency:
                adjacency[node_name_1] = set()
            if node_name_2 not in adjacency:
                adjacency[node_name_2] = set()
            adjacency[node_name_1].add(node_name_2)
            adjacency[node_name_2].add(node_name_1)

    visited_node = set()

    node_net_num = {}
    for node_name in node_info:
        node_net_num[node_name] = len(node_to_net_dict[node_name])

    node_net_num_fea= {}
    node_net_num_max = max(node_net_num.values())
    print("node_net_num_max", node_net_num_max)
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
        for node_name in node_info:
            if node_name not in candidates and node_name not in visited_node:
                candidates[node_name] = 0
        if len(candidates) > 0:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*100000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
                else:
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*1000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
            else:
                add_node = max(candidates, key = lambda v: candidates[v]*30000 + node_net_num[v]*1000 +\
                    node_info[v]['x']*node_info[v]['y']*1 +int(hash(v)%10000)*1e-6)
        else:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*100000 + node_info[v]['x']*node_info[v]['y']*1)
                else:
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)
            else:
                add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)

        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node])) 
        node_net_num.pop(add_node)
    for i, (node_name, _) in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
        
    node_id_to_name_res = [x for x, _ in node_id_to_name]
    
    return node_id_to_name_res

