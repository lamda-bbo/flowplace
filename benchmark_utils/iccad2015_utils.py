import os
from abc import abstractmethod
from copy import deepcopy
import re
import numpy as np
import torch

def convert_to_iccad15(position: torch.Tensor, benchmark_name: str, chip_size: torch.Tensor = torch.Tensor([2.0, 2.0]), output_path: str = 'placements'):
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    args = Args(BENCHMARK_DIR="benchmarks/iccad2015",
            grid_soft_coeff=10,
            grid=224,
            n_macro=512)

    benchmark_list = ["superblue1", "superblue3", "superblue4", "superblue5", 
                    "superblue7", "superblue10", "superblue16", "superblue18"]
    assert benchmark_name in benchmark_list, f"Unsupported benchmark name: {benchmark_name}"
    print(f"Converting placement for {benchmark_name} to ICCAD 2015 DEF format.")
    problem_instance = ProblemInstance(args, benchmark_name)
    if isinstance(position, np.ndarray):
        position = torch.from_numpy(position)
    if position.dim() == 3 and position.size(0) == 1:
        position = position.squeeze(0)
    
    assert position.dim() == 2 and position.size(0) == problem_instance.node_cnt and position.size(1) == 2, \
        f"Position tensor has incorrect shape: {position.shape}, expected ({problem_instance.node_cnt}, 2)"
    # Create macro_pos with grid-like positions adjusted for the shift
    macro_pos = {}
    for i in range(problem_instance.node_cnt):
        name = problem_instance.node_id_to_name[i]
        norm_cx = position[i, 0].item()
        norm_cy = position[i, 1].item()
        # Denormalize using problem_instance's max dimensions, ignoring chip_size for iccad15
        cx = (norm_cx + 1) / 2 * problem_instance.max_width
        cy = (norm_cy + 1) / 2 * problem_instance.max_height
        sx = problem_instance.node_info[name]['x']
        sy = problem_instance.node_info[name]['y']
        ll_x = cx - sx / 2
        ll_y = cy - sy / 2
        # only write ll_x and ll_y, size_x and size_y will be read from database
        #macro_size_x = problem_instance.node_info[name]['x'] / problem_instance.ratio_x
        #macro_size_y = problem_instance.node_info[name]['y'] / problem_instance.ratio_y
        macro_pos[name] = (ll_x, ll_y, 0, 0)

    # Compute limited_x and limited_y
    inv_scale_ratio_x, inv_scale_ratio_y = get_inv_scaling_ratio(problem_instance.database)
    limited_x = round(problem_instance.max_width * inv_scale_ratio_x)
    limited_y = round(problem_instance.max_height * inv_scale_ratio_y)

    # Set output path if None
    if output_path is None:
        output_path = f"placements/{benchmark_name}.def"
    else:
        os.makedirs(output_path, exist_ok=True)
        output_path = f"{output_path}/{benchmark_name}.def"
    
    # Write the DEF file
    write_def_from_tensor(macro_pos=macro_pos, 
              database=problem_instance.database, 
              def_file=output_path, 
              limited_x=limited_x,
              limited_y=limited_y,
              is_dataset=False)
    print(f"Converted placement for {benchmark_name} written to {output_path}")
    return macro_pos

class EntryFormat:
    def __init__(self, ruleList: list):
        self.ruleList = ruleList

    def __call__(self, entry: str):
        for rule in self.ruleList:
            match, output = rule(entry)
            if match:
                return {} if output == None else output
        return {}


class EntryRule:
    @abstractmethod
    def __call__(self, entry: str) -> (bool, dict):
        pass


class AlwaysMatchedRule(EntryRule):
    def __call__(self, entry: str) -> (bool, dict):
        return (True, {})

class SkipRule(AlwaysMatchedRule):
    def __call__(self, entry: str) -> (bool, dict):
        return (
            True,
            {
                "entry" : entry,
            }
        )

class AlwaysDismatchedRule(EntryRule):
    def __call__(self, entry: str) -> (bool, dict):
        return (False, {})


class PrefixRule(EntryRule):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, entry: str) -> (bool, dict):
        return (
            entry.strip().startswith(self.prefix),
            {}
        )


PrefixIgnoreRule = PrefixRule # alias

class PrefixSkipRule(PrefixRule):
    def __call__(self, entry: str) -> (bool, dict):
        return (
            entry.strip().startswith(self.prefix),
            {
                "entry": entry
            }
        )


class RegExRule(EntryRule):
    def __init__(self, regex: str, index_group_pairs: dict):
        self.regex = re.compile(regex)
        self.index_group_pairs = index_group_pairs
    
    def __call__(self, entry: str) -> (bool, dict):
        match = self.regex.search(entry)
        if match == None:
            return (False, {})
        output = {}
        for index, group in self.index_group_pairs.items():
            if group == -1:
                output[index] = entry
            else:
                output[index] = match.group(group)
        return (True, output)


class RuleGroup:
    def __init__(
        self,
        entrance_rule: EntryRule,
        exit_rule: EntryRule,
        ruleList: list,
        dismatch_policy = "exit_rule"
    ):
        self.entrance_rule = entrance_rule
        self.exit_rule = exit_rule
        self.ruleList = ruleList
        self.dismatch_policy = dismatch_policy

    def access(self, entry: str) -> bool:
        return self._enter(entry)

    def _enter(self, entry: str) -> bool:
        return self.entrance_rule(entry)[0]

    def _exit(self, entry: str) -> bool:
        return self.exit_rule(entry)[0]

    def _dismatch(self, entry: str) -> bool:
        def dismatch_policy_exit_rule(entry: str):
            return self._exit(entry)

        def dismatch_policy_exit(entry: str):
            return True 

        def dismatch_policy_stay(entry: str):
            return False

        dismatch_policy_set = {
            "default": dismatch_policy_exit_rule,
            "exit_rule": dismatch_policy_exit_rule,
            "exit": dismatch_policy_exit,
            "stay": dismatch_policy_stay,
        }

        dismatch_policy = dismatch_policy_set.get(
            self.dismatch_policy,
            dismatch_policy_set["default"]
        )

        return dismatch_policy(entry)

    def __call__(self, entry: str):
        for rule in self.ruleList:
            match, output = rule(entry)
            if match:
                return (
                    {} if output == None else output,
                    self._exit(entry)
                )
        
        return (
            {},
            self._dismatch(entry)
        )
        

class EntryFormatWithRuleGroups(EntryFormat):
    def __init__(self, ruleGroups: list):
        self.ruleGroups = ruleGroups
        self._nowGroup = None

    def inGroup(self) -> bool:
        return self._nowGroup != None

    def quitGroup(self):
        self._nowGroup = None

    def __call__(self, entry: str):
        if self._nowGroup == None:
            for ruleGroup in self.ruleGroups:
                if ruleGroup.access(entry):
                    self._nowGroup = ruleGroup
                    break
            else:
                # no available rule group
                return {}
        
        output, exited = self._nowGroup(entry)
        if exited:
            self.quitGroup()
        return output
    
def read_benchmark(database, benchmark, args=None):
    print("read database from benchmark %s" % (benchmark))
    database["benchmark_dir"] = benchmark
    return read_benchmark_from_def(database, os.path.join(database["benchmark_dir"], f'{benchmark}.def'), args)

def read_benchmark_from_def(database, benchmark, args=None):
    node_info = {}
    node_info_raw_id_name = {}

    node_cnt = 0
    port_info = {}
    standard_cell_name = []
    port_to_net_dict = {}


    database["def_file"] = os.path.basename(benchmark)

    design_name = os.path.splitext(database["def_file"])[0]

    database["files"] = {
        "def_file": database["def_file"],
        "lef_file": "%s.lef" % design_name,
        "v_file": "%s.v" % design_name
    }

    database["nodes"] = {}
    database["macros"] = []


    read_lef(
        database,
        os.path.join(
            database["benchmark_dir"],
            database["files"]["lef_file"]
        )
    )
    
    read_def(
        database,
        os.path.join(
            database["benchmark_dir"],
            database["files"]["def_file"]
        )
    )

    # compute area each macro type
    macro_type_area = {}
    for macro_type in database["macro_size"]:
        size_x, size_y = database["macro_size"][macro_type]
        area = size_x * size_y
        macro_type_area[macro_type] = area

    # compute area each cell
    cell_area_dict = {}
    cell_total_area = 0
    for cell in database["nodes"]:
        cell_type = database["nodes"][cell]["node_type"]
        area = macro_type_area[cell_type]
        cell_area_dict[cell] = area
        cell_total_area += area
    
    cell_lst = list(cell_area_dict.keys())
    cell_lst = sorted(cell_lst, key=lambda x: cell_area_dict[x], reverse=True)

    if args is None:
        n_macro = 512
    else:
        n_macro = args.n_macro
    
    macro_lst = cell_lst[:n_macro]
    standard_cell_name = cell_lst[n_macro:]


    max_height = 0
    max_width = 0
    ratio_x, ratio_y = get_scaling_ratio(database)
    for id, macro in enumerate(macro_lst):
        # scaling
        place_x = eval(database["nodes"][macro]['x']) * ratio_x
        place_y = eval(database["nodes"][macro]['y']) * ratio_y

        macro_type = database["nodes"][macro]["node_type"]
        size_x, size_y = database["macro_size"][macro_type]

        max_height = max(max_height, size_y + place_y)
        max_width = max(max_width, size_x + place_x)
        node_info[macro] = {"id": id, "x": size_x, "y": size_y}
        node_info[macro]["raw_x"] = place_x
        node_info[macro]["raw_y"] = place_y
        node_info_raw_id_name[id] = macro


    node_cnt = len(node_info)
    assert node_cnt == n_macro

    v_file = open(os.path.join(database["benchmark_dir"],database["files"]["v_file"]), 'r')
    
    net_info = read_v(v_file, node_info, database)
    net_cnt = len(net_info)

    placedb_info = {
        'node_info' : node_info,
        'node_info_raw_id_name' : node_info_raw_id_name,
        'node_cnt' : node_cnt,
        'port_info' : port_info,
        'net_info' : net_info,
        'net_cnt' : net_cnt,
        'max_height' : max_height,
        'max_width' : max_width,
        'standard_cell_name' : standard_cell_name,
        'port_to_net_dict' : port_to_net_dict,
        'cell_total_area' : cell_total_area,
    }
    return placedb_info
    

def read_v(fopen, node_info, database):
    net_info = {}
    net_cnt = 0
    flag = 0
    for line in fopen.readlines():
        if 'wire' in line:
            line_ls = line.split(" ")
            net_name = line_ls[1].split(";")[0]
            net_info[net_name] = {}
            net_info[net_name]["nodes"] = {}
            net_info[net_name]["ports"] = {}

        if 'Start cells' in line:
            flag = 1
            continue
        if flag == 1:
            if line == '\n':
                break
            pattern = r"\.(\w+)\((.*?)\)"
            matches = re.findall(pattern=pattern, string=line)
            line_ls = line.split(" ")
            node_type = line_ls[0]
            node_name = line_ls[1]

            for pin_net in matches:
                pin = pin_net[0]
                net = pin_net[1]
                if node_name in node_info.keys() and net in net_info.keys():
                    x_offset, y_offset = database["pin_offset"][node_type][pin]
                    net_info[net]["nodes"][node_name] = {
                        "x_offset": x_offset,
                        "y_offset": y_offset,
                    }


    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    return net_info

def read_lef(database, lef_file):
    macro_start_rule = RegExRule(
        r"MACRO\s+(\w+)",
        {
            "macro_name": 1,
        }
    )
    macro_or_pin_end_rule = RegExRule(
        r"END\s+(\w+)",
        {
            "macro_name_or_pin_name": 1,
        }
    )
    macro_size_rule = RegExRule(
        r"SIZE\s(\d+(\.\d+)?) BY (\d+(\.\d+)?)",
        {
            "size_x" : 1,
            "size_y" : 3,
        }
    )
    macro_pin_start_rule = RegExRule(
        r"(PIN)\s+(\w+)",
        {
            "pin_start" : 1, 
            "pin_name" : 2,
        }
    )
    macro_pin_offset_rule = RegExRule(
        r"RECT\s(-?\d+(\.\d+)?)\s(-?\d+(\.\d+)?)\s(-?\d+(\.\d+)?)\s(-?\d+(\.\d+)?)",
        {
            'x1' : 1,
            'y1' : 3,
            'x2' : 5,
            'y2' : 7,
        }
    )

    macro_rule_group = RuleGroup(
        macro_start_rule,
        AlwaysDismatchedRule(),
        [
            macro_start_rule,
            macro_or_pin_end_rule,
            macro_size_rule,
            macro_pin_start_rule,
            macro_pin_offset_rule,
            RegExRule(
                r"CLASS\s+(\w+)\s+;",
                {
                    "class": 1,
                }
            ),
            SkipRule()
        ]
    )

    other_rule_group = RuleGroup(
        AlwaysMatchedRule(),
        AlwaysMatchedRule(),
        [
            SkipRule()
        ]
    )

    lef_ent_format = EntryFormatWithRuleGroups(
        [
            macro_rule_group,
            other_rule_group
        ]
    )

    assert lef_file is not None and os.path.exists(lef_file), lef_file
    database["lef_macros"] = {}
    database["lef_origin"] = [""]
    database["macro_size"] = {}
    database["pin_offset"] = {}
    macro_name = None
    pin_name = None
    pin_flag = False
    with open(lef_file, "r") as f:
        for line in f:
            output = lef_ent_format(line)
            if not lef_ent_format.inGroup():
                database["lef_origin"][-1] += line
                continue

            if macro_name is None:
                macro_name = output.get("macro_name", None)
                if macro_name is not None:
                    database["lef_macros"][macro_name] = line
                    database["lef_origin"].append("")
                    database["pin_offset"][macro_name] = {}
                continue
            
            if "macro_name_or_pin_name" in output.keys():
                if pin_flag:
                    pin_flag = False

                    min_pin_x = np.min(pin_x)
                    max_pin_x = np.max(pin_x)
                    min_pin_y = np.min(pin_y)
                    max_pin_y = np.max(pin_y)
                    pin_name = output["macro_name_or_pin_name"]
                    database["pin_offset"][macro_name][pin_name] = ((min_pin_x + max_pin_x)/2,
                                                                    (min_pin_y + max_pin_y)/2)
                    del pin_x
                    del pin_y
                    pin_flag = False
                else:
                    database["lef_macros"][macro_name] += line
                    if output["macro_name_or_pin_name"] == macro_name:
                        lef_ent_format.quitGroup()
                        macro_name = None
            elif "size_x" in output.keys():
                database["macro_size"][macro_name] = (eval(output['size_x']), eval(output['size_y']))
            elif "class" in output.keys():
                # rule: CLASS (CORE|BLOCK)
                c = output["class"]
                if c == "CORE":
                    database["lef_macros"][macro_name] += line
                else:
                    database["lef_macros"][macro_name] += \
                        line.replace(c, "CORE", 1)
            elif "pin_start" in output.keys():
                pin_flag = True
                pin_x = []
                pin_y = []
            elif "x1" in output.keys():
                if pin_flag:
                    x1, y1 = eval(output["x1"]), eval(output["y1"])
                    x2, y2 = eval(output["x2"]), eval(output["y2"])
                    pin_x.append(x1)
                    pin_x.append(x2)
                    pin_y.append(y1)
                    pin_y.append(y2)
            else:
                # rule: skip
                database["lef_macros"][macro_name] += line

def read_def(database, def_file):
    
    
    component_start_rule = RegExRule(
        r"COMPONENTS\s+(\d+)\s*;",
        {
            "num_comps": 1,
        }
    )
    component_end_rule = RegExRule(
        r"END\s+COMPONENTS",
        {}
    )
    design_rule = RegExRule(
        r"\s*DESIGN\s+([a-z_A-Z]+)\s+([a-z_A-Z]+)\s+(\d+\.?\d+)\s*;\s*\n?",
        {
            "entry" : 0,
            "key" : 1,
            "value" : 3,
        }
    )
    diearea_rule = RegExRule(
        r"DIEAREA\s+\(\s*(\d+)\s*(\d+)\s*\)\s*\(\s*(\d+)\s*(\d+)\s*\)\s*;\s*\n?",
        {
            # "entry" : 0,
            "lower_x" : 1,
            "lower_y" : 2,
            "upper_x" : 3,
            "upper_y" : 4,
        }
    )
    row_rule = RegExRule(
        r"ROW\s+coreROW_(\d+)\s+core\s+(\d+)\s+(\d+)\s+N\s+DO\s+(\d+)\s+BY\s+(\d+)\s+STEP\s+(\d+)\s+(\d+)\s+;",
        {
            "id" : 1,
            "row_x" : 2,
            "row_y" : 3,
            "after_do" : 4,
            "after_by" : 5,
            "step_x" : 6,
            "step_y" : 7,
        }
    )
    component_rule_group = RuleGroup(
        component_start_rule,
        component_end_rule,
        [
            component_start_rule,
            component_end_rule,
            RegExRule(
                r"(-)\s+(\w+)\s+(\w+)",
                {
                    "head": 1,
                    "node_name": 2,
                    "node_type": 3,
                }
            ),
            RegExRule(
                r"(\+)\s+(\w+)\s*\(\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*\)\s*(\w+)\s+;",
                {
                    "head": 1,
                    "state": 2,
                    "x": 3,
                    "y": 7,
                    "dir": 11,
                }
            ),
        ]
    )

    other_rule_group = RuleGroup(
        
        AlwaysMatchedRule(),
        AlwaysMatchedRule(),
        [
            design_rule,
            diearea_rule,
            row_rule,
            SkipRule()
        ]
    )

    def_ent_format = EntryFormatWithRuleGroups(
        [
            component_rule_group,
            other_rule_group
        ]
    )

    assert def_file is not None and os.path.exists(def_file)
    database["def_origin"] = [""]
    database["design_config"] = {}
    database["diearea_rect"] = []
    database["row"] = {}
    with open(def_file, "r") as f:
        for line in f:
            output = def_ent_format(line)
            if output.get("entry", None) is not None:
                entry = output.get("entry")
                database["def_origin"][-1] += "%s" % (entry)
            else:
                database["def_origin"].append("")
            if output == {}:
                continue

            if "num_comps" in output.keys():
                database.update(output)
            elif "key" in output.keys():
                database["design_config"][output["key"]] = eval(output["value"]) 

            elif "lower_x" in output.keys():
                database["diearea_rect"].extend([eval(output['lower_x']), 
                                                eval(output['lower_y']),
                                                eval(output['upper_x']),
                                                eval(output['upper_y'])])
            elif "row_y" in output.keys():
                database["row"][output["id"]] = {}
                database["row"][output["id"]]["row_x"] = int(output["row_x"])
                database["row"][output["id"]]["row_y"] = int(output["row_y"])
                database["row"][output["id"]]["after_do"] = int(output["after_do"])
                database["row"][output["id"]]["after_by"] = int(output["after_by"])
                database["row"][output["id"]]["step_x"] = int(output["step_x"])
                database["row"][output["id"]]["step_y"] = int(output["step_y"])
            else:
                head = output.get("head", None)
                if head == '-':
                    node_name = output["node_name"]
                    database["nodes"][node_name] = deepcopy(output)
                elif head == '+':
                    state = output["state"]
                    if state == "FIXED":
                        database["macros"].append(node_name)
                    database["nodes"][node_name].update(output)
                else:
                    continue   
    
def get_scaling_ratio(database):
    design_range_x = database["design_config"]["FE_CORE_BOX_UR_X"] - database["design_config"]["FE_CORE_BOX_LL_X"]
    design_range_y = database["design_config"]["FE_CORE_BOX_UR_Y"] - database["design_config"]["FE_CORE_BOX_LL_Y"]
    diearea_range_x = database["diearea_rect"][2] - database["diearea_rect"][0]
    diearea_range_y = database["diearea_rect"][3] - database["diearea_rect"][1]

    ratio_x = design_range_x / diearea_range_x
    ratio_y = design_range_y / diearea_range_y

    return ratio_x, ratio_y

def get_inv_scaling_ratio(database):
    design_range_x = database["design_config"]["FE_CORE_BOX_UR_X"] - database["design_config"]["FE_CORE_BOX_LL_X"]
    design_range_y = database["design_config"]["FE_CORE_BOX_UR_Y"] - database["design_config"]["FE_CORE_BOX_LL_Y"]
    diearea_range_x = database["diearea_rect"][2] - database["diearea_rect"][0]
    diearea_range_y = database["diearea_rect"][3] - database["diearea_rect"][1]

    ratio_x = diearea_range_x / design_range_x
    ratio_y = diearea_range_y / design_range_y

    return ratio_x, ratio_y
        

def write_def(macro_pos, database, def_file, ratio_x, ratio_y, limited_x=None, limited_y=None, is_dataset=False):
    def_origin = list(reversed(database["def_origin"]))
    content = def_origin.pop()

    content += f"DIEAREA ( 0 0 ) ( {limited_x} {limited_y} ) ;\n"
    content += def_origin.pop()

    delta =  database["row"]['2']["row_y"] - database["row"]['1']["row_y"]
    row_id_lst = sorted(database["row"].keys(), key=lambda x:int(x))

    for row_id in row_id_lst:
        after_do = min(database["row"][row_id]["after_do"], (limited_x - database["row"][row_id]["row_x"]) // database["row"][row_id]["step_x"])
        if database["row"][row_id]["row_y"] + delta <= limited_y:
            content += f"ROW coreROW_{row_id} core {database['row'][row_id]['row_x']} " + \
                       f"{database['row'][row_id]['row_y']} N DO {after_do} " + \
                       f"BY {database['row'][row_id]['after_by']} STEP {database['row'][row_id]['step_x']} " + \
                       f"{database['row'][row_id]['step_y']} ;\n"
        content += def_origin.pop()
    
    content += "COMPONENTS %s ;\n" % (database["num_comps"])

    node_list = database["nodes"].keys()
    inv_ratio_x, inv_ratio_y = get_inv_scaling_ratio(database)
    for node_name in node_list:
        node_info = database["nodes"][node_name]
        content += \
            "- %(node_name)s %(node_type)s\n" % node_info
        if node_name in macro_pos.keys():
            x, y, _, _ = macro_pos[node_name]
            x = round(x * ratio_x + ratio_x) 
            y = round(y * ratio_y + ratio_y)

            # inv scaling
            x *= inv_ratio_x
            y *= inv_ratio_y
            
            if is_dataset:
                content += \
                    f"\t+ PLACED ( {x} {y} ) {node_info['dir']} ;\n"
            else:
                content += \
                    f"\t+ FIXED ( {x} {y} ) {node_info['dir']} ;\n"
        else:
            content += \
                "\t+ PLACED ( %(x)s %(y)s ) %(dir)s ;\n" % node_info
            
    content += "END COMPONENTS\n"

    while len(def_origin) > 0:
        content += def_origin.pop()


    with open(def_file, "w") as f:
        f.write(content)

def write_def_from_tensor(macro_pos, database, def_file, limited_x=None, limited_y=None, is_dataset=False):
    def_origin = list(reversed(database["def_origin"]))
    content = def_origin.pop()

    content += f"DIEAREA ( 0 0 ) ( {limited_x} {limited_y} ) ;\n"
    content += def_origin.pop()

    delta =  database["row"]['2']["row_y"] - database["row"]['1']["row_y"]
    row_id_lst = sorted(database["row"].keys(), key=lambda x:int(x))

    for row_id in row_id_lst:
        after_do = min(database["row"][row_id]["after_do"], (limited_x - database["row"][row_id]["row_x"]) // database["row"][row_id]["step_x"])
        if database["row"][row_id]["row_y"] + delta <= limited_y:
            content += f"ROW coreROW_{row_id} core {database['row'][row_id]['row_x']} " + \
                       f"{database['row'][row_id]['row_y']} N DO {after_do} " + \
                       f"BY {database['row'][row_id]['after_by']} STEP {database['row'][row_id]['step_x']} " + \
                       f"{database['row'][row_id]['step_y']} ;\n"
        content += def_origin.pop()
    
    content += "COMPONENTS %s ;\n" % (database["num_comps"])

    node_list = database["nodes"].keys()
    inv_ratio_x, inv_ratio_y = get_inv_scaling_ratio(database)
    for node_name in node_list:
        node_info = database["nodes"][node_name]
        content += \
            "- %(node_name)s %(node_type)s\n" % node_info
        if node_name in macro_pos.keys():
            x, y, _, _ = macro_pos[node_name]
            x = round(x) 
            y = round(y)

            # inv scaling
            x *= inv_ratio_x
            y *= inv_ratio_y
            
            if is_dataset:
                content += \
                    f"\t+ PLACED ( {x} {y} ) {node_info['dir']} ;\n"
            else:
                content += \
                    f"\t+ FIXED ( {x} {y} ) {node_info['dir']} ;\n"
        else:
            content += \
                "\t+ PLACED ( %(x)s %(y)s ) %(dir)s ;\n" % node_info
            
    content += "END COMPONENTS\n"

    while len(def_origin) > 0:
        content += def_origin.pop()


    with open(def_file, "w") as f:
        f.write(content)


import numpy as np
import os
import math
import torch
from torch_geometric.data import Data
from collections import defaultdict

from itertools import combinations 

class ProblemInstance():
    def __init__(self, args, benchmark, cache_dir = "benchmark_cache"):
        self.benchmark = benchmark
        self.args = args

        self.database = {}
        # print(args)
        placedb_info_cache_path = os.path.join(cache_dir, f"{benchmark}_placedb_info.pt")
        if False and os.path.exists(placedb_info_cache_path):
            # print(f"Loading cached placedb_info from {placedb_info_cache_path}")
            # placedb_info = torch.load(placedb_info_cache_path,weights_only=False )
            pass
        else:
            placedb_info = read_benchmark(database=self.database, benchmark=os.path.join(args.BENCHMARK_DIR, benchmark), args=args)
            # os.makedirs(cache_dir, exist_ok=True)
            # torch.save(placedb_info, placedb_info_cache_path)
            # print(f"Saved placedb_info to {placedb_info_cache_path}")
        
        self.node_info = placedb_info["node_info"]
        self.node_info_raw_id_name = placedb_info["node_info_raw_id_name"]
        self.node_cnt = placedb_info["node_cnt"]
        self.port_info = placedb_info["port_info"]
        self.net_info = placedb_info["net_info"]
        self.net_cnt = placedb_info["net_cnt"]
        self.max_height = placedb_info["max_height"]
        self.max_width = placedb_info["max_width"]
        self.port_to_net_dict = placedb_info["port_to_net_dict"]
        self.cell_total_area = placedb_info["cell_total_area"]

        self.node_to_net_dict = get_node_to_net_dict(self.node_info, self.net_info)
        self.node_id_to_name = get_node_id_to_name_topology(self.node_info, self.node_to_net_dict, self.net_info, self.benchmark)
        
        self.max_net_per_node = 0
        for node in self.node_to_net_dict:
            self.max_net_per_node = max(self.max_net_per_node, len(self.node_to_net_dict[node]))

        self.inv_scaling_ratio_x, self.inv_scaling_ratio_y = get_inv_scaling_ratio(self.database) # 2000, 2000

        # cache for compute regularity 
        self.original_max_height = self.max_height
        self.original_max_width = self.max_width
        self.original_ratio_x = self.max_width / self.args.grid
        self.original_ratio_y = self.max_height / self.args.grid

        self.ratio_x = self.max_width / self.args.grid
        self.ratio_y = self.max_height / self.args.grid
        self.ratio_sum = self.ratio_x + self.ratio_y

    def _build_pyg_data(self):
        V = self.node_cnt
        node_name_to_id = {name: i for i, name in enumerate(self.node_id_to_name)}

        # Normalize sizes to [0, 2] corresponding to chip normalized to [-1, 1]
        sizes = []
        for name in self.node_id_to_name:
            sx = 2 * self.node_info[name]['x'] / self.max_width
            sy = 2 * self.node_info[name]['y'] / self.max_height
            sizes.append([sx, sy])
        cond_x = torch.tensor(sizes, dtype=torch.float32)

        # is_ports all 0
        is_ports = torch.zeros((V, 1), dtype=torch.float32)

        # Normalize initial positions (centers) to [-1, 1]
        positions = []
        for name in self.node_id_to_name:
            center_x = self.node_info[name].get("raw_x", 0) + self.node_info[name]['x'] / 2
            center_y = self.node_info[name].get("raw_y", 0) + self.node_info[name]['y'] / 2
            norm_x = 2 * center_x / self.max_width - 1
            norm_y = 2 * center_y / self.max_height - 1
            positions.append([norm_x, norm_y])
        x = torch.tensor(positions, dtype=torch.float32)

        # Build edges and attrs using net_info directly
        edge_list = []
        attr_list = []
        for net_name in self.net_info:
            macro_to_off = {}
            for macro in self.net_info[net_name]["nodes"]:
                off_x = self.net_info[net_name]["nodes"][macro].get("x_offset", 0)
                off_y = self.net_info[net_name]["nodes"][macro].get("y_offset", 0)
                norm_off_x = 2 * off_x / self.max_width
                norm_off_y = 2 * off_y / self.max_height
                macro_to_off[macro] = (norm_off_x, norm_off_y)

            connected_macros = list(macro_to_off.keys())
            if len(connected_macros) < 2:
                continue

            for a, b in combinations(connected_macros, 2):
                off_a = macro_to_off[a]
                off_b = macro_to_off[b]
                i = node_name_to_id[a]
                j = node_name_to_id[b]
                edge_list.append([i, j])
                attr_list.append([off_a[0], off_a[1], off_b[0], off_b[1]])
                edge_list.append([j, i])
                attr_list.append([off_b[0], off_b[1], off_a[0], off_a[1]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(attr_list, dtype=torch.float32)
        else:
            print("Warning: no edges in the graph.")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)

        cond = Data(x=cond_x, edge_index=edge_index, edge_attr=edge_attr, is_ports=is_ports)
        chip_size = np.array([0.0, 0.0, 2.0, 2.0])  # normalized to [-1, 1]
        cond.chip_size = chip_size
        cond.benchmark_name = self.benchmark
        return x, cond

    

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




def comp_res(problem, node_pos):

    hpwl = 0.0
    for net_name in problem.net_info:
        max_x = 0.0
        min_x = problem.max_height * 1.1
        max_y = 0.0
        min_y = problem.max_height * 1.1
        for node_name in problem.net_info[net_name]["nodes"]:
            if node_name not in node_pos:
                continue
            h = problem.node_info[node_name]['x']
            w = problem.node_info[node_name]['y']
            pin_x = node_pos[node_name][0] + h / 2.0 + problem.net_info[net_name]["nodes"][node_name]["x_offset"]
            pin_y = node_pos[node_name][1] + w / 2.0 + problem.net_info[net_name]["nodes"][node_name]["y_offset"]
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)
        for port_name in problem.net_info[net_name]["ports"]:
            h = problem.port_info[port_name]['x']
            w = problem.port_info[port_name]['y']
            pin_x = h
            pin_y = w
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)
        if min_x <= problem.max_height:
            hpwl_tmp = (max_x - min_x) + (max_y - min_y)
        else:
            hpwl_tmp = 0
        if "weight" in problem.net_info[net_name]:
            hpwl_tmp *= problem.net_info[net_name]["weight"]
        hpwl += hpwl_tmp


    regularity = compute_regularity(node_pos=node_pos, problem=problem)
    return hpwl, regularity


def compute_regularity(node_pos, problem: ProblemInstance):
    """
    Compute the weighted average regularity of macro placements (adjusted to use boundary-based logic).
    
    Args:
        node_pos: dict, {node_name: (x, y, w, h)} - left-bottom corner coordinates and sizes in real circuit scale
        problem: ProblemInstance
    
    Returns:
        float: Average regularity (higher = more centered/regular).
    """
    inv_r_x, inv_r_y = get_inv_scaling_ratio(problem.database)

    max_width = problem.max_width
    max_height = problem.max_height
    bound_x = max_width * inv_r_x 
    bound_y = max_height * inv_r_y  

    if len(node_pos) == 0:
        print("No macros placed for regularity computation.")
        return 0.0
    
    total_area = 0.0
    total_reg = 0.0
    
    for node_name in node_pos:

        x, y, _, _ = node_pos[node_name]
        w, h = problem.node_info[node_name]['x'], problem.node_info[node_name]['y']
        

        x_scaled = x * inv_r_x 
        y_scaled = y * inv_r_y 
        w_scaled = w * inv_r_x 
        h_scaled = h * inv_r_y 
        
    
        right_x = x_scaled + w_scaled  
        top_y = y_scaled + h_scaled   
        

        area = w_scaled * h_scaled
        total_area += area
        
      
        dist_x_left = max(x_scaled, 0.0)  
        dist_x_right = max(bound_x - right_x, 0.0)  
        min_x_dist = min(dist_x_left, dist_x_right)

        dist_y_bottom = max(y_scaled, 0.0)  
        dist_y_top = max(bound_y - top_y, 0.0)  
        min_y_dist = min(dist_y_bottom, dist_y_top)
        
        reg_per_macro = min_x_dist + min_y_dist
        
        total_reg += reg_per_macro * area
    
    return total_reg / total_area if total_area > 0 else 0.0
