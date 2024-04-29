#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
import importlib
import eventlet
import inspect
import gast
import astor
import argparse
import shutil
import types
import sys
import re
import io
import os

import concurrent.futures

import paddle

from translate_src.modify_transformer import AddParamTransformer, DelParamTransformer, RenameParamTransformer, RepAttributeTransformer
from translate_src.upgrade_models_api_utils import get_cur_file_list, load_replace_dict, load_modify_dict, check_dir, load_delete_dict
from translate_src.upgrade_models_api_utils import print_info, import_module, check_paddle, load_config
from translate_src.node_operation import insert_import_module, insert_import_module_with_postion
from translate_src.replace_full_name_transformer import PaddleReplaceFullNameTransformer
from translate_src.import_transformer import ImportVisitor
from translate_src.from_count_transformer import FromCountVisitor


BUILD_IN_FUN = dir(__builtins__)
MODIFY_DICT = "./translate_src/dict/modify.dict"
ARGS_FILE = "./translate_src/conf/upgrade.conf"
DELETE_DICT = "./translate_src/dict/delete.dict"
PROCESS_OR_THREAD = "MULTI_PROCESS"
MAX_WORKERS = 7




def scan_module_import(root):
    """
    scan each py module to build the mapping dict for relacing api full name

    Args:
        root (gast node)

    Returns:
        dict(): combination of import_dict and from_import_dict
    """
    import_visitor = ImportVisitor(root)
    import_visitor.visit(root)

    import_dict, from_import_dict = import_visitor.import_dict, import_visitor.from_import_dict

    # captured input
    captured_dict = dict()

    for k, v in import_dict.items():
        print("key,value import_dict: ", k, v)
        if k and "paddle" in v:
            captured_dict.update({k: v})

    for k, v in from_import_dict.items():
        print("key, v, from_dict: ", k, v)
        if k and "paddle" in v:
            captured_dict.update({k: v})

    return captured_dict


def replace_full_name(root, import_map):
    """
    relace api name with api full name, in order for the
    upgrade function catching the mapping key
    e.g:
        from paddle.fluid.dygraph.nn import Conv2D
        self.conv1 = nn.Conv2D(1, 6, 5, act='sigmoid')
        -->
        self.conv1 = paddle.fluid.dygraph.nn.Conv2D(1, 6, 5, act='sigmoid')


    """
    replace_transformer = PaddleReplaceFullNameTransformer(root)
    replace_transformer.replace(import_map)
    return root


def transformer_root(root, modify_dict):
    RenameParamTransformer(root).replace(modify_dict)
    AddParamTransformer(root).add(modify_dict)
    DelParamTransformer(root).delete(modify_dict)
    RepAttributeTransformer(root).replace(modify_dict)
    return root


def transformer_file(upgrade_config_dict, input, modify_dict=None,
                     is_dir=False, delete_pattern=None):

    content = open(input, 'r').readlines()
    match = re.search(delete_pattern, "\n".join(content))
    if match:
        delete_api = match.group(0)
        print_info(
            "\033[1;31m %s API has been deleted, please check file %s, use a replacement policy and convert it manually\033[0m"
            % (delete_api, input))
    
    input = os.path.normpath(input)
    (dirpath, filename) = os.path.split(input)
    abs_path = os.path.abspath(input)

    if filename.startswith("."):
        return -1

    if is_dir:
        out_dir = os.path.join(upgrade_config_dict["output_path"], dirpath)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, 0o777)
        out_file = os.path.join(out_dir, filename)
    else:
        out_file = upgrade_config_dict["output_path"]

    out_file = os.path.normpath(out_file)
    check_stat = check_paddle(input)
    if filename.endswith(".sh") or check_stat != 0:
        with open(out_file, 'w') as fw:
            print_info("\033[1;34mStart upgrading model %s\033[0m" % (input))
            fw.write(open(input, 'r').read())
            print_info(
                "\033[1;34mUpgrade Complete. The updated file %s has been written sucess\033[0m"
                % (out_file))
            return -1

    cache_file = None
    module_name = filename.rstrip(".py")
    #TODO basic strategy for avoiding _builtin_ module conflict, which need more general solution
    if module_name in BUILD_IN_FUN + ["bert", "word2vec", "yolov3"]:
        cache_file = "cache_%s" % filename
        cache_dir = os.path.join(dirpath, cache_file)
        shutil.copyfile(input, cache_dir)
        module_name = cache_file.rstrip(".py")
    else:
        module_name = filename.rstrip(".py")

    try:
        mdl_inst = importlib.import_module(module_name, package=dirpath)
    except Exception as e:
        print_info(
            "\033[1;32m%s, so we use another strategy to dynamically import module\033[0m"
            % e)
        module_name, dirpath = os.path.split(abs_path)
        print("-->module name and package name:", module_name, dirpath)
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        new_mdl_inst = importlib.util.module_from_spec(spec)
        mdl_inst = new_mdl_inst

    size = os.path.getsize(input)
    print_info("\033[1;34mStart upgrading model %s\033[0m" % (input))

    if size != 0:
        try:
            root = gast.parse(inspect.getsource(mdl_inst))
            from_count_visitor = FromCountVisitor(root)
            from_count_visitor.visit(root)
            future_count = from_count_visitor.from_import_count
            print_info("\033[1;34mfuture count is %s \033[0m" % (future_count))

            insert_import_module_with_postion(root,
                                              mdl_name="paddle",
                                              pos=future_count)

            import_dict = scan_module_import(root)
            root = replace_full_name(root, import_dict)

            root = transformer_root(root, modify_dict)
            with open(out_file, 'w', encoding="utf8") as fw:
                fw.write(astor.to_source(gast.gast_to_ast(root)))
        except Exception as e:
            print_info(
                '\033[1;33;41mParser and upgrade %s error!!, please check API and convert it manually, with error %s \033[0m'
                % (input, e))
            return -1
    else:
        with open(out_file, 'w') as fw:
            fw.write(open(filename, 'r').read())
    if cache_file is not None:
        os.remove(cache_dir)

    print_info(
        "\033[1;34mUpgrade Complete. The updated file %s has been written sucess\033[0m"
        % (out_file))
    print_info("")


def main(upgrade_api_args):
    if not upgrade_api_args.get("args_file", None):
        print(
            "\033[1;34mPlease set config file!! Default path is translate_src/conf/upgrade.conf\033[0m"
        )
        exit(1)
    if not upgrade_api_args.get("modify_dict", None):
        print(
            "\033[1;34mPlease set modify_dict file!! Default path is translate_src/dict/modify.dict\033[0m"
        )
        exit(1)

    upgrade_config_dict = load_config(upgrade_api_args["args_file"])
    if not os.path.isfile(upgrade_config_dict["input_path"]):
        file_py_list = get_cur_file_list()
    else:
        file_py_list = upgrade_config_dict["input_path"]
    modify_dict = load_modify_dict(upgrade_api_args["modify_dict"])
    delete_list = load_delete_dict(upgrade_api_args["delete_dict"])
    delete_pattern = "|".join(delete_list)

    if isinstance(file_py_list, list):
        if PROCESS_OR_THREAD == "MULTI_PROCESS":
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
            future_list = []  
            for path in file_py_list:
                future = executor.submit(transformer_file, 
                                    upgrade_config_dict,
                                    path,
                                    modify_dict,
                                    True,
                                    delete_pattern
                                    ) # 生成future实例
                future_list.append(future)
            
        elif PROCESS_OR_THREAD == "MULTI_THREAD":
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
            future_list = []  
            for path in file_py_list:
                future = executor.submit(transformer_file, 
                                    upgrade_config_dict,
                                    path,
                                    modify_dict,
                                    True,
                                    delete_pattern
                                    ) # 生成future实例
                future_list.append(future)
        
        executor.shutdown()     
        for future in concurrent.futures.as_completed(future_list):
            if future.exception() is not None:
                print_info(
                    "\033[1;31m parallel error with future exception: %s \033[0m"%(future.exception())
                    )
            else:
                future.result()
        print_info(
                "\033[1;33m all done.\033[0m")


    elif file_py_list is None:
        print_info(
            "\033[1;31mInput error: input must be a directory or a python file\033[0m"
        )
    else:
        try:
            eventlet.monkey_patch()
            with eventlet.Timeout(30, False):
                try:
                    transformer_file(upgrade_config_dict,
                                        upgrade_config_dict["input_path"],
                                        modify_dict,
                                        is_dir=False,
                                        delete_pattern=delete_pattern
                                        )
                except Exception as e:
                    print_info(
                        "\033[1;31m %s upgrade error, please check file, use a replacement policy and convert it manually, with error %s. \033[0m"
                        % (upgrade_config_dict["input_path"], e))
        except Exception as e:
            print_info(
                "\033[1;31m %s upgrade timeout, please check file, use a replacement policy and convert it manually, with error %s.\033[0m"
                % (upgrade_config_dict["input_path"], e))


if __name__ == "__main__":
    upgrade_api_args = {
        "modify_dict": MODIFY_DICT,
        "args_file": ARGS_FILE,
        "delete_dict": DELETE_DICT
    }

    main(upgrade_api_args)
