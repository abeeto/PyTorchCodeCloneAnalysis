#!/usr/bin/python3
# -*- coding: utf-8 -*-
# SEE this link for symbols https://latex.wikia.org/wiki/List_of_LaTeX_symbols
# LEARN some python for asigmning expresions as ":=" in https://www.python.org/dev/peps/pep-0572/
from pylatex import Document, Section, Subsection, Command
from pylatex.package import Package
from pylatex.utils import italic, NoEscape
from os import path, chdir, listdir, remove
from subprocess import check_output, STDOUT, CalledProcessError
from re import match
import json
import argparse
import configparser
# Configure Initial Variables
config_file = 'config/nconfig.cfg'
# Configure Enviroment
config_file = path.join(path.abspath(path.dirname(__file__)), config_file)
current_dir_path = path.abspath(path.dirname(__file__))
if (current_dir_path != path.abspath(path.dirname(__file__))):
    chdir(current_dir_path)
assert current_dir_path == path.abspath(path.dirname(__file__))
print("__config file__: {}".format(config_file))
assert path.isfile(config_file)
config = configparser.ConfigParser()
config.read(path.join(path.abspath(path.dirname(__file__)), config_file))
# Start
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "--v", action="store_true", help="Increase output verbosity")
    parser.add_argument("--symbol", "--s", default= "data.json", type=str, help="Symbol path to the symbol '.json' file")
    parser.add_argument("--output", "--o", default= "out.pdf", type=str, help="Path to output file. The extension defines the output.")
    parser.add_argument("--clean", "--c", action="store_true", help="Whether or  not to clean output folder from latex related aux files generating while rendeing the '.tex' file.")
    parser.add_argument("--make_pdf", "--mp", default= True, type=bool, help="Flag wheter to produce '.pdf' output file")
    parser.add_argument("--make_tex", "--mt", default= False, type=bool, help="Flag wheter to produce '.tex' output file")
    args = vars(parser.parse_args())
    if args["verbose"]:
        print("<WORKING DIRECTORY>: {}".format(current_dir_path))
        print("<CONFIG FILE>: {}".format(config_file))
        print("<CONFIG FILE DATA>:")
        for i in ["\t{}: \n\t\t{}".format(section, dict(config.items(section))) for section in dict(config.items()).keys()]:
            print(i)
        print("<SCRIPT ARGUMENTS>: {}".format(args))
# Defined functions
def import_libs_in_tex(tex_file):
    for pkg in config.get("LATEX","pakages").split(','):
        tex_file.write(pkg.strip())
        if(args["verbose"]):
            print("<LATEX PAKAGE LOADED>: {}".format(pkg.strip()))
def latexmk_render(dir_path, file_name, desired_ext, clean = False, latexmk_params = 'CAS_Default'):
    if latexmk_params == 'CAS_Default':
        latexmk_params = [param.strip() for param in config.get("LATEX","latexmk_args").split(',')]
    if desired_ext == '.pdf':
        latexmk_params += ['--dvi-', '--pdf', '--ps-']
    elif desired_ext == '.dvi':
        latexmk_params += ['--dvi', '--pdf-', '--ps-']
    elif desired_ext == '.tex':
        latexmk_params += ['--dvi-', '--pdf-', '--ps-']
    elif desired_ext == '.ps':
        latexmk_params += ['--dvi-', '--pdf-', '--ps']
    else:
        assert False, ("OUTPUT file desired extensión must be either '.pdf', '.dvi', '.tex' or '.ps'")
    latexmk_params += [file_name, '--interaction=nonstopmode']
    command = [str(config.get("LATEX","latexmk_path"))] + latexmk_params
    seen_command = set() # set to filter no redudance in command
    command = [x for x in command if not (x in seen_command or seen_command.add(x))]
    if args["verbose"]:
        print('<LATEXMK COMMAND>: {}'.format(command))
    try:
        # Clean Directory
        output = check_output([str(config.get("LATEX","latexmk_path"))] + ['--C'], stderr=STDOUT)
        # Compile latex
        output = check_output(command, stderr=STDOUT)
        if args["verbose"]:
            print("<LATEXMK STDOUT START>: \n\t{}\n<LATEXMK STDOUT END>".format("\n\t".join(output.decode().split('\n'))))
        if clean: # Clean after folder
            command = [str(config.get("LATEX","latexmk_path"))] + ['--c']
            output = check_output(command, stderr=STDOUT)
            if args["verbose"]:
                print("<LATEXMK STDOUT (clean action) START>: \n\t{}\n<LATEXMK STDOUT (clean action) END>".format("\n\t".join(output.decode().split('\n'))))
    except CalledProcessError as exc:
        print("ERROR from LATEXMK: \n\t{}".format(exc.output))
    except:
        assert False, "ERROR invocando latexmk"
def make_symbol_tex(out = 'out.tex'):
    dir_path =  path.dirname(path.abspath(out))
    file_name, desired_ext = path.splitext(path.basename(out))
    assert desired_ext in [e.strip() for e in config.get("LATEX","latexmk_valid_out").split(',')], "OUTPUT file desired extensión must be either '.pdf', '.dvi', '.tex' or '.ps'"
    tex_file = path.join(dir_path,file_name+'.tex')
    if args["verbose"]:
        print("<MAKE_SYMBOL_TEX> args :\n\tdir_path: {}; \n\tfile_name: {}; \n\tdesired_ext: {}; \n\ttex_file: {}".format(dir_path, file_name, desired_ext, tex_file))
    with open(tex_file, 'w') as tex_file:
        tex_file.write(r"\documentclass[]{article}%"+"\n")
        tex_file.write(r"\usepackage[T1]{fontenc}%"+"\n")
        tex_file.write(r"\usepackage[utf8]{inputenc}%"+"\n")
        tex_file.write(r"\usepackage[]{lmodern}%"+"\n")
        tex_file.write(r"\usepackage[]{textcomp}%"+"\n")
        tex_file.write(r"\usepackage["+config.get("LATEX", "geometry")+r"]{geometry}%"+"\n")
        tex_file.write(r"\usepackage[]{amsmath}%"+"\n")
        tex_file.write(r"\usepackage[]{amsfonts}%"+"\n")
        tex_file.write(r"\usepackage[]{amssymb}%"+"\n")
        tex_file.write(r"\usepackage[]{tabularx}%"+"\n")
        tex_file.write(r"\usepackage[]{multirow}%"+"\n")
        tex_file.write(r"\usepackage[]{graphics}%"+"\n")
        tex_file.write(r"\usepackage[]{graphicx}%"+"\n")
        tex_file.write(r"\usepackage[]{ragged2e}%"+"\n")
        # tex_file.write(r"\newcolumntype{L}{>{\raggedleft\arraybackslash}X}"+"\n")
        # tex_file.write(r"\newcolumntype{U}{>{\raggedright\arraybackslash}X}"+"\n")
        tex_file.write(r"\usepackage{array}%"+"\n")
        tex_file.write(r"\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}%"+"\n")
        tex_file.write(r"\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}%"+"\n")
        tex_file.write(r"\newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}%"+"\n")
        tex_file.write(r"%"+"\n")
        tex_file.write(r"%"+"\n")
        tex_file.write(r"%"+"\n")
        tex_file.write(r"%"+"\n")
        tex_file.write(r"\begin{document}%"+"\n")
        tex_file.write(r"\normalsize%"+"\n")
        # tex_file.write(r"\begin{table}[]"+"\n")
        # tex_file.write(r"\begin{tabular}{crcll}"+"\n")
        # tex_file.write(r"\multicolumn{5}{c}{top}                                                    \\"+"\n")
        # tex_file.write(r"\multicolumn{1}{r}{a} & b   & \multirow{2}{*}{SYMBOL} & lower & lowerlower \\"+"\n")
        # tex_file.write(r"\multicolumn{1}{r}{c} & RXR &                         & d     & e          \\"+"\n")
        # tex_file.write(r"\multicolumn{5}{c}{bot}                                                   "+"\n")
        # tex_file.write(r"\end{tabular}"+"\n")
        # tex_file.write(r"\end{table}"+"\n")
        # tex_file.write(r"\end{document}")
        # SAecond way of making
        flag = True
        symWidth = 200
        symColsNum = 5
        TOP = {

            'a01':r'\multicolumn',
            'a02':r'{',
            'a03':r'5',
            'a04':r'}',
            'a05':r'{',
            'a06':r'|',
            'a07':r'c',
            'a08':r'|',
            'a09':r'}',
            'a10':r'{',
            'a11':r'TOP',
            'a12':r'}',
            
            'z01':r'\cline{1-1}',
            'z02':r'\cline{2-2}',
            'z03':r'\cline{4-4}',
            'z04':r'\cline{5-5}'
        }
        UP  = {
            
            'a01':r'\multicolumn',
            'a02':r'{',
            'a03':r'1',
            'a04':r'}',
            'a05':r'{',
            'a06':r'|',
            'a07':r'C{'+str(int(symWidth/symColsNum))+r'px}',
            'a08':r'|',
            'a09':r'}',
            'a10':r'{',
            'a11':r'a',
            'a12':r'}',
            
            'b01':r'\multicolumn',
            'b02':r'{',
            'b03':r'1',
            'b04':r'}',
            'b05':r'{',
            'b06':r'|',
            'b07':r'C{'+str(int(symWidth/symColsNum))+r'px}',
            'b08':r'|',
            'b09':r'}',
            'b10':r'{',
            'b11':r'b',
            'b12':r'}',
            
            'c01':r'\multirow',
            'c02':r'{',
            'c03':r'2',
            'c04':r'}',
            'c05':r'{',
            'c06':r'',
            'c07':r'*',
            'c08':r'',
            'c09':r'}',
            'c10':r'{',
            'c11':r'\fontsize{8}{10}\selectfont $\Omega$',
            'c12':r'}',
            
            'd01':r'\multicolumn',
            'd02':r'{',
            'd03':r'1',
            'd04':r'}',
            'd05':r'{',
            'd06':r'|',
            'd07':r'C{'+str(int(symWidth/symColsNum))+r'px}',
            'd08':r'|',
            'd09':r'}',
            'd10':r'{',
            'd11':r'f',
            'd12':r'}',
            
            'e01':r'\multicolumn',
            'e02':r'{',
            'e03':r'1',
            'e04':r'}',
            'e05':r'{',
            'e06':r'|',
            'e07':r'C{'+str(int(symWidth/symColsNum))+r'px}',
            'e08':r'|',
            'e09':r'}',
            'e10':r'{',
            'e11':r'g',
            'e12':r'}',
            
            'z01':r'\cline{1-1}',
            'z02':r'\cline{2-2}',
            'z03':r'\cline{4-4}',
            'z04':r'\cline{5-5}'
        }
        DOWN = {
            
            'a01':r'\multicolumn',
            'a02':r'{',
            'a03':r'1',
            'a04':r'}',
            'a05':r'{',
            'a06':r'|',
            'a07':r'C{'+str(int(symWidth/symColsNum))+r'px}',
            'a08':r'|',
            'a09':r'}',
            'a10':r'{',
            'a11':r'c',
            'a12':r'}',
            
            'b01':r'\multicolumn',
            'b02':r'{',
            'b03':r'1',
            'b04':r'}',
            'b05':r'{',
            'b06':r'|',
            'b07':r'C{'+str(int(symWidth/symColsNum))+r'px}',
            'b08':r'|',
            'b09':r'}',
            'b10':r'{',
            'b11':r'd',
            'b12':r'}',
            
            'c01':r'',
            'c02':r'',
            'c03':r'',
            'c04':r'',
            'c05':r'',
            'c06':r'',
            'c07':r'',
            'c08':r'',
            'c09':r'',
            'c10':r'',
            'c11':r'',
            'c12':r'',
            
            'd01':r'\multicolumn',
            'd02':r'{',
            'd03':r'1',
            'd04':r'}',
            'd05':r'{',
            'd06':r'|',
            'd07':r'C{'+str(int(symWidth/symColsNum))+r'px}',
            'd08':r'|',
            'd09':r'}',
            'd10':r'{',
            'd11':r'h',
            'd12':r'}',
            
            'e01':r'\multicolumn',
            'e02':r'{',
            'e03':r'1',
            'e04':r'}',
            'e05':r'{',
            'e06':r'|',
            'e07':r'C{'+str(int(symWidth/symColsNum))+r'px}',
            'e08':r'|',
            'e09':r'}',
            'e10':r'{',
            'e11':r'i',
            'e12':r'}',
            
            'z01':r'\cline{1-1}',
            'z02':r'\cline{2-2}',
            'z03':r'\cline{4-4}',
            'z04':r'\cline{5-5}'
        }
        BOTTOM = {
            
            'a01':r'\multicolumn',
            'a02':r'{',
            'a03':r'5',
            'a04':r'}',
            'a05':r'{',
            'a06':r'|',
            'a07':r'c',
            'a08':r'|',
            'a09':r'}',
            'a10':r'{',
            'a11':r'BOT',
            'a12':r'}',

            'z01':r'',
            'z02':r'',
            'z03':r'',
            'z04':r''
        }
        tex_file.write(r"\begin{table}[!h]"+"\n")
        tex_file.write(r"\begin{minipage}[b]{"+str(int(symWidth))+r"px}%"+"\n")
        tex_file.write(r"\addtolength{\tabcolsep}{-5pt}"+"\n")
        tex_file.write(r"\setlength{\arrayrulewidth}{0.1pt}"+"\n")
        tex_file.write(r"\fontsize{4px}{6px}\selectfont"+"\n")
        tex_file.write(r"\begin{tabular*}{"+str(int(symWidth))+r"px}{C{"+str(int(symWidth/symColsNum))+r"px} C{"+str(int(symWidth/symColsNum))+r"px} C{"+str(int(symWidth/symColsNum))+r"px} C{"+str(int(symWidth/symColsNum))+r"px} C{"+str(int(symWidth/symColsNum))+r"px}}"+"\n")
        tex_file.write("""{a01}{a02}{a03}{a04}{a05}{a06}{a07}{a08}{a09}{a10}{a11}{a12}\\\\{z01}{z02}{z03}{z04}""".format(**TOP)+"\n")
        tex_file.write("""{a01}{a02}{a03}{a04}{a05}{a06}{a07}{a08}{a09}{a10}{a11}{a12}&{b01}{b02}{b03}{b04}{b05}{b06}{b07}{b08}{b09}{b10}{b11}{b12}&{c01}{c02}{c03}{c04}{c05}{c06}{c07}{c08}{c09}{c10}{c11}{c12}&{d01}{d02}{d03}{d04}{d05}{d06}{d07}{d08}{d09}{d10}{d11}{d12}&{e01}{e02}{e03}{e04}{e05}{e06}{e07}{e08}{e09}{e10}{e11}{e12}\\\\{z01}{z02}{z03}{z04}""".format(**UP)+"\n")
        tex_file.write("""{a01}{a02}{a03}{a04}{a05}{a06}{a07}{a08}{a09}{a10}{a11}{a12}&{b01}{b02}{b03}{b04}{b05}{b06}{b07}{b08}{b09}{b10}{b11}{b12}&{c01}{c02}{c03}{c04}{c05}{c06}{c07}{c08}{c09}{c10}{c11}{c12}&{d01}{d02}{d03}{d04}{d05}{d06}{d07}{d08}{d09}{d10}{d11}{d12}&{e01}{e02}{e03}{e04}{e05}{e06}{e07}{e08}{e09}{e10}{e11}{e12}\\\\{z01}{z02}{z03}{z04}""".format(**DOWN)+"\n")
        tex_file.write("""{a01}{a02}{a03}{a04}{a05}{a06}{a07}{a08}{a09}{a10}{a11}{a12}\\\\{z01}{z02}{z03}{z04}""".format(**BOTTOM)+"\n")
        tex_file.write(r"\end{tabular*}"+"\n")
        tex_file.write(r"\end{minipage}"+"\n")
        tex_file.write(r"\end{table}"+"\n")
        tex_file.write(r"\end{document}")
    if config.get("LATEX", "generator").lower().strip() == 'latexmk':
        latexmk_render(dir_path = dir_path, file_name = file_name, desired_ext= desired_ext, clean = args["clean"])
    else: # Not recognized selected latex compiler
        assert False, ("'{}' is not recognized as a valid latex builder. Please configure for {} the variable to be one of the valid generators for latex: {}".format(config.get("LATEX", "generator"), config_file, ['latexmk']))
    # Assert task Ending
    if (not args["verbose"]):
        assert path.isfile(out), ("ERROR: output file not found. run script with --verbose option enable")
    else: 
        assert path.isfile(out), ("ERROR: output file not found.")
    print("__output file__: {}".format(path.abspath(out)))
# Ta,latexmk_params = ''sk
if __name__ == '__main__':
    # Document with `\maketitle`  + 'latexmk_params'command activated
    # make_document(name="out", geometry_options={"tmargin": "1pt", "lmargin": "10pt", "paperwidth":"100pt","paperheight":"100pt"})
    make_symbol_tex(out = args["output"])
