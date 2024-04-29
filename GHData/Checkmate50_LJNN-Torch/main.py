from subprocess import call
import bin.helper


"""
This script acts as the glue to automate the steps of creating the training/testing files for neural network training, training and testing a neural network, and creating the necessary files for integration with RuNNer++

In particular, this script activates the files given in the main.info file, particularly the createTT, trainNN, and testNN files.  This script provides each of these files with a link to the appropriate info files as follows:

... [createTT].extension [ttInfo] [symmInfo] [nnInfo]
... [*NN].extension [ttInfo] [nnInfo]

This script expects that createTT will create all necessary files in the appropriate runner directory except input.nn and input.nn.RuNNer++.  These two files are the responsibility of either [trainNN] or [translateNN] if provided.

For details on the main.info file which guides the use of this script, please examine the main.info included with this package.

Written by Dietrich Geisler
"""

def verify_info():
    """
    Using the information from main.info
    And the information given in the file pointed to by the dataInfo field, 
    Verify each info file has the required data
    And give warnings when a data field is unrecognized.

    Also modifies these files so that case-sensitivity matches when necessary
    """
    data = bin.helper.get_data("main.info", {"verbose" : False})
    verbose = data["verbose"]
    expect = {}
    expected_ref = ["expectInfo", "ttInfo", "nnInfo"]
    for ref in expected_ref:
        if not data.has_key(ref.lower()):
            helper.print_error(ref + " data field required in main.info")
            exit()
    with open(data["expectinfo"], "r") as expect_file:
        index = ""
        for line in expect_file:
            sline = line[:line.find("#")].strip().split()
            if len(sline) < 1:
                continue
            if sline[0][-1] == ":":
                index = sline[0][:-1]
                expect[index] = []
                continue
            if index == "":
                bin.helper.print_error("expect info file formatted incorrectly")
                exit()
            expect[index].append(sline)
    files = {"mainInfo" : "main.info", "ttInfo" : data["ttinfo"], "nnInfo" : data["nninfo"]}
    for key in files:
        data_fields = None
        data_fields = list(bin.helper.get_data(files[key], {}).keys())
        for expected in expect[key]:
            if expected[0].lower() not in data_fields:
                if expected[1].startswith("o"):
                    if verbose:
                        bin.helper.print_info("Option " + expected[0] + " not found in " + files[key])
                else:
                    bin.helper.print_error("ERROR: Required option " + expected[0] + " not found in " + files[key])
                    exit() 
        possible = list(map(str.lower, map(lambda x : x[0], expect[key])))
        for field in data_fields:
            if field not in possible:
                bin.helper.print_warning("WARNING: Unknown option " + field + " found in " + files[key] + " (will be ignored)")
        

def execute_file(data, filename, args=None, verbose=False):
    """
    Given a filename and arguments 'args'
    Attempts to execute that file with the given arguments
    Using the run command associated with that filetype
    Raises a ValueError if the file extension is not supported
    """
    print(filename)
    if filename.endswith(".py"):
        if verbose:
            bin.helper.print_info("python " + filename + " " + " ".join(args))
        e = call([data["python"], filename] + args)
    elif filename.endswith(".lua"):
        if verbose:
            bin.helper.print_info("th " + filename + " " + " ".join(args))
        e = call([data["torch"], filename] + args)
    else:
        bin.helper.print_error("ERROR: " + str(filename) + " not run (unrecognized file extension)")
        exit()
    if e != 0:
        bin.helper.print_error("Fatal error")
        exit()
        

def main():
    verify_info()
    defaults = {"createNewTT" : True, "createNewNN" : True, "testOnly" : False, "runTests" : True, "verbose" : False}
    data = bin.helper.get_data("main.info", defaults)
    if not data["testonly"]:
        if data["createnewtt"]:
            execute_file(data, data["creatett"], [data["ttinfo"], data["symminfo"], data["nninfo"]], verbose=data["verbose"])
        if data.has_key("createnn"):
            execute_file(data, data["createnn"], [data["ttinfo"], data["nninfo"]], verbose=data["verbose"])
        execute_file(data, data["trainnn"], [data["ttinfo"], data["nninfo"]], verbose=data["verbose"])
        if data.has_key("translatenn"):
            execute_file(data, data["translatenn"], [data["nninfo"]], verbose=data["verbose"])
    if data["runTests"]:
        execute_file(data, data["testnn"], [data["ttinfo"], data["nninfo"]], verbose=data["verbose"])
    if data["verbose"]:
        bin.helper.print_info("Exiting main.py")

if __name__ == "__main__":
    main()

    
