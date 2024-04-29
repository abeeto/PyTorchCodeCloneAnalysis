"""

A place to keep modifications


"""

#In _jit_internal

def parseExpr(expr: str, module: Callable):
    """

    A annotations converter. This will take
    an annotation defined as a string, and convert
    each element into an associated type. It
    will then return the result.

    :param expr: The expression to convert, in string format
    :param module: A callable which, when provided with a string
        representation of a type, returns the type from the
        module. For instance, giving it 'Union', and so long
        as Union was imported at the top level, it will
        spit out Union.
    :return: A instance of the type.
    """

    try:
        import re
        from collections import namedtuple
        # Create regex template. This code is verbose deliberately for ease of maintainability.
        split_on = ('[', ',', ']')
        split_on = re.escape("".join(split_on))  # Turns it into a properly escaped string

        pattern = "([{split}])".format(split=split_on)

        # Perform split. Strip out spaces, and then ' and ". The latter two
        # are to handle forward annotations. Then strip tokens to ready for processing
        strip_out = ("", ",")
        expr = expr.strip()
        expr_list = re.split(pattern, expr)
        expr_list = [item for item in expr_list if
                     item not in strip_out]  # regex will yield empty tokens sometimes.
        expr_list = [item.strip().strip("'").strip('"') for item in
                     expr_list]  # Strips spaces, and preps forward annotations.

        # Three conditions exist. You are either collecting
        # possible inputs, creating a new arg list,
        # or pushing the current arg list through a typing
        # class. This can easily be managed using a stack.
        stack: List[List[Any]] = []
        level: List[Any] = []
        for token in expr_list:
            if token == "[":
                # This stores the current level in the stack, then goes ahead
                # and creates a new level. We know that what we are accumulating must be an input
                # of some sort.
                stack.append(level)
                level = []
            elif token == "]":
                # Go and compile the last feature of the previous level. Whatever we have accumulated is
                # an input into a typing feature. Pop off the last thing that went by, call it with "[stuff]"
                # and stick it back into the stack.
                superlevel = stack.pop()
                unbuilt_typing = superlevel.pop()
                built_typing = unbuilt_typing[tuple(level) if len(level) > 1 else level[0]]
                superlevel.append(built_typing)
                level = superlevel
            elif token == "()":
                # Special case to manage Tuple[()] cases
                obj = ()
                level.append(obj)
            else:
                # Just create a normal old token, and append it.
                obj = lookupInModule(token, module)
                assert obj is not None, f"Unresolvable type {token}"
                level.append(obj)
        return level.pop()

    except Exception as err:
        """
        The python resolver fails in several cases in known unit tests, and is intended
        to fall back gracefully to the c++ resolver in general.  For example, python 2 style
        annotations which are frequent in our unit tests often fail with types e.g. int not
        resolvable from the calling frame.
        """
        return None