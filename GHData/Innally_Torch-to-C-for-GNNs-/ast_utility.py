import ast
from _ast import AST
from sre_constants import ASSERT

cls_symbol={'if_def':False,"clsname":'',"member_variable":{}}


def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def iter_child_nodes(node):
    """
    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """
    for name, field in iter_fields(node):
        if isinstance(field, AST):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, AST):
                    yield item

def iter_code(node,annotate_fields=True):
    c_code=''
    if isinstance(node, ast.Module):
        for b in node.body:
            c_code+=iter_code(b)
    if isinstance(node,ast.ClassDef):
        assert recursive_value(node.bases[0],0) == 'torch.nnModule','Now only support conversion \
                                               based on lib torch and Pyg'
        cls_symbol['if_def']=True
        cls_symbol['cls_name']=node.name
        c_code+='class %s{\npublic:\n'%(node.name)
        for cb in node.body:
            c_code+=iter_code(cb)
        c_code+="\n}"
    if isinstance(node,ast.FunctionDef):
        if node.name=='__init__':
            c_code+=stmt_FunctionDef_init(node)
        else:
            c_code += stmt_FunctionDef(node)
        c_code+='\n'
        # if isinstance(fd,list):
        #     return fd
        # else:
        #     result=my_iter_all(fd)
        #     c_code += result
    return c_code

def read_func_args(args):
    arglist=''
    for i in args:
        if i.arg=='self':
            # 'self' is not useful in C
            continue
        else:
            if arglist!='':
                arglist+=','
            arglist+=(i.arg)
    return arglist


def expr_call(node):
    func_name=recursive_value(node.func,0)
    args=''
    for a in node.args:
        if args!='':
            args+=','
        args+=recursive_value(a,0)
    for k in node.keywords:
        if k.arg=='training':
            continue
        if args!='':
            args+=','
        args+=str(k.value.value) # TODO: remember to add keyword feature when passing an arg to a func
    return '%s(%s)'%(func_name,args)

def recursive_value(arg,depth=0):
    name=''
    if 'value' in arg._fields:
        if not isinstance(arg.value,AST):
            return str(arg.value) # when the value is not recursive, e.g., an int or a double number.
        name=recursive_value(arg.value,depth=depth+1)
    elif arg.id =='self' or arg.id== 'F' or arg.id == 'torch.nn.functional':
        return ''  # eliminate python or pytorch style
    else:
        if depth>0:
            return arg.id + '.'
        else:
            return arg.id 
    return name+arg.attr




def stmt_assign(node):
    stmt=''
    target=[]
    value=[]
    try:
        for i in node.targets[0].elts:
            target.append(recursive_value(i,0))
    except AttributeError:
        target.append(recursive_value(node.targets[0],0))
    if isinstance(node.value,ast.Tuple):
        try:
            for i in node.value.elts:
                value.append(recursive_value(i,0))
        except AttributeError:
            value.append(recursive_value(node.value,0))
    elif isinstance(node.value,ast.Constant):
        value.append(node.value.value)
    elif isinstance(node.value,ast.Call):
        value.append(expr_call(node.value))
    

    # see if it's class def, if so, add to member
    if cls_symbol['if_def']==True:
        for i,j in zip(target,value):
            jtype=j.split('(')[0]
            stmt='%s %s;\n'%(jtype,i) # to use in member initialization
            cls_symbol['member_variable'][i]=j # to use in class construction funciton

            # TODO: to distinguish the class of initial member, such as str-based(GCNConv), and basic class(int, double) 
    else:
        for t,v in zip(target,value):
            stmt += '%s=%s;\n'%(t,v)
    return stmt
        
def stmt_return(node):
    retrun_stmt='return '
    if isinstance(node.value,ast.Call):
        temp = expr_call(node.value)
    return retrun_stmt+temp+';'

def stmt_FunctionDef_init(node):
    # todo: to add read args function.
    body=''
    for i in node.body:
        if isinstance(i,ast.Assign):
            body+=stmt_assign(i)
        else:
            print("// skip this line %d."%(i.lineno))
    if node.decorator_list != []:
        assert False,'To implement later.'
    if node.type_comment is not None:
        assert False,'To implement later.'

    # class construction function
    cst=''
    for k in cls_symbol['member_variable'].keys():
        cst+='%s=%s;\n'%(k,cls_symbol["member_variable"][k])
    body+='%s(){\n%s\n}'%(cls_symbol['cls_name'],cst)  
    return body

def stmt_FunctionDef(node):
    # function name
    func_name = node.name
    # todo: to add read args function.
    func_args = read_func_args(node.args.args)
    body=''
    for i in node.body:
        if isinstance(i,ast.Assign):
            body+=stmt_assign(i)
        if isinstance(i,ast.Return):
            body+=stmt_return(i)
    if node.decorator_list != []:
        assert False,'To implement later.'
    if node.type_comment is not None:
        assert False,'To implement later.'
    function_format='double %s(%s){\n%s\n}'%(func_name,func_args,body)
    return function_format
            
