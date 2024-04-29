import pyparsing as pp




nested_str = "<<>test <<a>>>"
nested_content = pp.Forward()
ignore_tier = pp.Literal("<") + pp.SkipTo(pp.Literal(">"), ignore=nested_content) + pp.Literal(">")
nested_content <<= ignore_tier
expression = pp.Literal("<") + pp.SkipTo(pp.Literal(">"), ignore=ignore_tier) + pp.Literal(">")
for match in expression.scan_string(nested_str):
    print(match)