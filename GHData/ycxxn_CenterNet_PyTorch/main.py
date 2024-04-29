from config import get_cfg

# a = cfg._C.clone()
a = get_cfg()

a.merge_from_file("test.yaml")

# print(a.MODEL.BACKBONE)
print(a)