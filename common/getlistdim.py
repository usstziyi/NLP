# 定义一个函数，用于获取列表的维度(层数)
def get_list_dimensions(lst):
    if not isinstance(lst, list):
        return 0
    if not lst:
        return 1  # 空列表视为1维
    dims = [get_list_dimensions(item) for item in lst]
    return 1 + max(dims)