import inspect

def get_module_info()-> str:
    caller_frame = inspect.currentframe().f_back
    filename = inspect.getframeinfo(caller_frame).filename
    function_name = caller_frame.f_code.co_name
    return "{}::{}".format(filename.split('/')[-1].split('.')[0], function_name)
