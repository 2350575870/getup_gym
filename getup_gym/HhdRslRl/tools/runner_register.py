runner_name = {}

def regist_runner(name, algo_class):
    runner_name[name] = algo_class

def get_runner(name):
    return runner_name[name]

