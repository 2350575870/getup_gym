
algo_registry = {}
net_registry = {}
encoder_registry = {}
encoder_alg_registry = {}

def regist_algo(name, algo_class):
    algo_registry[name] = algo_class


def get_algo_class(name):
    return algo_registry[name]


def regist_net(name, net_class):
    net_registry[name] = net_class


def get_net_class(name):
    return net_registry[name]

def regist_encoder(name, encoder_class):
    encoder_registry[name] = encoder_class

def get_encoder_class(name):
    return encoder_registry[name]

def regist_encoder_alg(name, encoder_alg_class):
    encoder_alg_registry[name] = encoder_alg_class

def get_encoder_alg_class(name):
    return encoder_alg_registry[name]
