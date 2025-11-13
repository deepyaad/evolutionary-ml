from profiler import profile


@profile
def development_time(sol):
    '''
    purpose: objective function to minimize development time
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: development time metric from sol.metrics
    '''
    return sol.metrics['development_time']


@profile
def total_layers(sol):
    '''
    purpose: objective function to minimize number of layers
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: layer count metric from sol.metrics
    '''
    return sol.metrics['total_layers']


@profile
def total_nodes(sol):
    '''
    purpose: objective function to minimize number of nodes
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: total nodes for all layers metric from sol.metrics
    '''
    return sol.metrics['total_nodes']


@profile
def loss(sol):
    '''
    purpose: objective function to minimize loss
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: loss metric from sol.metrics
    '''
    return sol.metrics['model']['loss']


@profile
def accuracy(sol):
    '''
    purpose: objective function to maximize accuracy (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: accuracy metric
    '''
    return 1 - sol.metrics['model']['accuracy_macro']


@profile
def true_positive_rate(sol):
    '''
    purpose: objective function to maximize true positive rate (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: true positive rate metric
    '''
    return 1 - sol.metrics['true_positive_rate']


@profile
def true_negative_rate(sol):
    '''
    purpose: objective function to maximize true negative rate (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: true negative rate metric
    '''
    return 1 - sol.metrics['true_negative_rate']


@profile
def false_pos_rate(sol):
    '''
    purpose: objective function to minimize false positive rate (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: false positive rate metric
    '''
    return 1 - sol.metrics['model']['false_pos_rate_macro']

@profile
def false_negative_rate(sol):
    '''
    purpose: objective function to minimize false negative rate (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: false negative rate metric
    '''
    return sol.metrics['false_negative_rate']


@profile
def precision(sol):
    '''
    purpose: objective function to maximize precision (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: precision metric
    '''
    return 1 - sol.metrics['model']['precision_macro']


@profile
def f1_score(sol):
    '''
    purpose: objective function to maximize f1 score (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: f1 score metric
    '''
    return 1 - sol.metrics['model']['f1_macro']