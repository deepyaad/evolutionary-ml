from profiler import profile


@profile
def training_time(sol):
    '''
    purpose: objective function to minimize training time
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: training time metric from sol.metrics
    '''
    return sol.metrics['training_time']


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
    return sol.metrics['loss']


@profile
def accuracy(sol):
    '''
    purpose: objective function to maximize accuracy (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: accuracy metric
    '''
    return 1 - sol.metrics['accuracy']


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
def false_positive_rate(sol):
    '''
    purpose: objective function to minimize false positive rate (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: false positive rate metric
    '''
    return sol.metrics['false_positive_rate']


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
    return 1 - sol.metrics['precision']


@profile
def f1_score(sol):
    '''
    purpose: objective function to maximize f1 score (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: f1 score metric
    '''
    return 1 - sol.metrics['f1']


@profile
def auc(sol):
    '''
    purpose: objective function to maximize area under the curve (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: area under the curve metric
    '''
    return 1 - sol.metrics['auc']