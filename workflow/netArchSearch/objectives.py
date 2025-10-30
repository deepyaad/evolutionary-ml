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
def binary_accuracy(sol):
    '''
    purpose: objective function to maximize binary accuracy (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: binary accuracy metric
    '''
    return 1 - sol.metrics['binary_accuracy']

@profile
def cosine_similarity(sol):
    '''
    purpose: objective function to maximize cosine similarity (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: cosine similarity metric
    '''
    return 1 - sol.metrics['cosine_similarity']

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
def mean_squared_error(sol):
    '''
    purpose: objective function to minimize mean squared error
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: mean_squared_error metric
    '''
    return sol.metrics['mean_squared_error']

@profile
def recall(sol):
    '''
    purpose: objective function to maximize recall (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: recall metric
    '''
    return 1 - sol.metrics['recall']

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
def specificity(sol):
    '''
    purpose: objective function to maximize specificity (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: specificity metric
    '''
    return 1 - sol.metrics['specificity']

@profile
def auc(sol):
    '''
    purpose: objective function to maximize area under the curve (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: area under the curve metric
    '''
    return 1 - sol.metrics['auc']

@profile
def sensitivity(sol):
    '''
    purpose: objective function to maximize sensitivity (using the complement)
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: sensitivity metric
    '''
    return 1 - sol.metrics['sensitivity']