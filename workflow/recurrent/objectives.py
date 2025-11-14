from profiler import profile

@profile
def unfairness(sol):
    '''
    purpose: objective function to maximize fairness (using the complement) by comparing true positive rate across all classes
    unfairness is calculate by the different between the maximum and minimum true positive rate across all classes
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: fairness metric from sol.metrics
    '''
    # get true positive rate for each class
    recall_tprs = [sol.metrics[lang+'_performance']['recall_tpr'] for lang in sol.configuration['labels_inorder']]

    # calculate and store fairness
    unfairness = max(recall_tprs) - min(recall_tprs)
    sol.metrics['unfairness'] = unfairness

    return unfairness


@profile
def miscalculation(sol):
    '''
    purpose: objective function to maximize all the macro metrics (using the complement)
    predictive performance is calculate by the average of the macro metrics
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: predictive performance metric from sol.metrics
    '''
    # get macro metrics for each class
    precision_macro = sol.metrics['precision_macro']
    recall_macro = sol.metrics['recall_macro']
    f1_macro = sol.metrics['f1_macro']
    accuracy_macro = sol.metrics['accuracy_macro']

    # calculate and store predictive performance
    predictive_performance = (precision_macro + recall_macro + f1_macro + accuracy_macro) / 4
    sol.metrics['predictive_performance'] = predictive_performance

    return 1 - predictive_performance


@profile
def model_complexity(sol):
    '''
    purpose: objective function to minimize model complexity and improve interpretability
    model complexity is calculated by count of trainable paramas, layers, neurons, and unique layer types
    TODO: determine mathematical complexity of each activation function
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: model complexity metric from sol.metrics
    '''
    # calculate complexity metrics
    trainable_params = sum([w.shape.num_elements() for w in sol.model.trainable_weights])
    num_layers = len(sol.configuration['hidden_layers']) + 2
    max_neurons = max(sol.configuration['neurons_per_layer'])
    num_unique_layer_types = len(set(sol.configuration['layer_names']))


    # normalize model metrics based on min-max heuristics to be between 0 and 1
    p = (trainable_params - 500) / (1000000 - 500)
    l = (num_layers - 2) / (10 - 2)
    n = (max_neurons - 6) / (1000 - 6)
    t = (num_unique_layer_types - 1) / (15 - 1)


    # store normalized and original metrics
    sol.metrics['normalized_trainable_params'] = p
    sol.metrics['normalized_num_layers'] = l
    sol.metrics['normalized_max_neurons'] = n
    sol.metrics['normalized_num_unique_layer_types'] = t

    sol.metrics['trainable_params'] = trainable_params
    sol.metrics['num_layers'] = num_layers
    sol.metrics['max_neurons'] = max_neurons
    sol.metrics['num_unique_layer_types'] = num_unique_layer_types


    # calculate and store model complexity
    model_complexity = p * (0.35) + l * (0.35) + n * (0.2) + t * (0.1)
    sol.metrics['model_complexity'] = model_complexity

    return model_complexity


@profile
def resource_utilization(sol):
    '''
    purpose: objective function to minimize how much it costs to run the model
    TODO: complete formulas on solution.py
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: resource utilization metric from sol.metrics
    '''
    # extract cost metrics
    development_time = sol.metrics['development_time']
    cpu = sol.metrics['cpu_util_percent']
    ram = sol.metrics['ram_usage_mb']
    throughput = sol.metrics['throughput']
    latency = sol.metrics['latency']

    # normalize model metrics based on min-max heuristics to be between 0 and 1
    time = (development_time - 1) / (600 - 1)
    cpu = (cpu - 0) / (100 - 0)
    ram = (ram - 0) / (1000 - 0)
    put = 1 - ((throughput - 1) / (10000 - 1))
    lat = (latency - 1) / (10 - 1)

    # store normalized and original metrics
    sol.metrics['normalized_development_time'] = time
    sol.metrics['normalized_cpu'] = cpu
    sol.metrics['normalized_ram'] = ram
    sol.metrics['normalized_throughput_complement'] = put
    sol.metrics['normalized_latency'] = lat

    # calculate and store resource utilization
    resource_utilization = time * 0.5  + put * 0.25 + lat * 0.25                   # TODO: add cpu and ram back in cpu * 0.05 + ram * 0.1
    sol.metrics['resource_utilization'] = resource_utilization

    return resource_utilization


