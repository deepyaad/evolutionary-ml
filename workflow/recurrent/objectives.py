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
    recall_tprs = [sol.metrics[lang+'_recall'] for lang in sol.configuration['labels_inorder']]
    fprs = [sol.metrics[lang+'_false_pos_rate'] for lang in sol.configuration['labels_inorder']]

    # calculate and store unfairness
    unfairness = max(recall_tprs) - min(recall_tprs)
    sol.metrics['unfairness'] = unfairness

    # find the most over-predicted and under-predicted classes
    highest_tpr_class = recall_tprs.index(max(recall_tprs))
    lowest_tpr_class = recall_tprs.index(min(recall_tprs))
    highest_fpr_class = fprs.index(max(fprs))
    lowest_fpr_class = fprs.index(min(fprs))
    
    sol.metrics['highest_tpr_class'] = highest_tpr_class
    sol.metrics['lowest_tpr_class'] = lowest_tpr_class
    sol.metrics['highest_fpr_class'] = highest_fpr_class
    sol.metrics['lowest_fpr_class'] = lowest_fpr_class

    return unfairness


@profile
def misclassification(sol):
    '''
    purpose: objective function to minimize misclassification
    misclassification is calculated using accuracy, f1 score, and loss
    params:
        sol: Solution object containing hyperparameters, model, and metrics
    returns: misclassification metric from sol.metrics
    '''
    # get macro metrics for each class
    # precision_macro = sol.metrics['precision_macro']
    # recall_macro = sol.metrics['recall_macro']
    f1_macro = sol.metrics['f1_macro']
    accuracy_macro = sol.metrics['accuracy_macro']

    # normalize loss to be between 0 and 1
    loss = sol.metrics['loss']
    normalized_loss = (loss - 0) / (999 - 0)
    sol.metrics['normalized_loss'] = normalized_loss

    # calculate and store predictive performance
    misclassification = (1 - f1_macro) * .45 + (1 - accuracy_macro) * .2 + normalized_loss * .35
    sol.metrics['misclassification'] = misclassification

    return misclassification


@profile
def complexity(sol):
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
    complexity = p * (0.4) + l * (0.3) + n * (0.2) + t * (0.1)
    sol.metrics['complexity'] = complexity

    return complexity


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
    throughput = sol.metrics['throughput']
    latency = sol.metrics['latency']

    # normalize model metrics based on min-max heuristics to be between 0 and 1
    time = (development_time - 1) / (600 - 1)
    put = 1 - ((throughput - 1) / (10000 - 1))
    lat = (latency - 0.01) / (1 - 0.01)

    # store normalized and original metrics
    sol.metrics['normalized_development_time'] = time
    sol.metrics['normalized_throughput_complement'] = put
    sol.metrics['normalized_latency'] = lat

    # calculate and store resource utilization
    resource_utilization = time * 0.5  + put * 0.25 + lat * 0.25
    sol.metrics['resource_utilization'] = resource_utilization

    return resource_utilization


