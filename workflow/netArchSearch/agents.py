from solution import Solution
import random as rnd
from profiler import profile

@profile
def add_layer(sol, data):
  '''
  purpose: agent to add a layer to the neural network
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]

  # update hyperparamters to add a new layer
  activation = rnd.choice(['relu', 'tanh', 'sigmoid'])
  new_activation_list = sol.hyperparams['activation_per_hidden_layer']
  new_activation_list.append(activation)

  new_units_list = sol.hyperparams['units_per_hidden_layer']
  units = rnd.randint(16, 128)
  new_units_list.append(units)

  new_optimizer = sol.hyperparams['optimizer']
  new_hidden_layer_count = sol.hyperparams['hidden_layer_count'] + 1
  new_epochs = sol.hyperparams['epochs']
  new_batch_size = sol.hyperparams['batch_size']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_loss_function = sol.hyperparams['loss_function']

  new_hyperparams = {
      'activation_per_hidden_layer': new_activation_list,
      'optimizer': new_optimizer,
      'units_per_hidden_layer': new_units_list,
      'hidden_layer_count': new_hidden_layer_count,
      'epochs': new_epochs,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }

  # generate new solution
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)

  return new_sol

@profile
def remove_layer(sol, data):
  '''
  purpose: agent to remove a layer from the neural network
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]

  if len(sol.model.layers) <= 2:
    return sol

  # update hyperparamters to remove a random hidden layer
  pick = rnd.randint(1, sol.hyperparams['hidden_layer_count'] - 1)
  new_activation_list = sol.hyperparams['activation_per_hidden_layer']
  new_activation_list.pop(pick)

  new_units_list = sol.hyperparams['units_per_hidden_layer']
  new_units_list.pop(pick)

  new_optimizer = sol.hyperparams['optimizer']
  new_hidden_layer_count = sol.hyperparams['hidden_layer_count'] - 1
  new_epochs = sol.hyperparams['epochs']
  new_batch_size = sol.hyperparams['batch_size']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_loss_function = sol.hyperparams['loss_function']

  # generate new solution
  new_hyperparams = {
      'activation_per_hidden_layer': new_activation_list,
      'optimizer':  new_optimizer,
      'units_per_hidden_layer': new_units_list,
      'hidden_layer_count': new_hidden_layer_count,
      'epochs': new_epochs,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)

  return new_sol

@profile
def shrink_layer(sol, data):
  '''purpose: agent to shrink the number of nodes in a random layer
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]

  if len(sol.model.layers) <= 2:
    return sol

  # update hyperparamters to shrink a random layer (aside from the last one)
  pick = rnd.randint(1, sol.hyperparams['hidden_layer_count'] - 1)
  current_units = sol.model.layers[pick].units

  if current_units <= 2:
    return sol # cannot reduce further

  reduced_units = rnd.randint(2, current_units - 1)

  # update hyperparams to match the change
  new_units = sol.hyperparams['units_per_hidden_layer'].copy()
  new_units[pick] = reduced_units
  new_activation = sol.hyperparams['activation_per_hidden_layer'].copy()
  new_optimizer = sol.hyperparams['optimizer']

  new_hidden_layer_count = sol.hyperparams['hidden_layer_count']
  new_epochs = sol.hyperparams['epochs']
  new_batch_size = sol.hyperparams['batch_size']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_loss_function = sol.hyperparams['loss_function']

  new_hyperparams = {
      'activation_per_hidden_layer': new_activation,
      'optimizer': new_optimizer,
      'units_per_hidden_layer': new_units,
      'hidden_layer_count': new_hidden_layer_count,
      'epochs': new_epochs,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)
  return new_sol

@profile
def grow_layer(sol, data):
  '''
  purpose: agent to grow the number of nodes in a random layer
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]

  if len(sol.model.layers) <= 2:
      return sol

  # update hyperparamters to grow a random layer (aside from the last one)
  pick = rnd.randint(1, len(sol.model.layers) - 2)
  current_units = sol.model.layers[pick].units
  increased_units = rnd.randint(current_units + 1, current_units * 2)
  new_units = sol.hyperparams['units_per_hidden_layer'].copy()
  new_units[pick] = increased_units
  new_activation = sol.hyperparams['activation_per_hidden_layer'].copy()
  new_optimizer = sol.hyperparams['optimizer']

  new_hidden_layer_count = sol.hyperparams['hidden_layer_count']
  new_epochs = sol.hyperparams['epochs']
  new_batch_size = sol.hyperparams['batch_size']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_loss_function = sol.hyperparams['loss_function']

  # generate new solution
  new_hyperparams = {
      'activation_per_hidden_layer': new_activation,
      'optimizer': new_optimizer,
      'units_per_hidden_layer': new_units,
      'hidden_layer_count': new_hidden_layer_count,
      'epochs': new_epochs,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)

  return new_sol

@profile
def change_activation(sol, data):
  '''
  purpose: agent to change the activation function of a random layer
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]

  if len(sol.model.layers) <= 2:
        return sol

  # update hyperparameters to change the activation of a random layer (aside from the last one)
  picks = [rnd.randint(1, len(sol.model.layers) - 2) for _ in range(rnd.randint(1,5))]
  for pick in picks:
    current_activation = sol.model.layers[pick].activation.__name__ # ensure that i am printing out the string and not function object
    options = ['relu', 'tanh', 'sigmoid']
    options.remove(current_activation)
    activation = rnd.choice(options)
    new_activation = sol.hyperparams['activation_per_hidden_layer'].copy()
    new_activation[pick] = activation
  
  new_units = sol.hyperparams['units_per_hidden_layer'].copy()
  new_optimizer = sol.hyperparams['optimizer']

  new_hidden_layer_count = sol.hyperparams['hidden_layer_count']
  new_epochs = sol.hyperparams['epochs']
  new_batch_size = sol.hyperparams['batch_size']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_loss_function = sol.hyperparams['loss_function']

  # generate new solution
  new_hyperparams = {
      'activation_per_hidden_layer': new_activation,
      'optimizer': new_optimizer,
      'units_per_hidden_layer': new_units,
      'hidden_layer_count': new_hidden_layer_count,
      'epochs': new_epochs,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)

  return new_sol

@profile
def change_optimizer(sol, data):
  '''
  purpose: agent to change the optimizer of the neural network
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]

  if len(sol.model.layers) <= 2:
        return sol

  # update hyperparameters to change the optimizer of the model
  options = ['adam', 'sgd', 'rmsprop']
  options.remove(sol.hyperparams['optimizer'])
  new_optimizer = rnd.choice(options)
  new_activation = sol.hyperparams['activation_per_hidden_layer'].copy()
  new_units = sol.hyperparams['units_per_hidden_layer'].copy()

  new_hidden_layer_count = sol.hyperparams['hidden_layer_count']
  new_epochs = sol.hyperparams['epochs']
  new_batch_size = sol.hyperparams['batch_size']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_loss_function = sol.hyperparams['loss_function']

  # generate new solution
  new_hyperparams = {
      'activation_per_hidden_layer': new_activation,
      'optimizer': new_optimizer,
      'units_per_hidden_layer': new_units,
      'hidden_layer_count': new_hidden_layer_count,
      'epochs': new_epochs,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)

  return new_sol

@profile
def change_units_per_layer(sol, data):
  '''
  purpose: agent to change the number of units in random layers
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]

  if len(sol.model.layers) <= 2:
    return sol

  # update hyperparamters to change the number of units in a random layer (aside from the last one)
  picks = [rnd.randint(1, len(sol.model.layers) - 2) for _ in range(rnd.randint(1,5))]
  for pick in picks:
    current_units = sol.model.layers[pick].units
    options = list(range(16, 129))
    options.remove(current_units)
    new_units_count = rnd.choice(options)

    # update hyperparams to match the change
    new_units = sol.hyperparams['units_per_hidden_layer'].copy()
    new_units[pick] = new_units_count
  
  new_activation = sol.hyperparams['activation_per_hidden_layer'].copy()
  new_optimizer = sol.hyperparams['optimizer']

  new_hidden_layer_count = sol.hyperparams['hidden_layer_count']
  new_epochs = sol.hyperparams['epochs']
  new_batch_size = sol.hyperparams['batch_size']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_loss_function = sol.hyperparams['loss_function']

  new_hyperparams = {
      'activation_per_hidden_layer': new_activation,
      'optimizer': new_optimizer,
      'units_per_hidden_layer': new_units,
      'hidden_layer_count': new_hidden_layer_count,
      'epochs': new_epochs,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)
  return new_sol

@profile
def change_epochs(sol, data):
  '''
  purpose: agent to change the number of epochs for training
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]

  # update hyperparamters to change the number of epochs
  options = list(range(5, 21))
  options.remove(sol.hyperparams['epochs'])
  new_epochs = rnd.choice(options)

  new_activation = sol.hyperparams['activation_per_hidden_layer'].copy()
  new_units = sol.hyperparams['units_per_hidden_layer'].copy()
  new_optimizer = sol.hyperparams['optimizer']

  new_hidden_layer_count = sol.hyperparams['hidden_layer_count']
  new_batch_size = sol.hyperparams['batch_size']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_loss_function = sol.hyperparams['loss_function']

  new_hyperparams = {
      'activation_per_hidden_layer': new_activation,
      'optimizer': new_optimizer,
      'units_per_hidden_layer': new_units,
      'epochs': new_epochs,
      'hidden_layer_count': new_hidden_layer_count,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)

  return new_sol

@profile
def change_batch_size(sol, data):
  '''
  purpose: agent to change the batch size for training
  params:
    sol: Solution object containing hyperparameters, model, and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]  

  # update hyperparamters to change the batch size
  options = [16, 32, 64, 128]
  options.remove(sol.hyperparams['batch_size'])
  new_batch_size = rnd.choice(options)
  new_activation = sol.hyperparams['activation_per_hidden_layer'].copy()
  new_units = sol.hyperparams['units_per_hidden_layer'].copy()
  new_optimizer = sol.hyperparams['optimizer']
  new_hidden_layer_count = sol.hyperparams['hidden_layer_count']
  new_epochs = sol.hyperparams['epochs']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_loss_function = sol.hyperparams['loss_function']

  new_hyperparams = {
      'activation_per_hidden_layer': new_activation,
      'optimizer': new_optimizer,
      'units_per_hidden_layer': new_units,
      'hidden_layer_count': new_hidden_layer_count,
      'epochs': new_epochs,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)
  return new_sol

@profile
def change_loss_func(sol, data):
  '''
  purpose: agent to change the loss function of the model
  params:
    sol: Solution object containing hyperparameters, model and metrics
    data: data dictionary with training and testing splits
  returns: new Solution Object
  '''
  # ensure its not a list
  if type(sol) == list:
    sol = sol[0]

  # update hyperparamters to change the loss function
  options = ['binary_crossentropy', 'binary_focal_crossentropy', 'cosine_similarity']
  options.remove(sol.hyperparams['loss_function'])
  new_loss_function = rnd.choice(options)
  new_activation = sol.hyperparams['activation_per_hidden_layer'].copy()
  new_units = sol.hyperparams['units_per_hidden_layer'].copy()
  new_optimizer = sol.hyperparams['optimizer']
  new_hidden_layer_count = sol.hyperparams['hidden_layer_count']
  new_epochs = sol.hyperparams['epochs']
  new_feature_count = sol.hyperparams['feature_count']
  new_class_count = sol.hyperparams['class_count']
  new_batch_size = sol.hyperparams['batch_size']

  new_hyperparams = {
      'activation_per_hidden_layer': new_activation,
      'optimizer': new_optimizer,
      'units_per_hidden_layer': new_units,
      'hidden_layer_count': new_hidden_layer_count,
      'epochs': new_epochs,
      'batch_size': new_batch_size,
      'feature_count': new_feature_count,
      'class_count': new_class_count,
      'loss_function': new_loss_function
  }
  new_sol = Solution(new_hyperparams)
  new_sol.develop_model(data)
  return new_sol