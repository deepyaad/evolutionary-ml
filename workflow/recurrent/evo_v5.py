"""
@authors: John Rachlin and Ananda Francis
@file: evo_v5.py: An evolutionary computing framework (version 5) adapted for neural architecture search.
Assumes no Solutions class.
"""

import random as rnd
import copy
from functools import reduce
import pickle
import time
import os
import json
from profiler import profile
from solution import Solution
from mutators import create_layer


class Environment:

    # Store all solutions generated and versions of Pareto set
    ARCHIVE_FILE = '../../outputs/recurrent/all_solutions.jsonl'
    EVO_FINAL_PARETO_FILE = '../../outputs/recurrent/evo_final_pareto.dat' 
    FINAL_VALIDATED_FILE = '../../outputs/recurrent/evo_final_pareto_validated.dat'
    HISTORICAL_VALIDATED_FILE = '../../outputs/recurrent/pareto_historical_validated.dat'


    def __init__(self):
        """ Population constructor """
        self.pop = {} # The solution population eval -> solution
        self.fitness = {} # Registered fitness functions: name -> objective function
        self.agents = {}  # Registered agents:  name -> (operator, num_solutions_input)
        self.data = None # Store the dataset (train/test/val)
        self.class_count = None # Store the number of classes
        self.feature_count = None

        # Clear previous files for new run
        for file_path in [self.ARCHIVE_FILE, self.EVO_FINAL_PARETO_FILE,
                         self.FINAL_VALIDATED_FILE, self.HISTORICAL_VALIDATED_FILE]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def size(self):
        """ The size of the current population """
        return len(self.pop)

    def add_fitness_criteria(self, name, f):
        """ Register a fitness criterion (objective) with the
        environment. Any solution added to the environment is scored
        according to this objective """
        self.fitness[name] = f

    def add_agent(self, name, op):
        """ Register a named agent with the population.
        The operator (op) function defines what the agent does.
        k defines the number of solutions the agent operates on. """
        self.agents[name] = op

    def add_dataset(self, data, class_count=None, feature_shape=None):
        """ 
        Add a dataset to the environment.
        Supports both legacy dict format and LazyDataLoader.
        
        For memory efficiency, use LazyDataLoader which loads features on-demand.
        This is especially important when evolving feature extraction methods,
        as you don't want all features loaded into memory at once.
        """
        from data_loader import LazyDataLoader
        
        self.data = data
        
        if isinstance(data, LazyDataLoader):
            # Extract metadata from LazyDataLoader
            self.class_count = data.class_count
            # feature_shape will be determined per-feature-type
            # Use first available feature as default
            if feature_shape is None:
                available_features = [f for f in LazyDataLoader.AVAILABLE_FEATURES 
                                    if data.is_feature_available(f)]
                if available_features:
                    self.feature_shape = data.get_feature_shape(available_features[0])
                else:
                    raise ValueError("No available features in LazyDataLoader")
            else:
                self.feature_shape = feature_shape
        else:
            # Legacy format
            if class_count is None or feature_shape is None:
                raise ValueError("class_count and feature_shape required for legacy data format")
            self.class_count = class_count
            self.feature_shape = feature_shape


    def get_random_solution(self):
        
        """ Pick random solution from the population """
        from data_loader import LazyDataLoader

        if self.size() == 0: # No solutions in population
            # create random configuration
            # Select feature based on what's available
            if isinstance(self.data, LazyDataLoader):
                available_features = [f for f in LazyDataLoader.AVAILABLE_FEATURES 
                                    if self.data.is_feature_available(f)]
                if not available_features:
                    raise ValueError("No available features in LazyDataLoader")
                feature_selection = rnd.choice(available_features)
                # Get feature shape for selected feature
                feature_shape = self.data.get_feature_shape(feature_selection)
            else:
                # Legacy format
                feature_selection = rnd.choice([
                    'stft', 'mel_specs', 'mfccs', 'ast', 
                    'clap', 'granite_speech', 'mctct', 'parakeet', 
                    'seamlessM4T', 'speech2Text', 'univNet', 'whisper'
                ])
                feature_shape = self.feature_shape
            
            hidden_layer_count = rnd.randint(2, 4)
            hidden_layers, layer_names, specifications, outputs = [], [], [], []
            input_size = feature_shape
            for _ in range(hidden_layer_count):

                # create hidden layers
                if _ == hidden_layer_count - 1:
                    # create last hidden layer
                    valid_last_layers = ['LSTM', 'SimpleRNN', 'GRU','GlobalAveragePooling1D', 'GlobalMaxPooling1D']
                    last_hidden_layer, name, specs, output_size = create_layer(input_size, valid_last_layers, last_layer=True)

                    # add last hidden layer to list of hidden layers
                    hidden_layers.append(last_hidden_layer)
                    layer_names.append(name)
                    specifications.append(specs)
                    outputs.append(output_size)

                else:
                    # intialize models with only recurrent layers
                    layer, name, specs, output_size = create_layer(input_size, ['LSTM', 'SimpleRNN', 'GRU'], last_layer=False)
                    hidden_layers.append(layer)
                    layer_names.append(name)
                    specifications.append(specs)
                    outputs.append(output_size)
                    input_size = output_size

            configuration = {

            # architecture specifications
            'hidden_layer_count': hidden_layer_count, 
            'hidden_layers': hidden_layers,
            'layer_names': layer_names,
            'layer_specifications': specifications,
            'neurons_per_layer': outputs,

            # hyperparameters
            'loss_function': rnd.choice(['categorical_crossentropy', 'categorical_focal_crossentropy', 'kl_divergence']),
            'optimizer': rnd.choice([
                'adamax', 'adadelta','ftrl', 'lamb',  'nadam', 'rmsprop', 
                'sgd', 'adagrad',  'lion',  'adamw', 'adafactor', 'adam'                                
            ]),

            'epochs': rnd.randint(3,32),
            'batch_size': rnd.randint(32, 512),

            # input data specifications
            'input_size': input_size,
            'output_size': self.class_count,
            'feature_shape': feature_shape,
            'class_count': self.class_count,
            'labels_inorder': self.data.labels_inorder if isinstance(self.data, LazyDataLoader) else self.data['labels_inorder'],
            'feature_selection': feature_selection

            }
        
            sol = Solution(configuration)
            sol.develop_model(self.data)
            return sol

        else:
            popvals = tuple(self.pop.values())
            return copy.deepcopy(rnd.choice(popvals))
        
    @profile
    def run_agent(self, name):
        """ Invoke an agent against the population """
        op = self.agents[name]
        random_solution = self.get_random_solution()
        new_solution = op(random_solution, self.data)
        self.add_solution(new_solution)

    @staticmethod
    def _archive_solution(sol):
        """Saves a single solution to the archive file using JSON Lines."""

        # get Solution dictionary
        sol_dict = sol.to_dict()

        # open the file in append mode ('a')
        with open(Environment.ARCHIVE_FILE, 'a') as f:
            json_line = json.dumps(sol_dict)
            f.write(json_line + '\n')

    def add_solution(self, sol):
        """ Add a solution to the population and archive it. """
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[eval] = sol
        Environment._archive_solution(sol) # archive solution for global tracking


    @profile
    def evolve(self, n=1, dom=100, viol=1000, status=1000, sync=1000, time_limit=None, reset=False, historical_pareto=False):
        """ Run n random agents (default=1)
        dom defines how often we remove dominated (unfit) solutions
        status defines how often we display the current population

        n = # of agent invocations
        dom = interval for removing dominated solutions
        viol = interval for removing solutions that violate user-defined upper limits
        status = interval for display the current population
        sync = interval for merging results with solutions.dat (for parallel invocation)
        time_limit = the evolution time limit (seconds).  Evolve function stops when limit reached
        historical_pareto = calculating the historical pareto optimal solutions (takes long to run)
        """

        # Initialize solutions file
        if reset and os.path.exists('../../outputs/recurrent/solutions.dat'):
            os.remove('../../outputs/recurrent/solutions.dat')

        # Initialize user constraints
        if reset or not os.path.exists('../../outputs/recurrent/constraints.json'):
            with open('../../outputs/recurrent/constraints.json', 'w') as f:
                json.dump({name:1 for name in self.fitness}, f, indent=4)

        start = time.time_ns()
        elapsed = (time.time_ns() - start) / 10**9
        agent_names = list(self.agents.keys())

        i = 0
        
        while i < n and self.size() > 0 and (time_limit is None or elapsed < time_limit):
            print('\n\n', f'ROUND {i} IN EVOLUTION with {self.size()} SOLUTIONS', '\n')
            pick = rnd.choice(agent_names)
            self.run_agent(pick)

            if i % sync == 0:
                try:
                    # Merge saved solutions into population
                    with open('../../outputs/recurrent/solutions.dat', 'rb') as file:
                        loaded = pickle.load(file)
                        for eval, sol in loaded.items():
                            self.pop[eval] = sol
                except Exception as e:
                    print(e)

                # Remove the dominated solutions
                self.remove_dominated()

                # Resave the non-dominated solutions
                with open('../../outputs/recurrent/solutions.dat', 'wb') as file:
                    pickle.dump(self.pop, file)


            if i % dom == 0:
                self.remove_dominated()

            if i % viol == 0:
                self.remove_constraint_violators()

            if i % status == 0:
                self.remove_dominated()
                
                # Memory management: clear feature cache periodically if using LazyDataLoader
                from data_loader import LazyDataLoader
                if isinstance(self.data, LazyDataLoader) and i % (status * 5) == 0:
                    print("Clearing feature cache to free memory...")
                    self.data.clear_cache()
                    import gc
                    gc.collect()

                print(self)
                print("Iteration          :", i)
                print("Population size    :", self.size())
                print("Elapsed Time (Sec) :", elapsed)
                if elapsed > 0:
                    print(f"Solutions per Sec  : {i / elapsed:.2f}")
                print("\n\n\n\n")

            i += 1
            elapsed = (time.time_ns() - start) / 10**9

        # Clean up the population
        print("Total elapsed time (sec): ", round(elapsed,4))

        # Save Pareto Set 1: Final Evolutionary Pareto Set
        self.remove_dominated()
        with open(self.EVO_FINAL_PARETO_FILE, 'wb') as file:
            pickle.dump(self.pop, file)
        print(f"Saved Final Pareto Set Post-Evolution to: {self.EVO_FINAL_PARETO_FILE}")
        print(f"Set Size: {len(self.pop)}")

        # ensure validation runs AFTER evolution is complete
        print("\nEvolution complete. Starting validation process.\n\n")
        self.create_validation_pareto_sets(historical_pareto)
    
    
    def create_validation_pareto_sets(self, historical_pareto):
        """
        Creates the two validation-based Pareto sets:
        - Set 1: Final evolutionary solutions re-evaluated on validation data
        - Set 2: All historical solutions filtered by training dominance, then validated
        """
        print("Creating validation-based Pareto sets...")

        # Pareto Set 1: Final evolutionary solutions validated
        self._create_final_validated_pareto()
        
        # Pareto Set 2: Historical solutions validated
        if historical_pareto:
            self._create_historical_validated_pareto()

    def _create_final_validated_pareto(self):
        """Take final evolutionary solutions, validate them, and re-filter"""
        
        print(f"Validating {len(self.pop)} solutions from final evolutionary set...")
        
        validated_solutions = {}
        for eval, sol in self.pop.items():
            
            # Validate on validation data - this should NOT overwrite training metrics
            sol.validate_model(self.data)
            
            # Create new evaluation using validation metrics
            new_eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
            validated_solutions[new_eval] = sol
        
        # Re-do non-domination filtering using VALIDATION metrics
        self.pop = validated_solutions
        self.remove_dominated()
        
        # Save Validated Pareto Set 1
        with open(self.FINAL_VALIDATED_FILE, 'wb') as file:
            pickle.dump(self.pop, file)
        print(f"Saved Evolutionary Valid Pareto Set 1 to: {self.FINAL_VALIDATED_FILE}")
        print(f"Set size: {len(self.pop)}", "\n\n\n\n")

    def _create_historical_validated_pareto(self):
        """Find non-dominated from ALL historical using testing metrics, then validate and re-filter"""
        
        # Step 1: Load all historical solutions and find non-dominated using TRAINING metrics
        self.pop = self._load_all_historical_solutions()
        print(f"Loaded {len(self.pop)} total historical solutions")
        
        # Find non-dominated set using original training metrics
        self.remove_dominated()
        print(f"Found {len(self.pop)} non-dominated solutions using training metrics")
        
        # Step 2: Validate these historically non-dominated solutions
        validated_solutions = {}
        for eval in self.pop:
            sol = self.pop[eval]
            
            # Validate on validation data
            sol.validate_model(self.data)
            
            # Create new evaluation using validation metrics
            new_eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
            validated_solutions[new_eval] = sol
        
        # Step 3: Re-do non-domination filtering using VALIDATION metrics
        self.pop = validated_solutions
        self.remove_dominated()
        
        # Save VALIDATED Pareto Set 2
        with open(self.HISTORICAL_VALIDATED_FILE, 'wb') as file:
            pickle.dump(self.pop, file)
        print(f"Saved Historical Validated Pareto Set to: {self.HISTORICAL_VALIDATED_FILE}")
        print(f"Set size: {len(self.pop)}")

    def _load_all_historical_solutions(self):
        """Load all solutions from the JSONL archive"""
        all_solutions = {}
        
        if not os.path.exists(self.ARCHIVE_FILE):
            print(f"Archive file {self.ARCHIVE_FILE} not found!")
            return all_solutions
            
        with open(self.ARCHIVE_FILE, 'r') as f:
            for line in f:
                sol_dict = json.loads(line.strip())
                sol = Solution(hyperparams=sol_dict['hyperparams'], metrics=sol_dict['metrics'])
                
                # Use training metrics for initial dominance check
                eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
                all_solutions[eval] = sol
                
        return all_solutions


    @staticmethod
    def _dominates(p, q):
        """ p = evaluation of solution: ((obj1, score1), (obj2, score2), ... )"""
        pscores = [score for _,score in p]
        qscores = [score for _,score in q]
        score_diffs = list(map(lambda x,y: y-x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0


    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Environment._dominates(p,q)}

    @profile
    def remove_dominated(self):
        """ Remove dominated solutions """
        nds = reduce(Environment._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k:self.pop[k] for k in nds}

    @staticmethod
    def _reduce_viol(S, T):
        objective, max_value = T
        return S - {q for q in S if dict(q)[objective] >= max_value}

    @profile
    def remove_constraint_violators(self):
        """ Remove solutions whose objective values exceed one or
        more user-defined constraints as listed in constraints.dat """

        # Read the latest constraints file into a dictionary
        with open('../../outputs/recurrent/constraints.json', 'r') as f:
            limits = json.load(f)

        # Determine non-violators and update population
        nonviol = reduce(Environment._reduce_viol, limits.items(), self.pop.keys())
        self.pop = {k:self.pop[k] for k in nonviol}


    def summarize(self, with_details=False, source=""):
        header = ",".join(self.fitness.keys())
        if source != "":
            header = "groupname,"+header
        print(header)

        for eval in self.pop:
            vals = ",".join([str(score) for _,score in eval])
            if source != "":
                vals = source + "," + vals
                print(vals)

        if with_details:
            counter = 0
            for eval, sol in self.pop.items():
                counter += 1
                print(f"\n\nSOLUTION {counter}")
                for objective, score in eval:
                    print(f"{objective:15}: {score}")
                print(str(sol))
                #print(pd.DataFrame(sol))

    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval,sol in self.pop.items():
            # round scores to 3 decimal places for readability
            rounded = {k: round(v, 3) for k, v in dict(eval).items()}
            rslt += str(rounded)+"\n" # +str(sol)+"\n"           

        return rslt

