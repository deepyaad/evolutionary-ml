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
import uuid
from parallel_evolution import RunPodEvolutionClient, convert_result_to_solution


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

    def add_dataset(self, data, class_count, feature_shape):
        """ Add a dataset to the environment """
        self.data = data
        self.class_count = class_count
        self.feature_shape = feature_shape


    def get_random_solution(self):
        
        """ Pick random solution from the population """

        if self.size() == 0: # No solutions in population
            # create random configuration
            hidden_layer_count = rnd.randint(2, 4)
            hidden_layers, layer_names, specifications, outputs = [], [], [], []
            input_size = self.feature_shape
            activations = [
                'celu', 'elu', 'gelu', 'hard_sigmoid', 'hard_shrink', 'hard_tanh', 'hard_silu', 
                'leaky_relu', 'linear', 'mish', 'relu', 'selu', 'silu', 
                'sigmoid', 'softmax', 'softplus', 'softsign', 'soft_shrink', 'swish', 'tanh',
                'tanh_shrink'
            ]
            rnn_archs = ['LSTM', 'SimpleRNN', 'GRU']
            for _ in range(hidden_layer_count):

                # randomize activation function and units
                activation = rnd.choice(activations)
                recurrent_activation = rnd.choice(activations)
                units = rnd.randint(8, 48)
                random_specs = {
                    'activation': activation,
                    'recurrent_activation': recurrent_activation,
                    'units': units
                }

                # create hidden layers
                if _ == hidden_layer_count - 1:

                    # create last hidden layer
                    last_hidden_layer, name, specs, output_size = create_layer(input_size, rnn_archs, last_layer=True, specs=random_specs)

                    # add last hidden layer to list of hidden layers
                    hidden_layers.append(last_hidden_layer)
                    layer_names.append(name)
                    specifications.append(specs)
                    outputs.append(output_size)

                else:
                    # intialize models with only recurrent layers
                    layer, name, specs, output_size = create_layer(input_size, rnn_archs, last_layer=False, specs=random_specs)
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
                'feature_shape': self.feature_shape,
                'class_count': self.class_count,
                'labels_inorder': self.data['labels_inorder'],

                # genetic information
                'id': uuid.uuid4(),
                'parent_id': 'start',
                'mutator': 'randomization'
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
        if name != 'crossover':
            random_solution = self.get_random_solution()
            new_solution = op(random_solution, self.data)
            self.add_solution(new_solution)

        # randomly select two DIFFERENT solutions from the population
        else:
            sol1 = self.get_random_solution()
            sol2 = self.get_random_solution()
            if sol1.configuration['id'] == sol2.configuration['id']:
                agent_names = list(self.agents.keys())
                pick = rnd.choice(agent_names)
                self.run_agent(pick)
            else:
                new_solution = op(sol1, sol2, self.data)
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
    
    @profile
    def evolve_parallel(self, batch_size=50, generations=None, dom=30, viol=10, status=30, 
                        time_limit=None, reset=False, historical_pareto=False,
                        runpod_endpoint_id=None, runpod_api_key=None, data_path='../../datasets/spotify_dataset.npz'):
        """
        Parallel evolution using RunPod serverless.
        
        Args:
            batch_size: Number of children to generate and train in parallel per generation
            generations: Number of generations to run (None = run until time_limit)
            dom: Interval for removing dominated solutions
            viol: Interval for removing constraint violators
            status: Interval for displaying population status
            time_limit: Maximum time to run evolution (seconds)
            reset: Whether to reset population files
            historical_pareto: Whether to calculate historical pareto
            runpod_endpoint_id: RunPod endpoint ID
            runpod_api_key: RunPod API key (or use RUNPOD_API_KEY env var)
            data_path: Path to dataset file
        """
        if runpod_endpoint_id is None:
            raise ValueError("runpod_endpoint_id must be provided for parallel evolution")
        
        # Initialize RunPod client
        client = RunPodEvolutionClient(runpod_endpoint_id, runpod_api_key)
        
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
        
        generation = 0
        total_solutions_generated = 0
        
        while (generations is None or generation < generations) and self.size() > 0 and (time_limit is None or elapsed < time_limit):
            print('\n\n', f'GENERATION {generation} IN PARALLEL EVOLUTION with {self.size()} SOLUTIONS', '\n')
            
            # Generate batch of children
            children = []
            for _ in range(batch_size):
                # Pick random agent and generate child
                pick = rnd.choice(agent_names)
                op = self.agents[pick]
                
                if pick != 'crossover':
                    random_solution = self.get_random_solution()
                    new_solution = op(random_solution, self.data)
                else:
                    sol1 = self.get_random_solution()
                    sol2 = self.get_random_solution()
                    if sol1.configuration['id'] == sol2.configuration['id']:
                        # Retry with different agent
                        pick = rnd.choice([a for a in agent_names if a != 'crossover'])
                        op = self.agents[pick]
                        random_solution = self.get_random_solution()
                        new_solution = op(random_solution, self.data)
                    else:
                        new_solution = op(sol1, sol2, self.data)
                
                children.append(new_solution)
            
            print(f"Generated {len(children)} children, dispatching to RunPod...")
            
            # Train all children in parallel
            results = client.train_solutions_parallel(children, data_path, timeout=3600)
            
            # Convert results back to solutions and add to population
            successful = 0
            for result in results:
                sol = convert_result_to_solution(result)
                if sol is not None:
                    self.add_solution(sol)
                    successful += 1
                else:
                    print(f"Failed to convert result to solution: {result.get('error', 'Unknown error')}")
            
            print(f"Successfully trained and added {successful}/{len(children)} solutions")
            total_solutions_generated += successful
            
            # Remove dominated solutions periodically
            if generation % dom == 0:
                self.remove_dominated()
            
            # Remove constraint violators periodically
            if generation % viol == 0:
                self.remove_constraint_violators()
            
            # Display status periodically
            if generation % status == 0:
                self.remove_dominated()
                
                print(self)
                print("Generation          :", generation)
                print("Population size    :", self.size())
                print("Total solutions    :", total_solutions_generated)
                print("Elapsed Time (Sec) :", elapsed)
                if elapsed > 0:
                    print(f"Solutions per Sec  : {total_solutions_generated / elapsed:.2f}")
                print("\n\n\n\n")
            
            generation += 1
            elapsed = (time.time_ns() - start) / 10**9
        
        # Clean up the population
        print("Total elapsed time (sec): ", round(elapsed, 4))
        print(f"Total solutions generated: {total_solutions_generated}")
        
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

