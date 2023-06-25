def information(d):
    # d is a dictionary of settings
    print(f"the problem type is: {d['problem']}")
    print(f"this file uses the {d['approach']} with {d['method']} using {d['algorithm']}")
    print(f"the goal is to {d['goal']}")
    print(f"the inputs are: {d['input']}; they are {d['input type']} that represent {d['input meaning']}")
    print(f"the outputs are: {d['output']}; they are {d['output type']} that represent {d['output meaning']}")
    return


siamese_settings_dictionary = {
    'problem'           : "regression",
    'approach'          : "few-shot learning",
    'method'            : "non-parametric",
    'algorithm'         : "siamese network",
    'goal'              : "learn a distribution using few samples from it",
    'input'             : "samples from a distribution",
    'input type'        : "vectors",
    'input meaning'     : "spectrum", 
    'output'            : "samples from a distribution",
    'output type'       : "one number",
    'output meaning'    : "temperature or pressure, depending on distribution",
    'number of ways'    : 2,
    'number of shots'   : 1,
    'number of folds'   : 8,
    'support-query ratio': 0.8,
    'task size'         : 5
}