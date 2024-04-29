configs = [
    # Config 1. Takes around 7 seconds on CPU.
    {
        'METHOD': 'Numpy',
        'PLOT_GRAPHS': True,
        'NUM_FEATURES' : 50,        
        'NUM_FACTORS' : 12,         
        'NUM_SAMPLES' : 20,      
        'NUM_ITERATIONS' : 1000,
        'LOG_FREQ' : 1,
        'RANDOM_SEED' : 1,
    },

    # Config 2. Takes around 11 seconds on CPU.
    {
        'METHOD': 'Standard',
        'PLOT_GRAPHS': True,
        'NUM_FEATURES' : 50,        
        'NUM_FACTORS' : 12,         
        'NUM_SAMPLES' : 20,      
        'NUM_ITERATIONS' : 1000,
        'LOG_FREQ' : 1,
        'RANDOM_SEED' : 1,
    },

    # Config 3. Takes around 60 seconds on CPU. Takes
    # too long time to run on other imlementations.
    {
        'METHOD': 'Vectorised',
        'PLOT_GRAPHS': True,
        'NUM_FEATURES' : 2000,        
        'NUM_FACTORS' : 90,         
        'NUM_SAMPLES' : 570,      
        'NUM_ITERATIONS' : 120,
        'LOG_FREQ' : 1,
        'RANDOM_SEED' : 1,
    }
]