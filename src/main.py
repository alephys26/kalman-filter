from experiments import experiments, run_experiment

for i, experiment in enumerate(experiments):
    experiment_name = f"experiment_{i+1}_{experiment['acceleration_pattern']}"
    print(
        f"\nRunning Experiment {i+1} with {experiment['acceleration_pattern']} acceleration pattern")
    run_experiment(
        T=experiment["T"],
        acceleration_pattern=experiment["acceleration_pattern"],
        measurement_noise_cov=experiment["measurement_noise_cov"],
        experiment_name=experiment_name
    )
