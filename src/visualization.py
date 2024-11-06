import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_positions(true_positions, estimated_positions, experiment_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot3D(true_positions[:, 0], true_positions[:, 1],
              true_positions[:, 2], 'b-', alpha=0.5, label='True Position')

    ax.plot3D(estimated_positions[:, 0], estimated_positions[:, 1],
              estimated_positions[:, 2], 'r-', alpha=0.5, label='Estimated Position')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('True vs Estimated Position in 3D')
    ax.legend()

    plt.savefig(f"../results/{experiment_name}/true_vs_estimated_position.png")
    plt.close(fig)


def plot_errors(errors, experiment_name):
    plt.figure()
    plt.plot(range(1, len(errors) + 1), errors, 'r-',
             marker='o', label='Estimation Error')

    plt.xlabel('Time Step')
    plt.ylabel('Euclidean Error')
    plt.title('Estimation Error over Time')
    plt.legend()

    plt.savefig(f"../results/{experiment_name}/estimation_error.png")
    plt.close()
