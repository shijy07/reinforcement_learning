import matplotlib.pyplot as plt


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])

    plt.xlabel("Time Step")
    plt.ylabel("Cumulate Regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.k)
    plt.legend()
    plt.show()