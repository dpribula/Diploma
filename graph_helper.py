import matplotlib.pyplot as plt


def show_graph(graph_rmse_train, graph_rmse_test):
    plt.plot(graph_rmse_test, label='test_RMSE',color='blue')
    plt.plot(graph_rmse_train, label='train_RMSE',color='green')
    plt.ylabel('RMSE')
    plt.gca().set_ylim([0, 0.7])
    plt.show()