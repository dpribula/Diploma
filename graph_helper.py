import matplotlib.pyplot as plt

def show_graph(graph_rmse_train, graph_rmse_test):
    plt.plot(graph_rmse_test, label='test_RMSE',color='blue')
    plt.plot(graph_rmse_train, label='train_RMSE',color='green')
    plt.plot([0 for _ in range(0,len(graph_rmse_test))],color='red',linewidth=4)
    plt.ylabel('Training')
    plt.gca().set_ylim([0,0.51])
    plt.show()