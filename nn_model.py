import abc


class nn_model(abc.ABC):
    @abc.abstractmethod
    def run_model_train(self, batch_x, batch_y, batch_target_x, batch_target_y, batch_seq, session):
        pass
    @abc.abstractmethod
    def run_model_test(self, batch_x, batch_y, batch_seq):
        pass
    @abc.abstractmethod
    def save_model(self):
        pass
