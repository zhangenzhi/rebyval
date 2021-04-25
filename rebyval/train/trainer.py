from rebyval.train.base_trainer import BaseTrainer


class TargetTrainer(BaseTrainer):
    def __init__(self, args):
        super(TargetTrainer, self).__init__(args=args)

    def during_train(self):
        pass

    def during_valid(self):
        pass

    def during_test(self):
        pass

class SurrogateTrainer(BaseTrainer):
    def __init__(self, args):
        super(SurrogateTrainer, self).__init__(args=args)

    def during_train(self):
        pass

    def during_valid(self):
        pass

    def during_test(self):
        pass
