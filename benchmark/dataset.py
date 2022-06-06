class BenchmarkTrain:

    def __init__(self, X, Y):
        """
        :param X: tensor -> (seq_len, batch, input_size)
        :param Y: tensor -> (seq_len, batch, input_size+1)
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        """
        :param item: any given item that is an integer 
        return X & Y, item is the index 
        """
        X = self.X[item, :]
        Y = self.Y[item]
    
        return {
            "X" : X,
            "Y" : Y
        }