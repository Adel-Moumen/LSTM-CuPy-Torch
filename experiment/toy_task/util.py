class Util:

    @staticmethod
    def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


    def print_setup(args):
        """
        Print experimental setup
        """
        print('\n\n\tExperimental setup')
        print('\t------------------\n')
        print('\tModel informations:')
        print(f'\tModel: {args.model}')
        print(f'\tNumber of layers: {args.num_layers}')
        print(f'\tHidden size: {args.hidden_size}')
        print(f'\tseed: {args.seed}')

        print('\tTraining informations:')
        print(f'\tBatch size: {args.batch_size}')
        print(f'\tLearning rate: {args.learning_rate}')
        print(f'\tseq length: {args.seq_length}')
        print('\t------------------\n')
        print('\tSTARTING TRAINING')
        print('\t------------------\n')
        print('\n\n')