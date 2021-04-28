default_values_dict ={ 

    'experiment': {
        'context': {
            'name': "None",
            'author': "None",
            'log_path': "./",
        },
        'main_loop_control': {
            'warmup_steps': {
                'target_model_samples': 15
            },
            'target_samples_per_iter': 3
        },
        'target_trainer': {
            'dataloader': {
                'name': "cifar10"
            },
            'model': {
                'name': "dnn",
                'deep_dims': "128,64,64,32",
                'activation_for_all': "relu,relu,relu,softmax",
                'restore_mode': None
                },
            'loss': None,
            'optimizer': {
                'name': 'SGD',
                'learning_rate': 0.01,
                'scheduler': None
            },
            'train_loop_control': {
                'train': {
                    'check_should_train': True,
                    'max_training_steps': 10000,
                    'max_training_epochs': 10
                },
                'valid': {
                    'check_should_valid': True,
                    'valid_gap': 1000,
                    'valid_steps': 100,
                    'save_model': None
                },
                'test': {
                    'check_should_test': True
                }

            }
        },
        'surrogate_trainer': {
            'dataloader': {
                'name': "dnn_weights",
                'path': "./dataset/weights_pool/dnn"
            },
            'model': {
                'name': "dnn",
                'deep_dims': "128,64,64,32",
                'activations': "relu,relu,relu,softmax",
                'restore_mode': None
            },
            'loss': None,
            'optimizer': {
                'name': 'SGD',
                'learning_rate': 0.01,
            },
            'train_loop_control': {
                'train': {
                    'check_should_train': True,
                    'max_training_steps': 10000,
                    'max_training_epochs': 10
                },
                'valid': {
                    'check_should_valid': True,
                    'valid_gap': 1000,
                    'valid_steps': 100,
                    'save_model': None
                },
                'test': {
                    'check_should_test': True
                }
            }
        } # surrogate model
    }# experiment
}# dict
