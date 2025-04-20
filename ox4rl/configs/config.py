from yacs.config import CfgNode

# dirname = os.checkpointdir.basename(os.checkpointdir.dirname(os.checkpointdir.abspath(__file__)))

cfg = CfgNode({
    'seed': 8848,
    'exp_name': '',
    'model': 'SPACE',
    'flow': False,
    # Resume training or not
    'resume': True,
    # If resume is true, then we load this checkpoint. If '', we load the last checkpoint
    'resume_ckpt': '',
    # Whether to use multiple GPUs
    'parallel': False,
    'resolution': (128, 128),
    # Device ids to use
    'device_ids': [0, 1],
    'device': 'cuda:0',
    'logdir': '../output/logs/',
    'checkpointdir': '../output/checkpoints/',
    'latentsdir': '../output/latents',
    'evaldir': '../output/eval/',
    'demodir': '../output/demo/',
    'save_relevant_objects': False,
    # Dataset to use
    'dataset': 'OBJ3D_LARGE',
    'dataset_style': 'space_like',

    'dataset_roots': {
        'ATARI': '../data/ATARI',
        'OBJ3D_LARGE': '../data/OBJ3D_LARGE',
        'OBJ3D_SMALL': '../data/OBJ3D_SMALL',
    },

    # For Atari
    'gamelist': [
        'Atlantis-v0',
        'Asterix-v0',
        'Carnival-v0',
        'DoubleDunk-v0',
        'Kangaroo-v0',
        'MontezumaRevenge-v0',
        'MsPacman-v0',
        'Pooyan-v0',
        'Qbert-v0',
        'SpaceInvaders-v0',
        'Pong-v0',
        'Tennis-v0',
    ],


    # For train_model
    'train': {
        'log': True,
        'batch_size': 16,
        'max_epochs': 1000,
        'max_steps': 1000000,

        'num_workers': 4,
        # Gradient clipping. If 0.0 we don't clip
        'clip_norm': 1.0,
        'max_ckpt': 5,

        'print_every': 500,
        'save_every': 1000,
        'eval_on': True,
        'eval_every': 1000,
        'log_latents': True,
        'log_latents_every': 500,
        
        # Loss weights
        'weight_mask': 1.0,    # Weight for motion loss
        'weight_oc': 1.0,      # Weight for object consistency loss
        'weight_temporal': 0.5, # Weight for temporal consistency loss

        'solver': {
            'slot': {
                'optim': 'Adam',
                'lr': 1e-4
            },
            'fg': {
                'optim': 'RMSprop',
                'lr': 1e-5,
            },
            'bg': {
                'optim': 'Adam',
                'lr': 1e-3,
            }
        },
        'black_background': False,  
        'dilation': False,
    },

    # for validation in train_model
    'validation': {
        # What to evaluate
        'metrics': ['ap','mse','cluster'],

        'num_samples': {
            'mse': 4,
        }
    },

    # for testing in eval_model
    'test': {
        # What to evaluate
        'metrics': ['cluster', 'mse', 'ap'],

        'num_samples': {
            'mse': 4,
        }
    },

    # For eval_model
    'eval': {
        # One of 'best', 'last'
        'checkpoint': 'best',
        # Manually specify the eval checkpoint
        'eval_ckpt': '', 
        # Either 'ap_dot5' or 'ap_avg'
        'metric': 'ap_avg',
        'use_precomputed_latents': False,  # Set to true to use pre-computed latents
        'step': 0
    },

    # For engine.show_images
    'show': {
        # Either 'validation' or 'test'
        'mode': 'validation',
        # Indices into the dataset
        'indices': [0]
    }
})

from ox4rl.configs.space_cfg import space_cfg
from ox4rl.configs.moc_cfg import moc_cfg
from ox4rl.configs.dataset_size_cfg import dataset_size_cfg
from ox4rl.configs.classifier_cfg import classifier_cfg
from ox4rl.models.slot.arch import arch_slot

# For these three, please go to the corresponding file
cfg.moc_cfg = moc_cfg
cfg.space_cfg = space_cfg
cfg.dataset_size_cfg = dataset_size_cfg
cfg.classifier = classifier_cfg
cfg.arch_slot = arch_slot
