import numpy as np
from ocatari.ram.pong import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_PONG
from ocatari.ram.pong import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_PONG
from ocatari.ram.boxing import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_BOXING
from ocatari.ram.boxing import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_BOXING
from ocatari.ram.tennis import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_TENNIS
from ocatari.ram.tennis import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_TENNIS
from ocatari.ram.skiing import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_SKIING
from ocatari.ram.skiing import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_SKIING
from ocatari.ram.carnival import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_CARNIVAL
from ocatari.ram.carnival import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_CARNIVAL
from ocatari.ram.spaceinvaders import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_SPACE_INVADERS
from ocatari.ram.spaceinvaders import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_SPACE_INVADERS
from ocatari.ram.riverraid import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_RIVER_RAID
from ocatari.ram.riverraid import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_RIVER_RAID
from ocatari.ram.mspacman import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_MSPACMAN
from ocatari.ram.mspacman import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_MSPACMAN
from ocatari.ram.carnival import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_CARNIVAL
from ocatari.ram.carnival import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_CARNIVAL
from ocatari.ram.asterix import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_ASTERIX
from ocatari.ram.asterix import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_ASTERIX
from ocatari.ram.bowling import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_BOWLING
from ocatari.ram.bowling import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_BOWLING
from ocatari.ram.freeway import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_FREEWAY
from ocatari.ram.freeway import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_FREEWAY
from ocatari.ram.kangaroo import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_KANGAROO
from ocatari.ram.kangaroo import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_KANGAROO
from ocatari.ram.seaquest import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_SEAQUEST
from ocatari.ram.seaquest import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_SEAQUEST


no_label_str = "no_label"

label_list_carnival = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_CARNIVAL.keys()))
label_list_carnival_moving = sorted(list(MAX_NB_OBJECTS_MOVING_CARNIVAL.keys()))
moving_indices_carnival = [label_list_carnival.index(moving_label) for moving_label in label_list_carnival_moving]

label_list_mspacman = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_MSPACMAN.keys()))
label_list_mspacman_moving = sorted(list(MAX_NB_OBJECTS_MOVING_MSPACMAN.keys()))
moving_indices_mspacman = [label_list_mspacman.index(moving_label) for moving_label in label_list_mspacman_moving]

label_list_pong = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_PONG.keys()))
label_list_pong_moving = sorted(list(MAX_NB_OBJECTS_MOVING_PONG.keys()))
moving_indices_pong = [label_list_pong.index(moving_label) for moving_label in label_list_pong_moving]

label_list_boxing = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_BOXING.keys()))
label_list_boxing_moving = sorted(list(MAX_NB_OBJECTS_MOVING_BOXING.keys()))
moving_indices_boxing = [label_list_boxing.index(moving_label) for moving_label in label_list_boxing_moving]

label_list_tennis = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_TENNIS.keys()))
label_list_tennis_moving = sorted(list(MAX_NB_OBJECTS_MOVING_TENNIS.keys()))
moving_indices_tennis = [label_list_tennis.index(moving_label) for moving_label in label_list_tennis_moving]                                                   

label_list_space_invaders = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_SPACE_INVADERS.keys()))
label_list_space_invaders_moving = sorted(list(MAX_NB_OBJECTS_MOVING_SPACE_INVADERS.keys()))
moving_indices_space_invaders = [label_list_space_invaders.index(moving_label) for moving_label in label_list_space_invaders_moving]

label_list_riverraid = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_RIVER_RAID.keys()))
label_list_riverraid_moving = sorted(list(MAX_NB_OBJECTS_MOVING_RIVER_RAID.keys()))
moving_indices_riverraid = [label_list_riverraid.index(moving_label) for moving_label in label_list_riverraid_moving]

label_list_skiing = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_SKIING.keys()))
label_list_skiing_moving = sorted(list(MAX_NB_OBJECTS_MOVING_SKIING.keys()))
moving_indices_skiing = [label_list_skiing.index(moving_label) for moving_label in label_list_skiing_moving]

label_list_asterix = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_ASTERIX.keys()))
label_list_asterix_moving = sorted(list(MAX_NB_OBJECTS_MOVING_ASTERIX.keys()))
moving_indices_asterix = [label_list_asterix.index(moving_label) for moving_label in label_list_asterix_moving]

label_list_bowling = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_BOWLING.keys()))
label_list_bowling_moving = sorted(list(MAX_NB_OBJECTS_MOVING_BOWLING.keys()))
moving_indices_bowling = [label_list_bowling.index(moving_label) for moving_label in label_list_bowling_moving]

label_list_kangaroo = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_KANGAROO.keys()))
label_list_kangaroo_moving = sorted(list(MAX_NB_OBJECTS_MOVING_KANGAROO.keys()))
moving_indices_kangaroo = [label_list_kangaroo.index(moving_label) for moving_label in label_list_kangaroo_moving]

label_list_freeway = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_FREEWAY.keys()))
label_list_freeway_moving = sorted(list(MAX_NB_OBJECTS_MOVING_FREEWAY.keys()))
moving_indices_freeway = [label_list_freeway.index(moving_label) for moving_label in label_list_freeway_moving]

# Temporary fix for no (relevant) bboxes in the gt data. #TODO fix in atari_dataset.py instead
label_list_seaquest = [no_label_str, "NoObject"] + sorted(list(MAX_NB_OBJECTS_ALL_SEAQUEST.keys()))
label_list_seaquest_moving = ["NoObject"] + sorted(list(MAX_NB_OBJECTS_MOVING_SEAQUEST.keys()))
moving_indices_seaquest = [label_list_seaquest.index(moving_label) for moving_label in label_list_seaquest_moving]


# determined by collecting min and max values of a dataset of at least 128 x 4 images for each game
# manually extended for some games
# store as dictionary: game -> (min_x_min, min_y_min, max_x_max, max_y_max)
relevant_area_borders = {}
relevant_area_borders["skiing"] = (-0.05, 0.13333334028720856, 1.05, 0.8619047403335571)
relevant_area_borders["asterix"] = (-0.05, -0.05, 1.05, 0.7142857313156128)
relevant_area_borders["tennis"] = (-0.05, -0.05, 1.05, 1.05)
relevant_area_borders["seaquest"] = (-0.05, 0.21904762089252472, 1.10, 0.74)
relevant_area_borders["kangaroo"] = (-0.05, -0.05, 1.05, 0.91)
relevant_area_borders["freeway"] = (-0.05, 0.08, 1.05, 1.05)
relevant_area_borders["bowling"] = (0.08749999850988388, 0.5095238089561462, 0.9437500238418579, 0.8142856955528259)
relevant_area_borders["pong"] = (-0.05, 0.16190476715564728, 1.05, 0.9285714030265808)
relevant_area_borders["boxing"] = (0.21875, 0.18571428954601288, 0.793749988079071, 0.8238095045089722)

EXTRA_MARGIN = 0.02

# The goal of this function to  filter out predicted bboxes that outside of the relevant area of a game.
# The purpose is two-fold:
# 1. Filter out predicted bboxes that are obviously incorrect.
# 2. Filter out predicted bboxes that are correspond to an object (e.g. score) that is not relevant for the game.
def filter_relevant_boxes_masks(game, boxes_batch, boxes_gt):
    # use relevant area borders to filter out boxes
    game = game.lower()
    game = get_fuzzy_game_match(game)
    min_x_min, min_y_min, max_x_max, max_y_max = relevant_area_borders[game]
    min_x_min -= EXTRA_MARGIN
    min_y_min -= EXTRA_MARGIN
    max_x_max += EXTRA_MARGIN
    max_y_max += EXTRA_MARGIN

    if game == "tennis":
        # add extra rule to filter out only the scores in tennis: namely y_max of must be greater than 0.05
        return [(box_bat[:, 0] > min_y_min) & (box_bat[:, 1] < max_y_max) & (box_bat[:, 2] > min_x_min) & (box_bat[:, 3] < max_x_max) &
                (box_bat[:, 1] > 0.1) for box_bat in boxes_batch]

    # boxes_batch: list of arrays of shape (N, 4) where N is number of objects in that frame: y_min, y_max, x_min, x_max
    return [(box_bat[:, 0] > min_y_min) & (box_bat[:, 1] < max_y_max) & (box_bat[:, 2] > min_x_min) & (box_bat[:, 3] < max_x_max) for box_bat in boxes_batch]
    

def get_fuzzy_game_match(game):
    game = game.lower()
    if "mspacman" in game:
        return "mspacman"
    elif "carnival" in game:
        return "carnival"
    elif "space" in game and "invaders" in game:
        return "space_invaders"
    elif "pong" in game:
        return "pong"
    elif "boxing" in game:
        return "boxing"
    elif "riverraid" in game:
        return "riverraid"
    elif "tennis" in game:
        return "tennis"
    elif "skiing" in game:
        return "skiing"
    elif "asterix" in game:
        return "asterix"
    elif "bowling" in game:
        return "bowling"
    elif "kangaroo" in game:
        return "kangaroo"
    elif "freeway" in game:
        return "freeway"
    elif "seaquest" in game:
        return "seaquest"
    else:
        raise ValueError(f"Game {game} could not be found in labels")

def get_moving_indices(game):
    game = game.lower()
    if "mspacman" in game:
        return moving_indices_mspacman.copy()
    elif "carnival" in game:
        return moving_indices_carnival.copy()
    elif "space" in game and "invaders" in game:
        return moving_indices_space_invaders.copy()
    elif "pong" in game:
        return moving_indices_pong.copy()
    elif "boxing" in game:
        return moving_indices_boxing.copy()
    elif "riverraid" in game:
        return moving_indices_riverraid.copy()
    elif "tennis" in game:
        return moving_indices_tennis.copy()
    elif "skiing" in game:
        return moving_indices_skiing.copy()
    elif "asterix" in game:
        return moving_indices_asterix.copy()
    elif "bowling" in game:
        return moving_indices_bowling.copy()
    elif 'kangaroo' in game:
        return moving_indices_kangaroo.copy()
    elif 'freeway' in game:
        return moving_indices_freeway.copy()
    elif 'seaquest' in game:
        return moving_indices_seaquest.copy()
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def to_relevant(game, labels_moving):
    """
    Return Labels from line in csv file
    """
    no_label_idx = label_list_for(game).index(no_label_str)
    relevant_idx = [[l_m != no_label_idx for l_m in labels_seq] for labels_seq in labels_moving]
    return relevant_idx, [[l_m[rel_idx] for l_m, rel_idx in zip(labels_seq, rel_seq)]
                          for labels_seq, rel_seq in zip(labels_moving, relevant_idx)]


def label_list_for(game):
    """
    Return Labels from line in csv file
    """
    game = game.lower()
    if "mspacman" in game:
        return label_list_mspacman.copy()
    elif "carnival" in game:
        return label_list_carnival.copy()
    elif "pong" in game:
        return label_list_pong.copy()
    elif "boxing" in game:
        return label_list_boxing.copy()
    elif "tennis" in game:
        return label_list_tennis.copy()
    elif "riverraid" in game:
        return label_list_riverraid.copy()
    elif "space" in game and "invaders" in game:
        return label_list_space_invaders.copy()
    elif "skiing" in game:
        return label_list_skiing.copy()
    elif "asterix" in game:
        return label_list_asterix.copy()
    elif "bowling" in game:
        return label_list_bowling.copy()
    elif 'kangaroo' in game:
        return label_list_kangaroo.copy()
    elif 'freeway' in game:
        return label_list_freeway.copy()
    elif 'seaquest' in game:
        return label_list_seaquest.copy()
    else:
        raise ValueError(f"Game {game} could not be found in labels")
