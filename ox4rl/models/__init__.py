from .space.space import Space
from .slot.model import SlotAttentionAutoEncoder
def get_model(cfg):
    """
    :param cfg:
    :return:
    """
    model = None
    if cfg.model.lower() == 'space':
        model = Space()
    if cfg.model.lower() == 'slot':
        model = SlotAttentionAutoEncoder(num_slots=cfg.arch_slot.num_slots, resolution=cfg.resolution, hid_dim=cfg.arch_slot.hid_dim)
    return model