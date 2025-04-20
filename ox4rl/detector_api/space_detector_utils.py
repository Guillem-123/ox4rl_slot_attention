import joblib
from ox4rl.models.space.space import Space
from  ox4rl.models.space.inference_space import WrappedSPACEforInference
from ox4rl.training.checkpointing.checkpointer import Checkpointer
import os.path as osp
from ox4rl.utils.load_config import get_config_v2

# get current directory as absolute path
space_and_moc_base_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
models_path = osp.join(space_and_moc_base_path, "scobots_spaceandmoc_detectors")


def load_classifier(game_name):
    classifier_file_name = "z_what-classifier_relevant_nn.joblib.pkl"
    classifier_path = osp.join(models_path, game_name, "classifier", classifier_file_name)
    classifier = joblib.load(classifier_path)
    return classifier

def load_space_for_inference(game_name, cfg):
    device = "cpu" if cfg.device == "cpu" else "cuda"
    model_path = osp.join(models_path, game_name, "space_weights", "model_000005001.pth")
    model = Space()
    model = model.to(device)
    checkpointer = Checkpointer("dummy_path", max_num=4) #TODO replace dummy path
    checkpointer.load(model_path, model, None, None, device)
    model = WrappedSPACEforInference(model)
    return model