'''
Output: Dictionary bzw. JSON mit allen erkannten Objekten
- für jedes Objekt: 
    - Klasse des erkannten Objektes (z.B. "Ball" bei Pong, oder "Tree" bei Skiing) 
    - Koordinaten des Objektes (x und y, normiert oder in absoluten Pixeln?) 
    - Maße des Objektes (Höhe und Breite oder Bounding Box Objekt) 
    - RGB Farbe des Objektes
- Beispiel für Skiing:
{‘player’: [ (corner_top_left_x, corner_top_left_y, width, heigth, confidence, rgb) ] , 
‘tree’ : [(corner_top_left_x, corner_top_left_y, width, heigth, confidence, rgb), (corner_top_left_x, corner_top_left_y, width, heigth, confidence, rgb)], 
'''
import torch
from torchvision import transforms
import numpy as np
from ox4rl.models.space.postprocess_latent_variables import latent_to_boxes_and_z_whats
from ox4rl.dataset.atari_labels import filter_relevant_boxes_masks, label_list_for
from collections import defaultdict
from ox4rl.utils.load_config import get_config_v2
from ox4rl.detector_api.space_detector_utils import load_classifier, load_space_for_inference
import os.path as osp

# get current directory as absolute path
space_and_moc_base_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
models_path = osp.join(space_and_moc_base_path, "scobots_spaceandmoc_detectors")

class Detector:
    '''
    Abstract class for detectors
    '''

    def __init__(self):
        pass

    def detect(self, frame):
        '''
        Detect objects in a frame
        :param frame: frame to detect objects in
        :return: dictionary with detected objects:
            - for each object:
                - class of the detected object
                - coordinates of the object (x and y, normalized or in absolute pixels?)
                - dimensions of the object (height and width or bounding box object)
                - RGB color of the object
        '''
        pass

class SPACEDetector(Detector):
    '''
    Detector using SPACE
    '''

    def __init__(self, game_name, hud=False, classifier = None, wrapped_space = None,):
        self.game_name = game_name
        config_path = osp.join(space_and_moc_base_path, "configs", "detector_api_configs", f"my_atari_{game_name}.yaml")
        self.cfg = get_config_v2(config_path) # get config must be called because it updates e.g. space_cfg.G which is set differently in different scobi.config
        self.hud = hud
        self.classifier = load_classifier(game_name) if not classifier else classifier
        self.wrapped_space = load_space_for_inference(game_name, self.cfg) if not wrapped_space else wrapped_space


    def detect(self, frame):

        # preprocess frame
        frame = self.scobi_image2space_image(frame, self.cfg.device)

        # forward pass using wrapped space
        self.wrapped_space.eval()
        with torch.no_grad():
            latent_logs_dict = self.wrapped_space.forward(frame)
        
        # postprocess latents
        predbboxs, z_whats = latent_to_boxes_and_z_whats(latent_logs_dict)
        if len(z_whats) == 0:
            return np.array([]), np.array([]), np.array([])
        
        if not self.hud:
            mask = filter_relevant_boxes_masks(self.game_name, predbboxs, None)[0]
        else:
            mask = [True for box_bat in predbboxs][0]
        if not torch.any(mask):
            return np.array([]), np.array([]), np.array([])
        predbboxs = predbboxs[0][mask]
        z_whats = z_whats[mask]
        predbboxs = predbboxs.to("cpu").numpy()
        z_whats = z_whats.to("cpu").numpy()

        # retrieve bboxes in correct format
        bboxes = self.transform_bbox_to_common_AILab_format(predbboxs)
        bboxes = np.array(bboxes * 128)

        # classify objects
        class_ids = self.classifier.predict(z_whats)
        # map class ids to class names
        class_ids = [label_list_for(self.game_name)[class_id] for class_id in class_ids]

        # dummy confidences (because the current classifier does not return confidences)
        confidences = np.zeros(len(class_ids))
        # TODO: clarify what is meant by confidence: probability for localization or classification or both? if both, how to combine them?

        # format final output
        output_dict = self.format_final_AILab_output(bboxes, confidences, class_ids)
        return output_dict
    
    def transform_bbox_format(self, bboxes):
        """
        Transform from (y_min, y_max, x_min, x_max) to (xmin, ymin, width, height) format.
        """
        new_format_bboxes = np.array(bboxes)
        new_format_bboxes[:, 0] = bboxes[:, 2]
        new_format_bboxes[:, 1] = bboxes[:, 0]
        new_format_bboxes[:, 2] = bboxes[:, 3] - bboxes[:, 2]
        new_format_bboxes[:, 3] = bboxes[:, 1] - bboxes[:, 0]
        return new_format_bboxes

    def transform_bbox_to_common_AILab_format(self, bboxes):
        """
        Transform from (y_min, y_max, x_min, x_max) to (top_left_x, top_left_y, width, height) format.
        """
        new_format_bboxes = np.array(bboxes)
        # TODO: check assumption: counting starts from top left corner of the image -> x_min, y_min, width, height
        width = bboxes[:, 3] - bboxes[:, 2]
        height = bboxes[:, 1] - bboxes[:, 0]
        new_format_bboxes[:, 0] = bboxes[:, 2]
        new_format_bboxes[:, 1] = bboxes[:, 0]
        new_format_bboxes[:, 2] = width
        new_format_bboxes[:, 3] = height
        return new_format_bboxes

    def format_final_AILab_output(self, bboxes, confidences, class_ids):
        """
        Format the final output in the format required by AILab.
        """

        # rescale bbox coordinates
        bboxes = self.space_bboxes2scobi_bboxes(bboxes)

        # create dictionary with class ids as keys
        output_dict = defaultdict(list)
        for i in range(len(bboxes)):
            output_dict[class_ids[i]].append((bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], confidences[i], None))
        output_dict = dict(output_dict)
        return output_dict
    
    @staticmethod
    def space_bboxes2scobi_bboxes(bboxes):
        if len(bboxes) == 0:
            return bboxes

        bboxes[:, 0] = bboxes[:, 0] * (160 / 128)
        bboxes[:, 1] = bboxes[:, 1] * (210 / 128)
        bboxes[:, 2] = bboxes[:, 2] * (160 / 128)
        bboxes[:, 3] = bboxes[:, 3] * (210 / 128)
        bboxes = np.array(bboxes, dtype=np.int32)
        return bboxes
    
    @staticmethod
    def scobi_image2space_image(img, target_device):
        #img =  Image.fromarray(img[:, :, ::-1], 'RGB')
        #img = img.resize((128, 128), Image.LANCZOS)
        #img = transforms.ToTensor()(img).unsqueeze(0).to("cuda")
        #return img

        # store image before resizing
        #original_img = Image.fromarray(img[:, :, ::-1], 'RGB')
        ## save original_img
        #original_img.save("original_img.png")
        #original_img = original_img.resize((128, 128), Image.LANCZOS)
        #original_img = transforms.ToTensor()(original_img).unsqueeze(0).to("cuda")
        img = img[:, :, ::-1].copy()
        img_tensor = torch.from_numpy(img).to(target_device)
        img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
        # Resize the image using PyTorch
        # Note: Interpolation mode 'bilinear' is usually a good balance between speed and quality
        resize = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        img_tensor = resize(img_tensor)
        img_tensor = img_tensor.unsqueeze(0)

        # save img_tensor for debugging
        #img_tensor = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        #img_tensor_for_save = Image.fromarray((img_tensor * 255).astype(np.uint8)[:,:,::-1], 'RGB')
        #img_tensor_for_save.save("img_tensor.png")

        return img_tensor