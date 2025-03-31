import argparse
import cv2
import numpy as np
import yaml
from torch.utils.data import DataLoader
import train_app.utils as utils
import train_app.dataset
import torch
import tqdm
import train_app.models as m #noqa
from train_app.utils import print_config
from eval import Evaluator

def print_conf(conf_p):
    with open(conf_p, "r") as c:
        config = yaml.safe_load(c)
    print_config(config)

def get_data(conf_p):
    with open(conf_p, "r") as c:
        config = yaml.safe_load(c)
    
    dataset_conf = config["dataset"]["valid"] if "test" not in config["dataset"] else config["dataset"]["test"]
    dataset_class = getattr(train_app.dataset, dataset_conf["type"])
    dataset_conf.pop("type", None)
    dataset_conf["mode"] = "test"
    dataset = dataset_class(**dataset_conf)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=6, pin_memory=False)
    return dataset, loader

def get_model(conf_p, ckpt):
    with open(conf_p, "r") as c:
        config = yaml.safe_load(c)
    model_config = config["model"]
    weights = torch.load(ckpt)
    model, _ = utils.generate_model(model_config)
    model.load_state_dict(weights['state_dict'] if 'state_dict' in weights else weights)
    
    return model.eval()

def get_metrics(args):
    evaluator = Evaluator(args.K)
    return evaluator
def paint_mask(mask : torch.Tensor):
    color_dict = {
        0 : (0, 0, 0), # background
        1 : (255, 255, 255), # Building_No_Damage
        2 : (255, 204, 153), # Building_Minor_Damage
        3 : (255, 102, 102), # Building_Major_Damage
        4 : (255, 0, 0),   # Building_Total_Destruction
        5 : (255, 255, 51),   # Vehicle
        6 : (128, 128, 128),   # Road
        7 : (0, 255, 0),     # Tree
        8 : (128, 128, 128),
        9 : (0, 0, 255),
        10 : (255, 0, 0),
        
    }
    mask = mask.squeeze().cpu().numpy()
    painted_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for val, color in color_dict.items():
        painted_image[mask == val, :] = color

    return painted_image

def main(args):
    conf_p = args.conf
    print_conf(conf_p)

    state_d = args.pth
    dataset, loader = get_data(conf_p)
    print("Data has been loaded.")
    model = get_model(conf_p, state_d)
    print("Model has been loaded.")
    

    model.cuda()
    ev = get_metrics(args)


    if args.vis:

        for inp in tqdm.tqdm(dataset):
            org_image = cv2.resize(cv2.imread(inp["path"]), inp["inputs"].shape[1 :])
            image =[im.unsqueeze(0).cuda() for im in inp["inputs"]] if isinstance(inp["inputs"], list) else inp["inputs"].unsqueeze(0).cuda()
            mask = inp["mask"].unsqueeze(0).cuda()
            with torch.no_grad():
                logits = model(image)
                logits = logits[-1] if isinstance(logits, tuple) else logits
                probs = torch.softmax(logits, dim=1)
            predicted_mask = torch.argmax(probs, 1).squeeze().cpu().numpy().astype(np.uint8)


            # entropy_map = utils.get_entropy(logits)
            entropy_heat = utils.tensor_to_heatmap(logits)

            colored_mask = paint_mask(mask)
            colored_prediction = paint_mask(torch.tensor(predicted_mask))
            cv2.imshow("Entropy", entropy_heat)
            cv2.imshow("Mask", cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))
            cv2.imshow("Image", org_image)
            cv2.imshow("Pred", cv2.cvtColor(colored_prediction, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'): break
            
        cv2.destroyAllWindows()
     
        
    else:
        for batch in tqdm.tqdm(loader):
            image = batch["inputs"].cuda()
            mask = batch["mask"].cuda()
            with torch.no_grad():
                logits = model(image)
                logits = logits[-1] if isinstance(logits, tuple) else logits
            
            ev.add_batch(mask, logits)

        ev.eval()
        print("IoU", ev.Intersection_over_Union)
        print("PixelAccuracy", ev.pixelAccuracy)
        print("Precision", ev.Precision)
        print("Recall", ev.Recall)
        print("F1", ev.F1)
        print("IoU_dict", ev.IoU_dict)
        print("meanIoU", ev.mIoU)

        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=False, default="run/train/ablation1/mccm-net.yml")
    parser.add_argument("--vis", required=False, default=True, type=bool)
    parser.add_argument("--pth", required=False, default="run/train/ablation1/weights/best.ckpt")
    parser.add_argument("--K", required=False, default=8, type=int)

    args = parser.parse_args()
    main(args)    