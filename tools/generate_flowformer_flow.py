
from tools.data import FlowFormerDataset
import argparse
import os, sys
import importlib
import torch
import tqdm

core_path = os.path.abspath(os.path.join('submodules', 'flowformer', 'core'))
if core_path not in sys.path:
    sys.path.insert(0, core_path)


# Import other modules if needed
from submodules.flowformer.configs.sintel import get_cfg
from submodules.flowformer.core.FlowFormer import build_flowformer

def generate_flowformer_flow(root, fps, v_id, dset):
    
    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    
    model_path = os.path.join(args.models_root, f"{args.model}.pth")
    print(f"Loading from {model_path}...")
    model.load_state_dict(torch.load(model_path))

    model.cuda()
    model.eval()
    
    dataset = FlowFormerDataset(root, fps, v_id, dset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    flow_path = dataset.dset.flowformer_root(v_id)
    os.makedirs(flow_path, exist_ok=True)
    base_filename = 'flow_{:010d}.pth'
    i = 1
    for image1, image2 in tqdm.tqdm(loader, total=len(loader)):
        with torch.no_grad():
            image1 = image1.cuda()
            image2 = image2.cuda()
            flow, _ = model(image1, image2)
            flow = flow.squeeze().cpu()
            filename = os.path.join(flow_path, base_filename.format(i))
            i += 1
            torch.save(flow, filename)
                    
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data')
    parser.add_argument('--fps', type=float, default=float('inf'))
    parser.add_argument('--v_id', default=None, help='video or kitchen ID')
    parser.add_argument('--dset', type=str, default='epic')
    parser.add_argument('--models_root', type=str, default='submodules/flowformer/models')
    parser.add_argument('--model', type=str, default='sintel')
    args = parser.parse_args()
    
    generate_flowformer_flow(args.root, args.fps, args.v_id, args.dset)
