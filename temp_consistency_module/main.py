#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from statistics import mode, StatisticsError
from sklearn.utils.extmath import weighted_mode

import pandas as pd
import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *
import skimage.io as io
from scipy.ndimage import binary_fill_holes
from utils import AverageMeter
from utils import compute_mask_IU

import models, datasets
from utils import flow_utils, tools, warp_utils

import matplotlib.pyplot as plt

# fp32 copy of parameters for update
global param_copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--num-classes', type=int, default=7, help="number of classes in dataset")
    parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument("--rgb_max", type=float, default = 255.)
    parser.add_argument('--assignment_strategy', type=str, default='mode', help="Class assigment strategy")
    parser.add_argument('--threshold', type=float, default=0.0, help="Min jaccard score for candidate match")

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true', help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')
    parser.add_argument('--save_predictions', action='store_true', help='save predictions to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024., help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    
    # csv variables
    parser.add_argument('--csv_path', default='./data.csv', type=str, help='path to save csv w/ name')
    parser.add_argument('--task', type=int, default=1, help="task number")
    parser.add_argument('--algorithm', default='tracker.csv', type=str, help='algorithm name for csv')

    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='RobotsegTrackerDataset', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'img_dir': 'images',
                                                        'ann_dir': 'annotations',
                                                        'cand_dir': 'candidates',
                                                        'coco_ann_dir': 'coco_anns.json',
                                                        'segm_dir': 'segm.json',
                                                        'prev_frames': 7,
                                                        'nms': True,
                                                        'maskrcnn_inference': False,
                                                        'dataset': '2017'})


    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE',  action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
        args.log_file = join(args.save, 'args.txt')

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}
        args.inference_dir = "{}/inference".format(args.save)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Source Code')
    print(('  Current Git Hash: {}\n'.format(args.current_hash)))

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus
        gpuargs = {'num_workers': args.effective_number_workers, 
                   'pin_memory': True, 
                   'drop_last' : True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers

        if exists(args.inference_dataset_img_dir):
            inference_dataset = args.inference_dataset_class(**tools.kwargs_from_args(args, 'inference_dataset'))
            block.log('Inference Dataset: {}'.format(args.inference_dataset))
            block.log('Inference Input: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][0]])))
            block.log('Inference Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][1]])))
            inference_loader = DataLoader(inference_dataset, batch_size=args.effective_inference_batch_size, shuffle=False, **inf_gpuargs)

    # Dynamically load model and loss class with parameters passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class Model(nn.Module):
            def __init__(self, args):
                super(Model, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                
            def forward(self, data, target, inference=False ):
                output = self.model(data)
                return output

        model = Model(args)

        block.log('Effective Batch Size: {}'.format(args.effective_batch_size))
        block.log('Number of parameters: {}'.format(sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])))

        # assing to cuda or wrap with dataparallel, model and loss 
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            block.log('Parallelizing')
            model = nn.parallel.DataParallel(model, device_ids=list(range(args.number_gpus)))

            block.log('Initializing CUDA')
            model = model.cuda().half()
            torch.cuda.manual_seed(args.seed) 
            param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]

        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model = model.cuda()
            block.log('Parallelizing')
            model = nn.parallel.DataParallel(model, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed) 

        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        # Load weights if needed, otherwise randomly initialize
        if args.resume and os.path.isfile(args.resume):
            block.log("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_err = checkpoint['best_EPE']
            model.module.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))
        elif args.resume and args.inference:
            block.log("No checkpoint found at '{}'".format(args.resume))
            quit()
        else:
            block.log("Random initialization")

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    # Log all arguments to file
    for argument, value in sorted(vars(args).items()):
        block.log2file(args.log_file, '{}: {}'.format(argument, value))

    # Reusable function for inference
    def inference(args, epoch, data_loader, model, offset=0):
        model.eval()
        
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if args.save_flow or args.render_validation:
            flow_folder = "{}/inference/{}-flow".format(args.save, args.name.replace('/', '.'))
            if not os.path.exists(flow_folder):
                os.makedirs(flow_folder)
        
        if args.save_predictions:
            predictions_folder = join(args.save, 'predictions')
            if not os.path.exists(predictions_folder):
                os.makedirs(predictions_folder)


        args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches

        progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches), desc='Inferencing ', 
            leave=True, position=offset)

        statistics = []
        total_loss = 0
        all_im_iou_acc = []
        all_im_iou_acc_challenge = []
        cum_I, cum_U = 0, 0
        class_ious = {c: [] for c in range(1, args.num_classes+1)}
        outputs = []
        changed = 0
        matched = 0
        if args.csv_path is not None:
            if not os.path.exists(os.path.dirname(args.csv_path)):
                os.makedirs(os.path.dirname(args.csv_path))
            toolbox_dict = {'Task':[], 'TestCase':[],
                            'Algorithm':[], 'MetricValue':[]}

        for batch_idx, (data_list, target_list, cand_list, pred_list, score_list, full_mask, inference, img_name) in enumerate(progress):
            _, h, w = full_mask.size()
            prediction = torch.zeros(((h, w, args.num_classes + 1)), dtype=torch.float)
            if inference:
                # 1. Calculate backward optical flow
                for data, target in zip(data_list, target_list):
                    if args.cuda:
                        data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
                    data, target = [Variable(d) for d in data], [Variable(t) for t in target]
                    with torch.no_grad():
                        output = model(data[0].unsqueeze(0), target[0].unsqueeze(0), inference=True)

                    if args.save_flow or args.render_validation:
                        for i in range(args.inference_batch_size):
                            _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                            flow_utils.writeFlow(join(flow_folder, '%06d.flo'%(batch_idx * args.inference_batch_size + i)),  _pflow)
                    
                    if len(outputs) >= (args.inference_dataset_prev_frames - 1):
                        outputs.pop(-1)
                        outputs.append(output)
                        break
                    else:
                        outputs.append(output)

                # 2. Warp previous frame candidates to current frame
                for i, (cands, output) in enumerate(zip(reversed(cand_list[1:]), reversed(outputs))):
                    if len(cands) > 0:
                        if i > 0:
                            cands = torch.cat((warped_cands, cands.float()), dim=1)
                        warped_cands = warp_utils.warp_candidates(cands.squeeze(0), output.squeeze(0)).unsqueeze(0)

                if len(cand_list[0]) > 0:
                    # 3. Match previous frame candidates with current candidates
                    cand_match = torch.Tensor(cand_list[0].shape[1], len(cand_list[1:])).fill_(np.inf) 
                    for i, cands in enumerate(reversed(cand_list[1:])):
                        if len(cands) > 0:
                            cands = warped_cands[:, np.arange(warped_cands.shape[1]) < cands.shape[1], :, :]
                            warped_cands = warped_cands[:, np.arange(warped_cands.shape[1]) >= cands.shape[1], :, :]
                            match = warp_utils.match(cand_list[0].squeeze(0), cands.squeeze(0), args.threshold)
                            cand_match[:, i] = match

                    # 4. Generate prediction
                    for i in range(cand_match.shape[0]):
                        match = cand_match[i, :]
                        if (match != np.inf).sum() > 1:
                            cand_preds = [pred_list[0][i].item()]
                            cand_scores = [score_list[0][i].item()]
                            matched += 1
                            for c_idx, preds, scores in zip(match, reversed(pred_list[1:]), reversed(score_list[1:])):
                                if c_idx != np.inf:
                                    cand_preds.append(preds[int(c_idx.item())].item())
                                    cand_scores.append(scores[int(c_idx.item())].item())
                                                          
                            if args.assignment_strategy == 'mode':
                                try:
                                    new_pred = mode(cand_preds)
                                except StatisticsError:
                                    new_pred = cand_preds[0]                                  
                            elif args.assignment_strategy == 'weighted_mode':
                                # assign class according to score
                                new_pred, _ = weighted_mode(cand_preds, cand_scores)
                                new_pred = np.squeeze(new_pred)
                            elif args.assignment_strategy == 'max':
                                # assign class according to max over all mean(scores of class i)
                                cand_scores = np.array(cand_scores)
                                unique_classes = np.unique(cand_preds)
                                mean_scores = [np.mean(cand_scores[cand_preds == c]) for c in unique_classes]
                                class_idx = np.argmax(mean_scores)
                                new_pred = unique_classes[class_idx]

                            new_score =  np.array(cand_scores)[np.array(cand_preds) == new_pred].mean()
                            if new_pred != cand_preds[0]:
                                changed += 1

                            prediction[:, :, new_pred] += cand_list[0][0][i, :, :] * new_score
                        else:
                            prediction[:, :, pred_list[0][i]] += cand_list[0][0][i, :, :].unsqueeze(2) * score_list[0][i]
            else:
                outputs = []
                if len(cand_list) > 0:
                    cand_list = cand_list.squeeze(0)
                    for i in range(cand_list.shape[0]):
                        prediction[:, :, pred_list[i]] += cand_list[i, :, :].unsqueeze(2) * score_list[i]

            # Non-existent classes in 2018-augmented are deleted
            if args.inference_dataset_dataset == '2018' and args.num_classes == 9:
                prediction[:, :, 4] = 0
                prediction[:, :, 5] = 0
            elif args.inference_dataset_dataset == '2017' and args.num_classes == 9:
                prediction[:, :, 8] = 0
                prediction[:, :, 9] = 0
            prediction = np.argmax(prediction.numpy(), 2) # colapse class dim

            # Save predictions
            if args.save_predictions:
                filename = os.path.join(args.save, 'predictions', img_name[0])
                io.imsave(filename, prediction.astype(np.uint8))

            # calculate image challenge metrics
            im_iou = []
            im_iou_challenge = []
            target = full_mask.numpy()
            gt_classes = np.unique(target)
            gt_classes.sort()
            gt_classes = gt_classes[gt_classes > 0] # remove background
            if np.sum(prediction) == 0:
                if target.sum() > 0: 
                    # Annotation is not empty and there is no prediction
                    all_im_iou_acc.append(0)
                    all_im_iou_acc_challenge.append(0)
                    for class_id in gt_classes:
                        class_ious[class_id].append(0)
                    if args.csv_path is not None:
                        toolbox_dict['Task'].append(args.task)
                        toolbox_dict['TestCase'].append(img_name[0])
                        toolbox_dict['Algorithm'].append(args.algorithm)
                        toolbox_dict['MetricValue'].append(0)
                # Annotation is empty and there is no prediction
                continue
            
            gt_classes = torch.unique(full_mask)
            for class_id in range(1, args.num_classes + 1): 
                current_pred = (prediction == class_id).astype(np.float)
                current_target = (full_mask.numpy() == class_id).astype(np.float)
                if current_pred.astype(np.float).sum() != 0 or current_target.astype(np.float).sum() != 0:
                    i, u = compute_mask_IU(current_pred, current_target)       
                    im_iou.append(i/u)
                    cum_I += i
                    cum_U += u
                    class_ious[class_id].append(i/u)
                    if class_id in gt_classes:
                        # consider only classes present in gt
                        im_iou_challenge.append(i/u)
            if len(im_iou) > 0:
                # to avoid nans by appending empty list
                all_im_iou_acc.append(np.mean(im_iou))
            if len(im_iou_challenge) > 0:
                # to avoid nans by appending empty list
                all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))
            if args.csv_path is not None:
                toolbox_dict['Task'].append(args.task)
                toolbox_dict['TestCase'].append(img_name[0])
                toolbox_dict['Algorithm'].append(args.algorithm)
                toolbox_dict['MetricValue'].append(np.mean(im_iou))

            if batch_idx % 100 == 0:
                tqdm.write('class mIoU: {:.5f}, IoU: {:.5f}, challenge IoU: {:.5f}'.format(
                            cum_I / cum_U, np.mean(all_im_iou_acc),
                            np.mean(all_im_iou_acc_challenge)))
        # calculate final metrics
        final_im_iou = cum_I / cum_U
        mean_im_iou = np.mean(all_im_iou_acc)
        mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)
        
        final_class_im_iou = torch.zeros(9)
        print('Final cIoU per class:')
        print('| Class | cIoU |')
        print('-----------------')
        for c in range(1, args.num_classes + 1):
            final_class_im_iou[c-1] = torch.tensor(class_ious[c]).mean()
            print('| {} | {:.5f} |'.format(c, final_class_im_iou[c-1]))
        print('-----------------')
        mean_class_iou = torch.tensor([torch.tensor(values).mean() for c, values in class_ious.items() if len(values) > 0]).mean()
        print('mIoU: {:.5f}, IoU: {:.5f}, challenge IoU: {:.5f}, mean class IoU: {:.5f}'.format(
                                                final_im_iou,
                                                mean_im_iou,
                                                mean_im_iou_challenge,
                                                mean_class_iou))
        print('Match candidates: {} Changed candidates: {}'.format(matched, changed))
        if args.csv_path is not None:
            # save csv
            df = pd.DataFrame(toolbox_dict)
            df.to_csv(args.csv_path, sep=';', index=False)

        progress.close()

        return

    inference(args=args, epoch=0, data_loader=inference_loader, model=model, offset=1)

    print("\n")
