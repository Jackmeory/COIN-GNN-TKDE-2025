import sys, json, argparse, random, re, os, shutil
sys.path.append("src/")
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
import os.path as osp
import networkx as nx
import pdb
import pickle
import csv
import pandas as pd
from openpyxl import load_workbook
import os

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
import torch.multiprocessing as mp
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch, k_hop_subgraph

from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from utils.data_convert import generate_samples
from src.model.modeld import Basic_Model
from model.ewcc import EWC
from src.Dataset import COINDataset
from src.model import detect
from src.model import replay
from utils.DMC import DM

result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
result_new = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
result_old = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
pin_memory = False
n_work = 0 


def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]

def init(args):    
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    del info

def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger
    return logger

def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test_model(model, args, testset, pin_memory, lossfunc):
    model.eval()
    lossfunc = args.loss
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            Loss, pred = args.shell.observeT(data, args, model, lossfunc)
            loss += Loss
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss/cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        mae = metric(truth_, pred_, args)
        pred_new, truth_new = pred_[:, args.en[args.year-args.begin_year]:, :], truth_[:, args.en[args.year-args.begin_year]:, :]
        mae = metric_new(truth_new, pred_new, args)
        pred_old, truth_old = pred_[:, :args.en[args.year-args.begin_year], :], truth_[:, :args.en[args.year-args.begin_year], :]
        mae = metric_old(truth_old, pred_old, args)
        return loss

def metric(ground_truth, prediction, args):
    global result
    pred_time = [3, 6, 12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i, mae, rmse, mape))
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
    return mae

def metric_new(ground_truth, prediction, args):
    global result_new
    pred_time = [3, 6, 12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i, mae, rmse, mape))
        result_new[i]["mae"][args.year] = mae
        result_new[i]["mape"][args.year] = mape
        result_new[i]["rmse"][args.year] = rmse
    return mae

def metric_old(ground_truth, prediction, args):
    global result_old
    pred_time = [3, 6, 12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i, mae, rmse, mape))
        result_old[i]["mae"][args.year] = mae
        result_old[i]["mape"][args.year] = mape
        result_old[i]["rmse"][args.year] = rmse
    return mae

def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)
    vars(args)["shell"] = DM(args)  # Creating memory buffer

    for year in range(args.begin_year, args.end_year):
        vars(args)['year'] = year

        # Load current year data
        inputs = generate_samples(31, osp.join(args.save_data_path, str(year)+'_30day'), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=True) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+".npz"), allow_pickle=True)
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        args.logger.info("[*] Year {} load from {}.npz".format(year, osp.join(args.save_data_path, str(year))))

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

        # Load next year data
        inputsF = generate_samples(31, osp.join(args.save_data_path, str(year+1)+'_30day'), np.load(osp.join(args.raw_data_path, str(year+1)+".npz"))["x"], graph, val_test_mix=True) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year+1)+".npz"), allow_pickle=True)
        fgraph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year+1)+"_adj.npz"))["x"])
        vars(args)["fgraph_size"] = fgraph.number_of_nodes()
        args.logger.info("[*] Year {} load from {}.npz".format(year+1, osp.join(args.save_data_path, str(year))))

        adjf = np.load(osp.join(args.graph_path, str(args.year+1)+"_adj.npz"))["x"]
        adjf = adjf / (np.sum(adjf, 1, keepdims=True) + 1e-6)
        vars(args)["adjf"] = torch.from_numpy(adjf).to(torch.float).to(args.device)
        vars(args)['nodeID'] = np.array(range(args.adj.size(0)))
        vars(args)['fnodeID'] = np.array(range(args.adjf.size(0)))
        vars(args)["sub_adj"] = vars(args)["adj"]

        # Load best model using provided path
        best_model_path = args.test_path + '/' + str(args.year)
        best_model_file = min([os.path.join(best_model_path, f) for f in os.listdir(best_model_path) if f.endswith('.pkl')], 
                             key=lambda x: float(os.path.basename(x).split('.pkl')[0]))
        
        args.logger.info("[*] Loading best model from {}".format(best_model_file))
        best_model = Basic_Model(args)
        best_model.load_state_dict(torch.load(best_model_file, map_location=args.device)["model_state_dict"])
        best_model = best_model.to(args.device)

        # Test the model
        idx = np.array(range(len(inputsF['train_x'])))
        test_loader = DataLoader(COINDataset(inputsF, "test", args.fnodeID, args.year+1, idx=idx), 
                               batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
        
        if args.loss == "mse": 
            lossfunc = func.mse_loss
        elif args.loss == "huber": 
            lossfunc = func.smooth_l1_loss
            
        test_model(best_model, args, test_loader, pin_memory, lossfunc)

    # Print results
# Save results to Excel - add this code directly here
    excel_path = f'results.xlsx'
    sheet_name = f'Hum'

    # Create rows for the DataFrame
    rows = []

    # Add data for horizon 3, mae and rmse
    for year in range(args.begin_year, args.end_year):
        row = {'Year': year}
        
        # Loop through specified metrics
        for i in [3]:
            for j in ['mae', 'rmse']:
                # All nodes
                if i in result and j in result[i] and year in result[i][j]:
                    row[f'All_{j.upper()}'] = result[i][j][year]
                    
                # New nodes
                if i in result_new and j in result_new[i] and year in result_new[i][j]:
                    row[f'New_{j.upper()}'] = result_new[i][j][year]
                    
                # Old nodes
                if i in result_old and j in result_old[i] and year in result_old[i][j]:
                    row[f'Old_{j.upper()}'] = result_old[i][j][year]
        
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    try:
        # Try to read the existing Excel file
        existing_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Append new data to existing data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        
        # Write back to Excel, replacing the sheet
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    except FileNotFoundError:
        # If file doesn't exist, create it with the new data
        df.to_excel(excel_path, sheet_name=sheet_name, index=False)
        
    except ValueError:
        # If sheet doesn't exist, create it with the new data
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type=str, default="conf/Hum.json", help='Hum: configure for humidity, Tem: configure for temperature, Traffic: configure for traffic')
    parser.add_argument("--test_path", type=str, default='/home/jz/OODG/TKDE2024/res/baselinetraffic/INCREASE', help='Path to model checkpoints')
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--loss", type=str, default="mse", help='Loss function: mse or huber')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logname", type=str, default="test_info")
    parser.add_argument("--warn_ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    init(args)
    seed_set(13)
    vars(args)['o_loss'] = [16.265, 15.4431, 15.1376, 17.3629, 13.4656, 14.6947]
    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = device
    vars(args)['time_stamp'] = []
    vars(args)['list'] = []
    main(args)
