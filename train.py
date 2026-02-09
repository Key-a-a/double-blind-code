import torch
import numpy as np
import argparse
import time
import os
import os as _os
import util
from engine import trainer
import os
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='WDGSNET', help='choose model to run')
parser.add_argument('--device', type=str, default='0', help='graphics card')
parser.add_argument('--data', type=str, default='Datata/datat1/Xdo7', help='data path')
parser.add_argument('--adjdata', type=str, default='Datata/datat1/Xdo7/adj_mx.pkl', help='adj data path')
# parser.add_argument('--seq_length', type=int, default=12, help='prediction length')
parser.add_argument('--seq_length', type=int, default=42, help='use past 42 time steps (7 days × 6 samples/day)')

parser.add_argument('--nhid', type=int, default=40, help='')
parser.add_argument('--in_dim', type=int, default=4, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=6, help='number of nodes')
parser.add_argument('--out_dim', type=int, default=1, help='predict next 1 day pH')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--top_k', type=int, default=4, help='top-k sampling')
parser.add_argument('--print_every', type=int, default=100, help='')
parser.add_argument('--save', type=str, default='./garage/metr-la', help='save path')
parser.add_argument('--seed', type=int, default=530302, help='random seed')

args = parser.parse_args()
print(args)

# === Model switch for baseline comparisons ===
if args.model == 'lstm':
    # Defer to the lightweight LSTM baseline while reusing the same data pipeline.
    import sys as _sys
    try:
        from LSTM_baseline import run as _run_lstm
    except ImportError:
        # also allow alternative filename LSTM.py if user prefers
        try:
            from LSTM import run as _run_lstm
        except ImportError as _e:
            raise ImportError("Could not import LSTM baseline. Ensure LSTM_baseline.py or LSTM.py is in project root.") from _e

    # Build a minimal argv for the LSTM script to avoid argparse conflicts.
    lstm_argv = [
        'LSTM',
        '--device', str(args.device),
        '--data', str(args.data),
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.learning_rate),
        '--weight_decay', str(args.weight_decay),
        '--mse_lambda', '0.5'
    ]
    _bak = _sys.argv[:]
    _sys.argv = lstm_argv
    try:
        _run_lstm()
    finally:
        _sys.argv = _bak
    raise SystemExit(0)


def setup_seed(seed):
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def main():
    setup_seed(args.seed)

    _os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    adj_mx = util.load_adj(args.adjdata)
    supports = [torch.tensor(i).cuda() for i in adj_mx]

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    trainx = dataloader['train_loader'].xs
    print("trainx.shape =", trainx.shape)

    # ---- Build ATSH hyperedges via KNN (local) + KMeans (regional) as in the paper ----
    hypergraph_pkl = args.adjdata.replace("adj_mx.pkl", "hypergraph_knn_kmeans.pkl")
    if not os.path.exists(hypergraph_pkl):
        # Use training inputs (numpy) to build node features for clustering
        trainx_np = dataloader["train_loader"].xs  # [S, T, N, C_in]
        util.save_hypergraph_knn_kmeans(
            hypergraph_pkl, trainx_np, k_knn=args.top_k, num_clusters=max(2, int(args.top_k/2)), seed=args.seed)

    # Load hypergraph incidence (H_a: local, H_b: regional) and other matrices
    H_a, H_b, H_T_new, lwjl, G0, G1, indices, G0_all, G1_all = util.load_hadj(
        args.adjdata, args.top_k, hypergraph_pkl=hypergraph_pkl)

    scaler = dataloader['scaler']

    # (Disabled) corr-graph support removed to keep the training pipeline aligned with the paper/main diagram.

    # 自动对齐输入通道维度（考虑时间特征追加等）
    args.in_dim = dataloader['x_train'].shape[-1]
    print(f"[INFO] in_dim auto-set to {args.in_dim}")
    lwjl = (((lwjl.t()).unsqueeze(0)).unsqueeze(3)).repeat(args.batch_size, 1, 1, 1)

    H_a = H_a.cuda()
    H_b = H_b.cuda()
    G0 = torch.tensor(G0).cuda()
    G1 = torch.tensor(G1).cuda()
    H_T_new = torch.tensor(H_T_new).cuda()
    lwjl = lwjl.cuda()
    indices = indices.cuda()

    G0_all = torch.tensor(G0_all).cuda()
    G1_all = torch.tensor(G1_all).cuda()

    engine = trainer(
        args.batch_size,
        scaler,
        args.in_dim,
        args.seq_length,
        args.num_nodes,
        args.nhid,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        supports, H_a, H_b, G0, G1, indices,
        G0_all, G1_all, H_T_new, lwjl,
        args.clip,              # clip（位置参数）
        args.lr_decay_rate,     # lr_de_rate（位置参数，已保留但不再使用）
        out_dim=args.out_dim    # 只这一处传 out_dim
    )

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    print('[INFO] Starting training loop...')
    for i in range(1, args.epochs + 1):

        engine.mse_lambda = 0.5


        print('***** Epoch: %03d START *****' % i)
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).cuda()
            # dataloader yields [B, T, N, C]; convert to paper layout [B, N, T, C]
            trainx = trainx.permute(0, 2, 1, 3).contiguous()
            trainy = torch.Tensor(y).cuda()
            trainy = trainy.permute(0, 2, 1, 3).contiguous()  # [B,N,1,C_out]
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)

        # 每个 epoch 结束后调度一次学习率（Warmup + Cosine）
        engine.scheduler.step()

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).cuda()
            testx = testx.permute(0, 2, 1, 3).contiguous()
            testy = torch.Tensor(y).cuda()
            testy = testy.permute(0, 2, 1, 3).contiguous()
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

        print('***** Epoch: %03d END *****' % i)
        print('\n')

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).cuda()  # [S,1,N,1]
    realy = realy.permute(0, 2, 1, 3).contiguous()[:, :, 0, 0]  # [S,N]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).cuda()
        testx = testx.permute(0, 2, 1, 3).contiguous()
        with torch.no_grad():
            raw = engine.model(testx)  # [B,N,1,1]
            preds = raw[:, :, 0, 0]  # [B,N]
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    print("Best model epoch:", str(bestid + 1))

    # ===== 测试集评估（含 MAE, MAPE, RMSE, R2） =====
    real_np = realy.cpu().numpy()  # 形状 (N, V)
    pred_np = scaler.inverse_transform(yhat.cpu().numpy())  # 形状 (N, V)

    # 转为 Tensor 以计算 MAE/ MAPE/ RMSE
    pred_tensor = torch.from_numpy(pred_np).cuda()
    real_tensor = torch.from_numpy(real_np).cuda()
    mae, mape, rmse = util.metric(pred_tensor, real_tensor)

    # 计算 R²
    ss_res = np.sum((real_np - pred_np) ** 2)
    ss_tot = np.sum((real_np - real_np.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 打印结果（下一天 pH）
    print(f"do1-> :MAE {mae:.4f}, MAPE {mape:.4f}, RMSE {rmse:.4f}, R2 {r2:.4f}")

    # 保存最佳模型
    torch.save(engine.model.state_dict(),
               args.save + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
