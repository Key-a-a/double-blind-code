import torch
import numpy as np
import argparse
import time
import os
import os as _os
import util
from engine import trainer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='WDGSNET', help='choose model to run')
parser.add_argument('--device', type=str, default='0', help='graphics card')
parser.add_argument('--data', type=str, default='Datata/datat1/Xdo7', help='data path')
parser.add_argument('--adjdata', type=str, default='Datata/datat1/Xdo7/adj_mx.pkl', help='adj data path')

parser.add_argument('--seq_length', type=int, default=42, help='use past 42 time steps (7 days × 6 samples/day)')

parser.add_argument('--nhid', type=int, default=40, help='')
parser.add_argument('--in_dim', type=int, default=6, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=6, help='number of nodes')
parser.add_argument('--out_dim', type=int, default=1, help='predict next 1 step')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--epochs', type=int, default=200, help='max training epochs')

# ===== Delayed Early Stopping =====
parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
parser.add_argument('--min_delta', type=float, default=1e-5, help='minimum val loss improvement')
parser.add_argument('--min_epochs', type=int, default=30, help='minimum epochs before early stopping is allowed')

parser.add_argument('--top_k', type=int, default=2, help='top-k sampling')
parser.add_argument('--print_every', type=int, default=100, help='')
parser.add_argument('--save', type=str, default='./garage/metr-la', help='save path prefix')
parser.add_argument('--seed', type=int, default=530302, help='random seed')

args = parser.parse_args()
print(args)

# === Model switch for baseline comparisons ===
if args.model == 'lstm':
    import sys as _sys
    try:
        from LSTM_baseline import run as _run_lstm
    except ImportError:
        try:
            from LSTM import run as _run_lstm
        except ImportError as _e:
            raise ImportError(
                "Could not import LSTM baseline. Ensure LSTM_baseline.py or LSTM.py is in project root."
            ) from _e

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    setup_seed(args.seed)

    _os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ===== Load adjacency =====
    adj_mx = util.load_adj(args.adjdata)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    # ===== Load dataset =====
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    trainx = dataloader['train_loader'].xs
    print("trainx.shape =", trainx.shape)

    # ===== Auto align input dim =====
    args.in_dim = dataloader['x_train'].shape[-1]
    print(f"[INFO] in_dim auto-set to {args.in_dim}")

    # ===== Build ATSH hyperedges =====
    hypergraph_pkl = args.adjdata.replace("adj_mx.pkl", "hypergraph_knn_kmeans.pkl")
    if not os.path.exists(hypergraph_pkl):
        trainx_np = dataloader["train_loader"].xs  # [S, T, N, C_in]
        util.save_hypergraph_knn_kmeans(
            hypergraph_pkl,
            trainx_np,
            k_knn=args.top_k,
            num_clusters=max(2, int(args.top_k / 2)),
            seed=args.seed
        )

    H_a, H_b, H_T_new, lwjl, G0, G1, indices, G0_all, G1_all = util.load_hadj(
        args.adjdata,
        args.top_k,
        hypergraph_pkl=hypergraph_pkl
    )

    scaler = dataloader['scaler']

    lwjl = (((lwjl.t()).unsqueeze(0)).unsqueeze(3)).repeat(args.batch_size, 1, 1, 1)

    H_a = H_a.to(device)
    H_b = H_b.to(device)
    G0 = torch.tensor(G0).to(device)
    G1 = torch.tensor(G1).to(device)
    H_T_new = torch.tensor(H_T_new).to(device)
    lwjl = lwjl.to(device)
    indices = indices.to(device)

    G0_all = torch.tensor(G0_all).to(device)
    G1_all = torch.tensor(G1_all).to(device)

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
        args.clip,
        args.lr_decay_rate,
        out_dim=args.out_dim
    )

    print("start training...", flush=True)

    his_loss = []
    val_time = []
    train_time = []

    # ===== Early stopping state =====
    best_val_loss = float('inf')
    best_epoch = -1
    wait = 0
    best_model_path = args.save + "_best_model.pth"

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
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.permute(0, 2, 1, 3).contiguous()   # [B, N, T, C]

            trainy = torch.Tensor(y).to(device)
            trainy = trainy.permute(0, 2, 1, 3).contiguous()   # [B, N, 1, C_out]

            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        # 每个 epoch 结束后调度一次学习率
        engine.scheduler.step()

        # ===== Validation =====
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).to(device)
            valx = valx.permute(0, 2, 1, 3).contiguous()

            valy = torch.Tensor(y).to(device)
            valy = valy.permute(0, 2, 1, 3).contiguous()

            metrics = engine.eval(valx, valy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        print('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)

        log = ('Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
               'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch')
        print(log.format(
            i, mtrain_loss, mtrain_mape, mtrain_rmse,
            mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)
        ), flush=True)

        # ===== NaN protection =====
        if np.isnan(mvalid_loss):
            print('[WARNING] Validation loss is NaN, stopping training.')
            break

        # ===== Delayed Early Stopping Logic =====
        if mvalid_loss < best_val_loss - args.min_delta:
            best_val_loss = mvalid_loss
            best_epoch = i
            wait = 0
            torch.save(engine.model.state_dict(), best_model_path)
            print(f'[EARLY STOP] Validation loss improved to {best_val_loss:.6f}. Best model saved at epoch {best_epoch}.')
        else:
            wait += 1
            print(f'[EARLY STOP] No improvement. Patience {wait}/{args.patience}. Minimum epoch gate: {i}/{args.min_epochs}')

        print('***** Epoch: %03d END *****' % i)
        print('\n')

        # 只有达到最小训练轮数后，才允许 early stopping 生效
        if i >= args.min_epochs and wait >= args.patience:
            print(f'[EARLY STOP] Triggered at epoch {i}. Best epoch was {best_epoch}, best val loss was {best_val_loss:.6f}.')
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time) if len(train_time) > 0 else 0.0))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time) if len(val_time) > 0 else 0.0))

    if best_epoch == -1:
        raise RuntimeError("No valid best model was saved. Please check training/validation loss.")

    print(f'Loading best model from epoch {best_epoch}...')
    engine.model.load_state_dict(torch.load(best_model_path, map_location=device))

    # ===== Test =====
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)   # [S,1,N,1]
    realy = realy.permute(0, 2, 1, 3).contiguous()[:, :, 0, 0]   # [S,N]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.permute(0, 2, 1, 3).contiguous()

        with torch.no_grad():
            raw = engine.model(testx)   # [B,N,1,1]
            preds = raw[:, :, 0, 0]     # [B,N]

        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(best_val_loss, 4)))
    print("Best model epoch:", str(best_epoch))

    # ===== Test metrics: MAE, MAPE, RMSE, R2 =====
    real_np = realy.cpu().numpy()
    pred_np = scaler.inverse_transform(yhat.cpu().numpy())

    pred_tensor = torch.from_numpy(pred_np).to(device)
    real_tensor = torch.from_numpy(real_np).to(device)
    mae, mape, rmse = util.metric(pred_tensor, real_tensor)

    ss_res = np.sum((real_np - pred_np) ** 2)
    ss_tot = np.sum((real_np - real_np.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    print(f"do1-> :MAE {mae:.4f}, MAPE {mape:.4f}, RMSE {rmse:.4f}, R2 {r2:.4f}")

    final_best_path = args.save + f"_best_{round(best_val_loss, 4)}.pth"
    torch.save(engine.model.state_dict(), final_best_path)
    print(f"[INFO] Best model also saved to: {final_best_path}")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
