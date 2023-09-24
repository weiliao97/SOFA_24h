import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datetime import date
today = date.today()
date = today.strftime("%m%d")
import models
import prepare_data
import utils
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=None, shuffle=False)
mse_loss = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for Tranformer models")
    # data
    parser.add_argument("--database", type=str, default='mimic', choices=['mimic', 'eicu'])
    # datapath 
    # parser.add_argument("--td_data", type=str, help = 'path to the time-series data')
    # parser.add_argument("--static_data", type=str, help = 'path to the static data')
    # parser.add_argument("--sofa_data", type=str, help = 'path to the SOFA target data')
    # data grouping and cohort 
    parser.add_argument("--bucket_size", type=int, default=300, help="bucket size to group different length of time-series data")
    parser.add_argument("--use_sepsis3", action='store_false', default=True, help="Whethe only use sepsis3 subset")
    
    # modeling
    parser.add_argument("--model_name", type=str, default='TCN', choices=['TCN', 'RNN', 'Transformer'])
    # how to fuse with transformer models and LSTM models is still pending
    parser.add_argument("--static_fusion", type=str, default='med',
                        choices=['no_static', 'early', 'med', 'late', 'all', 'inside'])

    parser.add_argument('--s_param', nargs='+', help='Fusion II, III, IV, V params', type=float)
    parser.add_argument('--c_param', nargs='+', help='Main model FC params', type=float)
    parser.add_argument('--sc_param', nargs='+', help='Fusion VI params', type=float)
   
    # regularization 
    parser.add_argument("--regularization", type=str, default = "none", choices = ['none', 'l1', 'l2'])
    parser.add_argument("--l1_strength", type=float, default=0.001, help="L1 regularization lambda")
    parser.add_argument("--l2_strength", type=float, default=0.0001, help="L2 regularization lambda")
   
    # model parameters
    # TCN
    parser.add_argument("--kernel_size", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout")
    parser.add_argument("--reluslope", type=float, default=0.1, help="Relu slope in the fc model")
    parser.add_argument('--num_channels', nargs='+', help='TCN model channels', type=int)

    # LSTM
    parser.add_argument("--rnn_type", type=str, default='lstm', choices=['rnn', 'lstm', 'gru'])
    parser.add_argument("--hidden_dim", type=int, default=256, help="RNN hidden dim")
    parser.add_argument("--layer_dim", type=int, default=3, help="RNN layer dim")

    # transformer
    parser.add_argument('--warmup', action='store_true', help="whether use learning rate warm up")
    parser.add_argument('--lr_factor', type=int, default=0.1, help="warmup_learning rate factor")
    parser.add_argument('--lr_steps', type=int, default=2000, help="warmup_learning warm up steps")
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model")
    parser.add_argument("--n_head", type=int, default=8, help="Attention head of the model")
    parser.add_argument("--dim_ff_mul", type=int, default=4, help="Dimension of the feedforward model")
    parser.add_argument("--num_enc_layer", type=int, default=2, help="Number of encoding layers")

    # learning parameters
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--data_batching", type=str, default='close', choices=['same', 'close', 'random'],
                        help='How to batch data, same: same length, close: close-enough length, random: random length')
    parser.add_argument("--bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")  # could be overwritten by warm up

    parser.add_argument("--checkpoint", type=str, default='med_fusion_ks3', help=" name of checkpoint model")
    # Parse and return arguments
    args = parser.parse_args()

    # arg_dict = vars(args)

    # # for fusion all
    # arg_dict['num_channels'] = [256, 256, 256, 256]
    # arg_dict['s_param'] = [256, 256, 256, 0.2]
    # arg_dict['c_param'] = [256, 256, 0.2]
    # arg_dict['sc_param'] = [256, 256, 256, 0.2]

  
    workname = date + '_' + args.database + '_' + args.model_name + '_' + args.checkpoint
    utils.creat_checkpoint_folder('./checkpoints/' + workname, 'params.json', vars(args))

    # load_data
    meep_mimic = np.load(
        '/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_compile_0911_2022.npy', \
        allow_pickle=True).item()
    train_vital = meep_mimic['train_head']
    dev_vital = meep_mimic['dev_head']
    test_vital = meep_mimic['test_head']
    mimic_static = np.load(
        '/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_static_0922_2022.npy', \
        allow_pickle=True).item()
    mimic_target = np.load(
        '/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_target_0922_2022.npy', \
        allow_pickle=True).item()

    train_head, train_static, train_sofa, train_id = utils.crop_data_target(args.database, train_vital, mimic_target, mimic_static, 'train')
    dev_head, dev_static, dev_sofa, dev_id = utils.crop_data_target(args.database, dev_vital, mimic_target, mimic_static, 'dev')
    test_head, test_static, test_sofa, test_id = utils.crop_data_target(args.database, test_vital, mimic_target, mimic_static, 'test')

    if args.use_sepsis3 == True:
        train_head, train_static, train_sofa, train_id = utils.filter_sepsis(args.database, train_head, train_static, train_sofa, train_id)
        dev_head, dev_static, dev_sofa, dev_id = utils.filter_sepsis(args.database, dev_head, dev_static, dev_sofa, dev_id)
        test_head, test_static, test_sofa, test_id = utils.filter_sepsis(args.database, test_head, test_static, test_sofa, test_id)

    input_dim =train_head[0].shape[0]
    static_dim = train_static[0].shape[0]

    if args.static_fusion != 'no_static':
        s_param_p = [int(i) if i > 1.0 else i for i in args.s_param]
        c_param_p = [int(i) if i > 1.0 else i for i in args.c_param]
        sc_param_p = [int(i) if i > 1.0 else i for i in args.sc_param]

    if args.static_fusion == 'no_static':

        if args.model_name == 'TCN':
            model = models.TemporalConv(num_inputs=input_dim, num_channels=args.num_channels,
                                        kernel_size=args.kernel_size, dropout=args.dropout)
        elif args.model_name == 'RNN':
            model = models.RecurrentModel(cell=args.rnn_type, input_dim = input_dim, hidden_dim=args.hidden_dim, layer_dim=args.layer_dim, \
                                        output_dim=1, dropout_prob=args.dropout, idrop=args.idrop)

        elif args.model_name == 'Transformer':
            model = models.Trans_encoder(feature_dim=input_dim, d_model=args.d_model, \
                  nhead=args.n_head, d_hid=args.dim_ff_mul * args.d_model, \
                  nlayers=args.num_enc_layer, out_dim=1, dropout=args.dropout)

    elif args.static_fusion == 'med':
        model = models.TemporalConvStatic(num_inputs=input_dim, num_channels=args.num_channels, \
                                            num_static=static_dim, kernel_size=args.kernel_size, dropout=args.dropout, 
                                            s_param=s_param_p, c_param=c_param_p)

    elif args.static_fusion == 'early':
        model = models.TemporalConvStaticE(num_inputs=input_dim + static_dim, num_channels=args.num_channels, \
                                            num_static=static_dim, kernel_size=args.kernel_size, dropout=args.dropout,
                                            c_param=c_param_p)

    elif args.static_fusion == 'late':
        model = models.TemporalConvStaticL(num_inputs=input_dim, num_channels=args.num_channels, \
                                            num_static=static_dim, kernel_size=args.kernel_size, dropout=args.dropout,
                                            c_param=c_param_p, sc_param=sc_param_p)
    elif args.static_fusion == 'all':
        model = models.TemporalConvStaticA(num_inputs=input_dim + static_dim, num_channels=args.num_channels, \
                                            num_static=static_dim, kernel_size=args.kernel_size, dropout=args.dropout,
                                            s_param = s_param_p, c_param=c_param_p, sc_param=sc_param_p)


    elif args.static_fusion == 'inside':  
        model = models.TemporalConvStaticI(num_inputs=input_dim + static_dim, num_channels=args.num_channels, num_static=static_dim,
                                            kernel_size=args.kernel_size, dropout=args.dropout, 
                                            s_param = s_param_p, c_param=c_param_p, sc_param=sc_param_p)
    
    else: 
        raise ValueError('Please specify a valid static fusion method')

    print('Model trainable parameters are: %d' % utils.count_parameters(model))
    torch.save(model.state_dict(), '/content/start_weights.pt')

    model.to(device)
    best_loss = 1e4

    # loss fn and optimizer
    loss_fn = nn.MSELoss()
    
    if args.regularization == 'l2':
    # fuse inside opt term
        model_opt = torch.optim.Adam([
            {'params': model.TB1.parameters()},
            {'params': model.TB2.parameters()},
            {'params': model.TB3.parameters()},
            {'params': model.TB4.parameters()},
            {'params': model.composite.parameters()},
            {'params': model.static.parameters(), 'weight_decay':  args.l2_strength},
            {'params': model.static1.parameters(), 'weight_decay': args.l2_strength},
            {'params': model.static2.parameters(), 'weight_decay': args.l2_strength},
            {'params': model.static3.parameters(), 'weight_decay': args.l2_strength},
            {'params': model.s_composite.parameters(), 'weight_decay':  args.l2_strength},
        ], lr=args.lr)
    else:
        model_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 10-fold cross validation
    trainval_head = train_head + dev_head
    trainval_static = train_static + dev_static
    trainval_stail = train_sofa + dev_sofa
    trainval_ids = train_id + dev_id

    for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_head)):
        best_loss = 1e4
        patience = 0
        if c_fold >= 1:
            model.load_state_dict(torch.load('/content/start_weights.pt'))
        print('Starting Fold %d' % c_fold)
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        train_head, val_head = utils.slice_data(trainval_head, train_index), utils.slice_data(trainval_head, test_index)
        train_static, val_static = utils.slice_data(trainval_static, train_index), utils.slice_data(trainval_static, test_index)
        train_stail, val_stail = utils.slice_data(trainval_stail, train_index), utils.slice_data(trainval_stail, test_index)
        train_id, val_id = utils.slice_data(trainval_ids, train_index), utils.slice_data(trainval_ids, test_index)

        train_dataloader, dev_dataloader, test_dataloader = prepare_data.get_data_loader(args, train_head, val_head,
                                                                                            test_head, 
                                                                                            train_stail, val_stail,
                                                                                            test_sofa,
                                                                                            train_static=train_static,
                                                                                            dev_static=val_static,
                                                                                            test_static=test_static,
                                                                                            train_id=train_id,
                                                                                            dev_id=val_id,
                                                                                            test_id=test_id)

        for j in range(args.epochs):
            model.train()
            sofa_list = []
            sofap_list = []
            loss_t = []
            loss_to = []

            for vitals, static, target, train_ids, key_mask in train_dataloader:
                # print(label.shape)
                if args.warmup == True:
                    model_opt.optimizer.zero_grad()
                else:
                    model_opt.zero_grad()
                # ti_data = Variable(ti.float().to(device))
                # td_data = vitals.to(device) # (6, 182, 24)
                # sofa = target.to(device)
                if args.static_fusion == 'no_static':
                    if args.model_name == 'TCN': 
                        sofa_p = model(vitals.to(device))
                    elif args.model_name == 'RNN':
                        # x_lengths have to be a 1d tensor
                        td_transpose = vitals.to(device).transpose(1, 2)
                        x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                        sofa_p = model(td_transpose, x_lengths)
                    elif args.model_name == 'Transformer':
                        tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                        sofa_p = model(vitals.to(device), tgt_mask, key_mask.bool().to(device))
                else:
                    sofa_p = model(vitals.to(device), static.to(device))

                loss = utils.mse_maskloss(sofa_p, target.to(device), key_mask.to(device))
                if args.regularization == 'l1':
                    l1_penalty = utils.calculate_l1(model)
                    loss = loss + 0.001*l1_penalty
                loss.backward()
                model_opt.step()

                sofa_list.append(target)
                sofap_list.append(sofa_p)
                loss_t.append(loss)

            loss_avg = np.mean(torch.stack(loss_t, dim=0).cpu().detach().numpy())

            model.eval()
            y_list = []
            y_pred_list = []
            ti_list = []
            td_list = []
            id_list = []
            loss_val = []
            with torch.no_grad():  # validation does not require gradient

                for vitals, static, target, val_ids, key_mask in dev_dataloader:
                    if args.static_fusion == 'no_static':
                        if args.model_name == 'TCN':
                            sofap_t = model(vitals.to(device))
                        elif args.model_name == 'RNN':
                            # x_lengths have to be a 1d tensor 
                            td_transpose = vitals.to(device).transpose(1, 2)
                            x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                            sofap_t = model(td_transpose, x_lengths)
                        elif args.model_name == 'Transformer':
                            tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                            sofap_t = model(vitals.to(device), tgt_mask, key_mask.bool().to(device))
                    else:
                         sofap_t = model(vitals.to(device), static.to(device))
                    loss_v = utils.mse_maskloss(sofap_t, target.to(device), key_mask.to(device))
                    y_list.append(target.detach().numpy())
                    y_pred_list.append(sofap_t.cpu().detach().numpy())
                    loss_val.append(loss_v)
                    id_list.append(val_ids)

            loss_te = np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())
            if loss_te < best_loss:
                patience = 0
                best_loss = loss_te
                torch.save(model.state_dict(),
                            './checkpoints/' + workname + '/' + 'fold%d' % c_fold + '_best_loss.pt')
            else:
                patience += 1
                if patience >= 10:
                    print('Start next fold')
                    break
            print('Epoch %d, : Train loss is %.4f, test loss is %.4f' % (j, loss_avg, loss_te))
