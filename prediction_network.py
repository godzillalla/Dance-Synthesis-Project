# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import torch
import time
import math
import random
import numpy as np
import torch.nn as nn
from utils import get_latest_weight_file, save_model_loss_info, save_loss_pic, load_dict
from config import Config
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from predition_loss import ReconstructionLoss, BoneLoss, VelocityLoss, ContactLoss, SmoothLoss, KeyframeLoss

from transformer.Models import Transformer


class Prediction(nn.Module):
    def __init__(self, hparams):
        super(Prediction, self).__init__()
        self.hparams = hparams
        self.state_fc1 = nn.Linear(hparams.state_encoder_input_size, hparams.encoder_hidden_size)
        self.state_fc2 = nn.Linear(hparams.encoder_hidden_size, hparams.encoder_output_size)

        self.derivative_fc1 = nn.Linear(hparams.derivative_encoder_input_size, hparams.encoder_hidden_size)
        self.derivative_fc2 = nn.Linear(hparams.encoder_hidden_size, hparams.encoder_output_size)

        self.target_fc1 = nn.Linear(hparams.target_encoder_input_size, hparams.encoder_hidden_size)
        self.target_fc2 = nn.Linear(hparams.encoder_hidden_size, hparams.encoder_output_size)

        self.root_converter = nn.Linear(3, hparams.root_transformer_model_size)
        self.root_transformer = Transformer(
            d_word_vec=hparams.root_transformer_model_size, device=hparams.device,
            n_position=hparams.trajectory_size, d_k=32, d_v=32,
            d_model=hparams.root_transformer_model_size, d_inner=hparams.root_transformer_model_size,
            n_layers=hparams.root_transformer_layer,
            n_head=8, batch_size=hparams.batch_size, dropout=0.1,
            out_dim=hparams.root_transformer_output_size).to(hparams.device)

        self.vel_converter = nn.Linear(hparams.vel_factor_dim, hparams.vel_transformer_model_size)
        self.vel_transformer = Transformer(
            d_word_vec=hparams.vel_transformer_model_size, device=hparams.device,
            n_position=hparams.velocity_control_size, d_k=32, d_v=32,
            d_model=hparams.vel_transformer_model_size, d_inner=hparams.vel_transformer_model_size,
            n_layers=hparams.vel_transformer_layer,
            n_head=8, batch_size=hparams.batch_size, dropout=0.1,
            out_dim=hparams.vel_transformer_output_size).to(hparams.device)

        self.time_fc1 = nn.Linear(hparams.time_encoder_input_size, hparams.time_encoder_output_size)

        self.lstm1 = nn.LSTM(hparams.lstm1_input_size, hparams.lstm1_output_size, batch_first=True)
        self.lstm2 = nn.LSTM(hparams.lstm2_input_size, hparams.lstm2_output_size, batch_first=True)

        self.state_de_fc1 = nn.Linear(hparams.state_decoder_input_size, hparams.decoder_hidden1_size)
        self.state_de_fc2 = nn.Linear(hparams.decoder_hidden1_size, hparams.decoder_hidden2_size)
        self.state_de_fc3 = nn.Linear(hparams.decoder_hidden2_size, hparams.state_decoder_output_size)

        self.root_de_fc1 = nn.Linear(hparams.root_decoder_input_size, hparams.decoder_hidden1_size)
        self.root_de_fc2 = nn.Linear(hparams.decoder_hidden1_size, hparams.decoder_hidden2_size)
        self.root_de_fc3 = nn.Linear(hparams.decoder_hidden2_size, hparams.root_decoder_output_size)

        self.prelu = nn.PReLU()

        self.model_dir = hparams.model_dir

    def init_hidden(self, batch_size, device):
        self.lstm1_h = torch.zeros(1, batch_size, self.hparams.lstm1_output_size).to(device)
        self.lstm1_c = torch.zeros(1, batch_size, self.hparams.lstm1_output_size).to(device)
        self.lstm2_h = torch.zeros(1, batch_size, self.hparams.lstm2_output_size).to(device)
        self.lstm2_c = torch.zeros(1, batch_size, self.hparams.lstm2_output_size).to(device)

    def state_encoder(self, x):
        out = self.prelu(self.state_fc1(x))
        out = self.prelu(self.state_fc2(out))
        return out

    def derivative_encoder(self, x):
        out = self.prelu(self.derivative_fc1(x))
        out = self.prelu(self.derivative_fc2(out))
        return out

    def target_encoder(self, x):
        out = self.prelu(self.target_fc1(x))
        out = self.prelu(self.target_fc2(out))
        return out

    def trajectory_encoder(self, x):
        out = self.prelu(self.trajectory_fc1(x))
        out = self.prelu(self.trajectory_fc2(out))
        return out

    def time_encoder(self, x):
        out = self.prelu(self.time_fc1(x))
        return out

    def state_decoder(self, z):
        out = self.prelu(self.state_de_fc1(z))
        out = self.prelu(self.state_de_fc2(out))
        out = self.state_de_fc3(out)
        return out

    def root_decoder(self, z):
        out = self.prelu(self.root_de_fc1(z))
        out = self.prelu(self.root_de_fc2(out))
        out = self.root_de_fc3(out)
        return out

    def concatenate_output(self, z_in1, z_in2):
        z0, z1 = z_in1.split([self.hparams.state_de_out_part1,
                              self.hparams.state_de_out_part2], dim=-1)
        z3, z4 = z_in2.split([self.hparams.root_de_out_part1,
                              self.hparams.root_de_out_part2], dim=-1)
        z = torch.cat((z0, z3, z1, z4), dim=-1)
        return z

    def root_transformer_encoder(self, true_trajectory, x4):
        src_seq = self.root_converter(true_trajectory)
        trg_seq = self.root_converter(x4)
        out = self.root_transformer(src_seq, trg_seq)
        return out

    def vel_transformer_encoder(self, x5):
        x = self.vel_converter(x5)
        out = self.vel_transformer(x, x)
        return out

    def forward(self, x1, x2, x3, x4, x5, t, noise, true_trajectory):
        x1 = self.state_encoder(x1)  # [batch_size, seq_len, encoder_output_size]  [60, *, 256]
        x2 = self.derivative_encoder(x2)  # [batch_size, seq_len, encoder_output_size]

        x3 = self.target_encoder(x3)  # [batch_size, seq_len, encoder_output_size]
        x4 = self.root_transformer_encoder(true_trajectory, x4)

        x5 = self.vel_transformer_encoder(x5)

        t = self.time_encoder(t)

        x12 = torch.cat([x1, x2, x4, x5], dim=-1)
        x12 = x12 + noise
        x12, (self.lstm1_h, self.lstm1_c) = self.lstm1(x12, (self.lstm1_h, self.lstm1_c))

        x = torch.cat([x12, x3], dim=-1) + t
        x, (self.lstm2_h, self.lstm2_c) = self.lstm2(x, (self.lstm2_h, self.lstm2_c))

        z1 = self.state_decoder(x)
        z2 = self.root_decoder(x)
        z = self.concatenate_output(z1, z2)  # , z3)

        return z


class DanceDataset(Dataset):
    def __init__(self, train_x):
        self.train_data = train_x

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, item):
        return torch.tensor(self.train_data[item], dtype=torch.float32)


def get_train_data(data, config):
    state_input_size = config.state_encoder_input_size
    derivative_input_size = config.derivative_encoder_input_size
    target_input_size = config.target_encoder_input_size
    label_size = config.label_size
    vel_factor_size = config.vel_factor_dim
    state_derivative_input = data[:, :, :state_input_size + derivative_input_size]

    target_input = data[:, -1, :target_input_size]
    target_input = np.expand_dims(target_input, 1).repeat(data.shape[1], axis=1)

    train_x = np.concatenate((state_derivative_input, target_input), axis=-1)
    vel_factor_seq = data[:, :, state_input_size + derivative_input_size:
                                state_input_size + derivative_input_size + vel_factor_size]
    train_y = data[:, :, 0:label_size]
    train_x_y = np.concatenate((train_x, vel_factor_seq, train_y), axis=-1)
    return train_x_y


def divide_data(data, window, win_step):
    divided_data = []
    for i, unit in enumerate(data):
        frame_num = len(unit)
        index = 0
        for start_frame in range(0, frame_num - window + 1, win_step):
            end_frame = start_frame + window - 1
            index += 1
            divided_data.append(unit[start_frame:end_frame + 1])
    return np.array(divided_data)


def get_random_root_pos_factor(time_factor):
    random_factor = np.zeros(time_factor.shape)
    seq_len = time_factor.shape[0]
    delta = 1 / seq_len
    random_scale = delta
    last_value = 0.0
    for i in range(seq_len):
        if i == 0:
            random_factor[i] = random.uniform(0.0, delta + random_scale)
        elif i == seq_len - 1:
            random_factor[i] = 1.0
        else:
            min_value = max(delta * (i + 1) - random_scale, last_value)
            max_value = min(delta * (i + 1) + random_scale, delta * (i + 2))
            random_factor[i] = random.uniform(min_value, max_value)
        last_value = random_factor[i]
    return random_factor


def get_time_label(win):
    time_label = []
    for i in range(1, win):
        time_label.append(i / (win - 1))

    return np.array(time_label)


def get_teacher_forcing_ratio(it, config):
    if config.sampling_type == "teacher_forcing":
        return 1.0
    elif config.sampling_type == "schedule":
        if config.schedule_sampling_decay == "exp":
            scheduled_ratio = config.ss_exp_k ** it
        elif config.schedule_sampling_decay == "sigmoid":
            if it / config.ss_sigmoid_k > 700000:
                scheduled_ratio = 0.0
            else:
                scheduled_ratio = config.ss_sigmoid_k / \
                                  (config.ss_sigmoid_k + math.exp(it / config.ss_sigmoid_k))
        else:
            scheduled_ratio = config.ss_linear_k - config.ss_linear_c * it
        scheduled_ratio = max(config.schedule_sampling_limit, scheduled_ratio)
        return scheduled_ratio
    else:
        return 0.0


def test_schedule_sample(config):
    max_it = 150000
    ratio_list = []
    for i in range(1, max_it):  # [1, max_it - 1]
        ratio = get_teacher_forcing_ratio(i, config)
        ratio_list.append(ratio)

    x = np.arange(1, max_it, 1)
    plt.plot(x, ratio_list, color='green')
    plt.show()


def train_one_iteration(model, rec_criterion, bone_criterion, vel_criterion, contact_criterion,
                        smooth_criterion, key_criterion, log_out, train_x, train_y, _noise, time_label,
                        vel_factor_seq, _mean, _std, optimizer, loss_dict, sample_ratio, config):
    device = config.device
    _train_x1 = train_x[..., :config.state_encoder_input_size].to(device)
    _train_x2 = train_x[..., config.state_encoder_input_size:
                             config.state_encoder_input_size + config.derivative_encoder_input_size].to(device)
    _train_x3 = train_x[..., -config.target_encoder_input_size:].to(device)
    _train_y = train_y.to(device)

    _trajectory = _train_x1[..., config.pos_dim:config.pos_dim + config.root_pos_dim]
    _trajectory = torch.cat([_trajectory, _train_y[:, -1:, config.pos_dim:config.pos_dim + config.root_pos_dim]], dim=1)
    true_trajectory = _trajectory.clone()

    _vel_factor = vel_factor_seq.to(device)

    now_batch_size = _train_y.shape[0]

    root_positions = train_y[:, :, config.pos_dim:config.pos_dim + config.root_pos_dim]
    time_label = time_label[np.newaxis, :, np.newaxis]
    time_label = time_label.repeat(now_batch_size, axis=0)

    _time_label = torch.tensor(time_label, dtype=torch.float32)
    position_code = torch.cat((root_positions, _time_label), dim=-1)

    _position_code = position_code.to(device)

    model.init_hidden(now_batch_size, config.device)

    seq_len = _train_y.shape[1]
    predict_seq = torch.zeros(_train_y.shape).to(config.device)

    use_ground_truth = True
    ran = random.random()
    if ran > sample_ratio:
        use_ground_truth = False

    for i in range(seq_len):
        if i == 0 or use_ground_truth:
            x1 = _train_x1[:, i:i + 1].clone()
            x2 = _train_x2[:, i:i + 1].clone()

        else:
            x1 = predict_seq[:, i - 1:i, :config.state_encoder_input_size].clone()
            x2 = predict_seq[:, i - 1:i, config.state_encoder_input_size:].clone()
            _trajectory[:, i:i + 1, :] = predict_seq[:, i - 1:i,
                                         config.pos_dim:config.pos_dim + config.root_pos_dim].clone()

        x4 = torch.zeros([now_batch_size, config.trajectory_size, 3]).to(device)
        x5 = torch.zeros([now_batch_size, config.velocity_control_size, config.vel_factor_dim]).to(device)
        true_x4 = torch.zeros([now_batch_size, config.trajectory_size, 3]).to(device)
        k = int(config.trajectory_size / 2)
        temp = 0
        for j in range(i - k, i + k + 1):
            if j < 0:
                true_x4[:, temp:temp + 1, :] = true_trajectory[:, :1, :]
                x4[:, temp:temp + 1, :] = _trajectory[:, :1, :]
                x5[:, temp:temp + 1, :] = _vel_factor[:, :1, :]
            elif j > seq_len:
                true_x4[:, temp:temp + 1, :] = true_trajectory[:, -1:, :]
                x4[:, temp:temp + 1, :] = _trajectory[:, -1:, :]
                x5[:, temp:temp + 1, :] = _vel_factor[:, -1:, :]
            else:
                true_x4[:, temp:temp + 1, :] = true_trajectory[:, j:j + 1, :]
                x4[:, temp:temp + 1, :] = _trajectory[:, j:j + 1, :]
                x5[:, temp:temp + 1, :] = _vel_factor[:, j:j + 1, :]
            temp += 1

        x3 = _train_x3[:, i:i + 1]
        pos_code = _position_code[:, i:i + 1]
        noise = _noise[i:i + 1]

        pre_frame = model.forward(x1, x2, x3, x4, x5, pos_code, noise, true_x4)

        predict_seq[:, i:i + 1, :] = pre_frame

    optimizer.zero_grad()

    rec_loss = rec_criterion(predict_seq, _train_y)
    bone_loss = bone_criterion(predict_seq, _train_x1, _train_x2)
    vel_cons_loss = vel_criterion(predict_seq, _train_x1, _train_x2, _vel_factor)
    contact_loss = contact_criterion(predict_seq, _train_x1, _train_x2)
    key_loss = key_criterion(predict_seq, _train_x1, _train_y)
    smooth_loss = smooth_criterion(predict_seq, _train_x1, _train_y)

    total_loss = rec_loss + bone_loss + vel_cons_loss + smooth_loss + key_loss + contact_loss * 2

    total_loss.backward()
    optimizer.step()

    if log_out:
        loss_dict["rec_loss"].append(rec_loss.item())
        loss_dict["bone_loss"].append(bone_loss.item())
        loss_dict["vel_cons_loss"].append(vel_cons_loss.item())
        loss_dict["contact_loss"].append(contact_loss.item())
        loss_dict["smooth_loss"].append(smooth_loss.item())
        loss_dict["key_loss"].append(key_loss.item())
        loss_dict["total_loss"].append(total_loss.item())
        print("rec_loss:", rec_loss.detach().cpu().numpy(),
              "bone_loss:", bone_loss.detach().cpu().numpy(),
              "vel_cons_loss:", vel_cons_loss.detach().cpu().numpy(),
              "contact_loss:", contact_loss.detach().cpu().numpy(),
              "smooth_loss:", smooth_loss.detach().cpu().numpy(),
              "key_loss:", key_loss.detach().cpu().numpy(),
              "total_loss:", total_loss.detach().cpu().numpy())


def train_prediction(data_set, raw_data_info, parents, gt_bone_length, mean, std):
    config = Config()
    start_time = time.asctime(time.localtime(time.time()))
    print("start_time:", start_time)
    start = time.time()

    device = config.device
    print("train on", device, torch.cuda.is_available())

    _mean = torch.tensor(mean, dtype=torch.float32).to(device)
    _std = torch.tensor(std, dtype=torch.float32).to(device)
    _gt_bone_length = torch.tensor(gt_bone_length, dtype=torch.float32).to(device)

    model = Prediction(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    rec_criterion = ReconstructionLoss(config).to(device)
    bone_criterion = BoneLoss(_gt_bone_length, parents, _mean, _std, config).to(device)
    vel_criterion = VelocityLoss(_mean, _std, config).to(device)
    contact_criterion = ContactLoss(_mean, _std, config).to(device)
    smooth_criterion = SmoothLoss(config).to(device)
    key_criterion = KeyframeLoss(config).to(device)

    model.train()

    train_x_dim = config.state_encoder_input_size + config.derivative_encoder_input_size + \
                  config.target_encoder_input_size
    train_y_dim = config.label_size

    loss_dict = {"iteration": [], "rec_loss": [], "bone_loss": [], "vel_cons_loss": [], "smooth_loss": [],
                 "contact_loss": [],
                 "key_loss": [], "total_loss": []}

    latest_info_file, latest_it = get_latest_weight_file(config.model_dir)
    print("lateset_file:", latest_info_file)
    if latest_info_file is not None:
        checkpoint = torch.load(latest_info_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_dict = load_dict(config.model_dir + 'loss_%07d' % latest_it)
        print("Read the information of iteration %d successfully. \nContinue training..." % latest_it)

    p_min = config.p_min
    epoch = 0
    it = 0
    min_loss = 135.0
    iteration_num = np.ones(config.p_max + 1) * -1
    loss_pic_freq = config.loss_pic_freq
    while True:  # epoch
        epoch += 1
        p_max = config.p_min + epoch - 1
        if p_max > config.p_max:
            break
        if epoch > 5 and epoch % 5 == 0:
            p_min += 4
        print("epoch:", epoch, p_min, p_max)
        for win in range(p_min, p_max + 1):
            if iteration_num[win] != -1 and it + iteration_num[win] < latest_it:
                it += int(iteration_num[win])
                continue
            item = raw_data_info[win]
            win_step, noise = item[0], item[1]
            train_data = divide_data(data_set, win, win_step)
            train_x_y = get_train_data(train_data, config)
            print("train data shape:", train_data.shape, "train_x_y:", train_x_y.shape, " iteration:", it)
            train_loader = DataLoader(DanceDataset(train_x_y), batch_size=config.batch_size)
            iteration_num[win] = len(train_loader)
            if it + iteration_num[win] < latest_it:
                del train_loader
                it += int(iteration_num[win])
                continue
            _noise = torch.tensor(noise, dtype=torch.float32).to(config.device)
            time_label = get_time_label(win)
            for i, _data in enumerate(train_loader):  # batch
                it = it + 1
                sample_ratio = get_teacher_forcing_ratio(it, config)
                if it <= latest_it:
                    continue
                if it > config.max_iteration:
                    break
                log_out = False
                if it % config.log_freq == 0 or it == latest_it + 1:
                    print("Iteration: %08d/%08d, transition length: %d" % (it, config.max_iteration, _data.shape[1]))
                    log_out = True
                    loss_dict["iteration"].append(it)
                train_x = _data[:, :-1, :train_x_dim]
                vel_factor_seq = _data[..., train_x_dim:train_x_dim + config.vel_factor_dim]
                train_y = _data[:, 1:, -train_y_dim:]
                train_one_iteration(model, rec_criterion, bone_criterion, vel_criterion, contact_criterion,
                                    smooth_criterion, key_criterion, log_out, train_x, train_y, _noise, time_label,
                                    vel_factor_seq, _mean, _std, optimizer, loss_dict, sample_ratio, config)
                del train_x
                del vel_factor_seq
                del train_y
                cur_loss = loss_dict["total_loss"][len(loss_dict["total_loss"]) - 1]
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    save_model_loss_info(it, model, optimizer, loss_dict, config, min_loss)
                elif it % config.save_freq == 0:
                    save_model_loss_info(it, model, optimizer, loss_dict, config, cur_loss)
                if it % loss_pic_freq == 0:
                    save_loss_pic(it, loss_dict, config)
            del _noise
            del time_label
            del train_loader

    save_model_loss_info(it, model, optimizer, loss_dict, config, 0.0)
    save_loss_pic(it, loss_dict, config)
    end_time = time.asctime(time.localtime(time.time()))
    print("\n------------------------------end------------------------------")
    print("iteration:", it)
    print("start_time:", start_time, "end_time:", end_time)
    end = time.time()
    print('Running time: %f Minutes = %f Hours' % ((end - start) / 60, (end - start) / 60 / 60))


def test_one_interval(model, data, target, time_factor, vel_factor, config):

    now_batch_size = data.shape[0]
    seq_len = data.shape[1]
    model.init_hidden(now_batch_size, config.device)
    init_data = data[:, :1]
    noise = torch.tensor(np.zeros([now_batch_size, 1, config.lstm1_input_size]), dtype=torch.float32).to(config.device)

    x3 = target.unsqueeze(dim=1)
    trajectory = time_factor.clone()
    true_trajectory = time_factor.clone()

    time_label = get_time_label(time_factor.shape[1])
    time_label = time_label[np.newaxis, :, np.newaxis].repeat(now_batch_size, axis=0)
    time_label = torch.tensor(time_label, dtype=torch.float32).to(config.device)
    time_factor = time_factor[:, 1:]
    time_factor = torch.cat((time_factor, time_label), dim=-1)

    pred_seq = np.zeros((now_batch_size, seq_len, config.label_size))
    for time_step in range(seq_len):
        x1 = init_data[..., :config.state_encoder_input_size]
        x2 = init_data[..., config.state_encoder_input_size:]
        t = time_factor[:, time_step:time_step + 1, :]
        if time_step != 0:
            trajectory[:, time_step:time_step + 1, :] = init_data[...,
                                                        config.pos_dim:config.pos_dim + config.root_pos_dim]

        x4 = torch.zeros([now_batch_size, config.trajectory_size, 3]).to(config.device)
        x5 = torch.zeros([now_batch_size, config.velocity_control_size, config.vel_factor_dim]).to(config.device)
        true_x4 = torch.zeros([now_batch_size, config.trajectory_size, 3]).to(config.device)
        k = int(config.trajectory_size / 2)
        temp = 0
        for j in range(time_step - k, time_step + k + 1):
            if j < 0:
                true_x4[:, temp:temp + 1, :] = true_trajectory[:, :1, :]
                x4[:, temp:temp + 1, :] = trajectory[:, :1, :]
                x5[:, temp:temp + 1, :] = vel_factor[:, :1, :]
            elif j >= trajectory.shape[1]:
                true_x4[:, temp:temp + 1, :] = true_trajectory[:, -1:, :]
                x4[:, temp:temp + 1, :] = trajectory[:, -1:, :]
                x5[:, temp:temp + 1, :] = vel_factor[:, -1:, :]
            else:
                true_x4[:, temp:temp + 1, :] = true_trajectory[:, j:j + 1, :]
                x4[:, temp:temp + 1, :] = trajectory[:, j:j + 1, :]
                x5[:, temp:temp + 1, :] = vel_factor[:, j:j + 1, :]

            temp += 1

        init_data = model.forward(x1, x2, x3, x4, x5, t, noise, true_x4)

        pred_seq[:, time_step:time_step + 1] = init_data.cpu().detach().numpy()

    return pred_seq


def test_prediction(mask, time_factor, vel_factor, test_data, target, model_path):
    config = Config()
    device = config.device

    model = Prediction(config).to(device)

    latest_info_file, latest_it = get_latest_weight_file(model_path)
    if latest_info_file:
        print("Load latest file", latest_info_file)
        checkpoint = torch.load(latest_info_file)
        model.load_state_dict(checkpoint['model'])
    else:
        print("Error: No model parameters file in", model_path)
        exit(-1)

    model.eval()

    batch_size = test_data.shape[0]
    seq_len = len(mask)
    last = 0
    seq_index = 0
    last_frame = []
    test_data = test_data[..., :-config.vel_factor_dim]
    predict_seq = np.zeros((batch_size, seq_len, config.label_size))
    for i in range(1, len(mask)):
        if mask[i] == 1:
            _time_in = torch.tensor(time_factor[:, last:i + 1], dtype=torch.float32).to(device)
            _vel_factor = torch.tensor(vel_factor[:, last:i + 1], dtype=torch.float32).to(device)
            data_in = test_data[:, last:i]
            if last != 0:
                data_in[:, 0, config.target_encoder_input_size:] = last_frame[:, config.target_encoder_input_size:]
            predict_seq[:, last, :] = data_in[:, 0]

            _data_in = torch.tensor(data_in, dtype=torch.float32).to(device)
            _target = torch.tensor(target[:, seq_index], dtype=torch.float32).to(device)
            pred_seq = test_one_interval(model, _data_in, _target, _time_in, _vel_factor, config)
            predict_seq[:, i + 1 - pred_seq.shape[1]:i] = pred_seq[:, :-1]
            last_frame = pred_seq[:, -1]
            last = i
            seq_index += 1
    last_pos = test_data[:, -1]
    last_pos[config.target_encoder_input_size:] = last_frame[config.target_encoder_input_size:]
    predict_seq[:, -1] = last_pos
    return predict_seq
