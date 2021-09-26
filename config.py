import torch


class Config:
    # train info
    use_cude = True
    model_dir = "/home/godzilla/Train Result/"
    device = torch.device("cuda:0" if use_cude and torch.cuda.is_available() else "cpu")
    train_data_proportion = 0.8
    log_freq = 200
    save_freq = 1000
    loss_pic_freq = 10000

    # general attribute
    batch_size = 128
    learning_rate = 0.0001
    max_iteration = 500000
    noise_factor = 0.001

    # data structure
    pos_dim = 23 * 3  # 69
    root_pos_dim = 3  # 3
    root_rot_dim = 1  # 1
    contact_dim = 4  # 4
    velocity_dim = 24 * 3  # 72
    vel_factor_dim = 5
    # acceleration_dim = 24 * 3  # 72
    left_foot = [2, 3]
    right_foot = [6, 7]
    joint_num = 23  # exclude root

    # prediction model
    p_min = 7  # minimum transition length is (p_min - 2)
    p_max = 72  # maximum transition length is (p_max - 2)   82

    win_step_factor = 0.1  # window step: math.ceil(win_step_factor * p_num)

    key_num = 5
    trajectory_size = 7
    velocity_control_size = 7

    # prediction network
    state_encoder_input_size = pos_dim + root_pos_dim + root_rot_dim + contact_dim  # 77
    derivative_encoder_input_size = velocity_dim  # 72
    target_encoder_input_size = pos_dim + root_pos_dim + root_rot_dim  # 73
    trajectory_encoder_input_size = 3 * trajectory_size
    velocity_control_encoder_input_size = vel_factor_dim * velocity_control_size
    encoder_hidden_size = 512
    encoder_output_size = 256

    root_transformer_model_size = encoder_output_size
    root_transformer_output_size = encoder_output_size
    root_transformer_layer = 2

    vel_transformer_model_size = encoder_output_size
    vel_transformer_output_size = encoder_output_size
    vel_transformer_layer = 2

    lstm1_input_size = encoder_output_size * 2 + root_transformer_output_size + vel_transformer_output_size
    lstm1_output_size = 256

    lstm2_input_size = encoder_output_size + lstm1_output_size
    lstm2_output_size = 256

    time_encoder_input_size = 4
    time_encoder_output_size = lstm1_output_size + encoder_output_size

    state_decoder_input_size = lstm2_output_size
    root_decoder_input_size = lstm2_output_size

    decoder_hidden1_size = 512
    decoder_hidden2_size = 256

    state_decoder_output_size = pos_dim + contact_dim + velocity_dim - 3  # 211
    root_decoder_output_size = root_pos_dim + root_rot_dim + 3  # 10
    label_size = state_decoder_output_size + root_decoder_output_size
    state_de_out_part1 = pos_dim
    state_de_out_part2 = contact_dim + velocity_dim - 3
    root_de_out_part1 = root_pos_dim + root_rot_dim
    root_de_out_part2 = 3

    # schedule sampling
    sampling_type = "schedule"
    schedule_sampling_decay = "exp"  # "exp" "sigmoid"  "linear"
    schedule_max_iteration = 100000
    ss_exp_k = 0.99995
    ss_sigmoid_k = 6000
    ss_linear_k = 1
    ss_linear_c = 0.000015
    schedule_sampling_limit = 0

