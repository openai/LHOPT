local initial_inner_hyperparameters = {
  lr: 1e-3,
  ombeta1: 0.1,
  ombeta2: 1e-2,
  epsilon: 1e-6,
  weight_decay: 1e-2,
  approximate_radam_warmup: false,
  group_clip_grad_norm: 0.8,
  scale_grad_norm_by_rolling_max: true,
  lamb_power: 1,
  inner_feature_frequency: 4,
  adam_update_include_weight_decay: true,
  lamb_min_trust: 1e-3,
  grad_clip_ombeta: 1e-2,
  denominator_norm: 'inf',
  lamb_ignore_vectors: true,
  lamb_update_norm_ombeta: 0.05,
};

local reward_definition = {
  reward_mode: ['adam_baseline_eval_loss'],
  adam_reward_mode: ['power_curve_fit'],
  adam_reward_clip: [3],
  adam_reward_log: [true],
  power_curve_fit_npoints: [14],
  power_curve_fit_extrapolation: [2.0],
  adam_reward_add_rand_loss: [true],
  adam_baseline_optimizer: ['polyak'],
};

local train_task_definitions = [
  // NQM
  {
    key: 'combiner_dsl',
    dataset: {
      key: ['multistep_nqm'],
      outer_steps: {
        choice_type: 'uniform',
        lower_bound: 32,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
      inner_steps: {
        choice_type: 'uniform',
        lower_bound: 16,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
      std: {
        choice_type: 'uniform',
        lower_bound: 0.1,
        upper_bound: 3.0,
        scale: 'log',
      },
      conditioning_number: {
        choice_type: 'uniform',
        lower_bound: 100.0,
        upper_bound: 30000.0,
        scale: 'log',
      },
      dimensions: {
        choice_type: 'uniform',
        lower_bound: 10,
        upper_bound: 10000,
        scale: 'log',
        dtype: 'int',
      },
    },
    architecture: {
      key: ['constant'],
    },
    loss: {
      train: ['mse'],
    } + reward_definition,
    environment_reuse: 4,
  },
  // algo
  {
    key: 'combiner_dsl',
    weight: 20,
    dataset: {
      key: ['xor', 'binary_addition'],
      difficulty: {
        choice_type: 'uniform',
        lower_bound: 0.0,
        upper_bound: 1.0,
      },
      outer_steps: {
        choice_type: 'uniform',
        lower_bound: 2,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
      training_multiplier: {
        choice_type: 'uniform',
        lower_bound: 0.5,
        upper_bound: 2.0,
        scale: 'log',
      },
      batch_size: {
        choice_type: 'uniform',
        lower_bound: 32,
        upper_bound: 256,
        scale: 'log',
        dtype: 'int',
      },
    },
    architecture: {
      key: ['rnn'],
      rnn_hidden_size: {
        choice_type: 'uniform',
        lower_bound: 16,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
      rnn_num_layers: [1, 2],
      rnn_type: ['rnn', 'gru', 'lstm'],
      dropout_prob: {
        choice_type: 'uniform',
        lower_bound: 0.0,
        upper_bound: 0.5,
      },
    },
    loss: {
      train: ['cce'],
    } + reward_definition,
    environment_reuse: 4,
  },
  // a more standard MNIST setup
  {
    key: 'combiner_dsl',
    weight: 10,
    dataset: {
      key: ['mnist_cached', 'kmnist_cached', 'fashion_mnist_cached'],
      batch_size: {
        choice_type: 'uniform',
        lower_bound: 50,
        upper_bound: 200,
        scale: 'log',
        dtype: 'int',
      },
      num_epochs: std.range(1, 5),
      outer_steps: {
        choice_type: 'uniform',
        lower_bound: 2,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
    },
    architecture: {
      key: ['mlp', 'cnn'],
      normalization: [null, 'bn', 'ln'],
      dropout_prob: {
        choice_type: 'uniform',
        lower_bound: 0.0,
        upper_bound: 0.2,
      },
      activation: ['relu', 'elu', 'leaky_relu', 'very_leaky_relu', 'prelu'],
      mlp_num_layers: std.range(1, 7),
      mlp_hidden_size: {
        choice_type: 'uniform',
        lower_bound: 32,
        upper_bound: 512,
        scale: 'log',
        dtype: 'int',
      },
      width: {
        choice_type: 'uniform',
        lower_bound: 1,
        upper_bound: 4,
        scale: 'log',
        dtype: 'int',
      },
    },
    loss: {
      train: ['cce'],
    } + reward_definition,
    environment_reuse: 4,
  },
  // MNIST with a ton of randomness
  {
    key: 'combiner_dsl',
    weight: 10,
    dataset: {
      key: ['mnist_cached', 'kmnist_cached', 'fashion_mnist_cached'],
      batch_size: {
        choice_type: 'uniform',
        lower_bound: 50,
        upper_bound: 200,
        scale: 'log',
        dtype: 'int',
      },
      num_epochs: std.range(1, 5),
      outer_steps: {
        choice_type: 'uniform',
        lower_bound: 2,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
    },
    architecture: {
      key: ['mlp'],
      normalization: [null, 'bn', 'ln'],
      dropout_prob: {
        choice_type: 'uniform',
        lower_bound: 0.0,
        upper_bound: 0.5,
      },
      activation: ['relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu', 'very_leaky_relu', 'prelu', 'celu', 'rrelu'],
      mlp_num_layers: std.range(1, 7),
      mlp_hidden_size: {
        choice_type: 'uniform',
        lower_bound: 32,
        upper_bound: 512,
        scale: 'log',
        dtype: 'int',
      },
      width: {
        choice_type: 'uniform',
        lower_bound: 1,
        upper_bound: 4,
        scale: 'log',
        dtype: 'int',
      },
    },
    loss: {
      train: ['cce', 'mse', 'mae', 'huber'],
    } + reward_definition,
    environment_reuse: 4,
  },
  // text datasets
  {
    key: 'combiner_dsl',
    weight: 10,
    dataset: {
      key: ['sst'],
      outer_steps: {
        choice_type: 'uniform',
        lower_bound: 4,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
      epochs: std.range(1, 5),
      bptt_length: {
        choice_type: 'uniform',
        lower_bound: 8,
        upper_bound: 32,
        scale: 'log',
        dtype: 'int',
      },
      batch_size: {
        choice_type: 'uniform',
        lower_bound: 16,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
      valid_batches: [null],
    },
    architecture: {
      key: ['rnn'],
      rnn_hidden_size: {
        choice_type: 'uniform',
        lower_bound: 16,
        upper_bound: 64,
        scale: 'log',
        dtype: 'int',
      },
      rnn_num_layers: [1],
      rnn_type: ['rnn', 'gru', 'lstm'],
      dropout_prob: [0.0],
      embedding_size: {
        choice_type: 'uniform',
        lower_bound: 32,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
    },
    loss: {
      train: ['cce'],
    } + reward_definition,
    environment_reuse: 4,
  },
  {
    key: 'combiner_dsl',
    weight: 23,
    dataset: {
      key: ['tokenized-text-v1'],
      name: ['textmix'],
      vocab_size: [256, 2048, 16384],
      outer_steps: {
        choice_type: 'uniform',
        lower_bound: 32,
        upper_bound: 128,
        scale: 'log',
        dtype: 'int',
      },
      bptt_length: [16, 32, 64, 128],
      combined_context_batch_size: [512, 1024],
      valid_batches: [10],
    },
    architecture: {
      key: ['transformer'],
      embedding_size: [64, 128],
      num_layers: [1, 2, 3],
      num_hidden: [64, 128],
      num_heads: [2, 4, 8],
    },
    loss: {
      train: ['cce'],
    } + reward_definition,
    environment_reuse: 4,
  },
];

local uniform_10x_scale = {
  choice_type: 'uniform',
  lower_bound: 0.1,
  upper_bound: 10,
  scale: 'log',
};
local uniform_2x_scale = {
  choice_type: 'uniform',
  lower_bound: 0.5,
  upper_bound: 2.0,
  scale: 'log',
};

local action_definitions = [
  {
    key: 'configurable_restart_registers',
    num_registers: 3,
    sub_actions: ['load', 'save', 'swap'],
  },
  {
    key: 'control_hyperparameter_v2',
    name: 'lr',
    scales: [0.5, 0.707, 0.9, 1, 1.1, 1.414, 2.0],
    upper_bound: 1.0,
    lower_bound: 1e-8,
    features_definition: {
      pi: {
        scale_relative: true,
      },
      V: {
        noise_features: true,
      },
    },
    noise_definition: {
      initial_scale: {
        choice_type: 'uniform',
        lower_bound: 1e-2,
        upper_bound: 1e2,
        scale: 'log',
      },
    },
  },
  {
    key: 'control_hyperparameter_v2',
    name: 'weight_decay',
    scales: [0.5, 2.0],
    upper_bound: 1.0,
    lower_bound: 1e-8,
    features_definition: {
      pi: {
        scale_relative: true,
      },
      V: {
        noise_features: true,
      },
    },
    noise_definition: {
      initial_scale: uniform_10x_scale,
    },
  },
  {
    key: 'control_hyperparameter_v2',
    name: 'epsilon',
    scales: [0.5, 2.0],
    upper_bound: 1.0,
    lower_bound: 1e-8,
    features_definition: {
      pi: {
        scale_relative: true,
      },
      V: {
        noise_features: true,
      },
    },
    noise_definition: {
      initial_scale: uniform_10x_scale,
    },
  },
  {
    key: 'control_hyperparameter_v2',
    name: 'ombeta2',
    scales: [0.5, 2.0],
    upper_bound: 0.1,
    lower_bound: 1e-5,
    features_definition: {
      pi: {
        scale_relative: true,
      },
      V: {
        noise_features: true,
      },
    },
    noise_definition: {
      initial_scale: uniform_10x_scale,
    },
  },
  {
    key: 'control_hyperparameter_v2',
    name: 'ombeta1',
    scales: [0.5, 2.0],
    upper_bound: 1.0,
    lower_bound: 1e-5,
    features_definition: {
      pi: {
        scale_relative: true,
      },
      V: {
        noise_features: true,
      },
    },
    noise_definition: {
      initial_scale: uniform_10x_scale,
    },
  },
  {
    key: 'control_hyperparameter_v2',
    name: 'group_clip_grad_norm',
    logit_shifts: [-1, -0.3, 0.3, 1],
    lower_bound: 1e-10,
    upper_bound: 1 - 1e-10,
    features_definition: {
      pi: {
        logit_relative: true,
      },
      V: {
        noise_features: true,
      },
    },
    noise_definition: {
      initial_logit_shift: {
        choice_type: 'uniform',
        lower_bound: -1,
        upper_bound: 1,
      },
    },
  },
  {
    key: 'control_hyperparameter_v2',
    name: 'grad_clip_ombeta',
    scales: [0.5, 1.0, 2.0],
    upper_bound: 1.0,
    lower_bound: 1e-5,
    features_definition: {
      pi: {
        scale_relative: true,
      },
      V: {
        noise_features: true,
      },
    },
    noise_definition: {
      initial_scale: uniform_2x_scale,
    },
  },
  {
    key: 'control_hyperparameter_v2',
    name: 'lamb_update_norm_ombeta',
    scales: [1.0 / 1.5, 1.0, 1.5],
    upper_bound: 1.0,
    lower_bound: 1e-5,
    features_definition: {
      pi: {
        scale_relative: true,
      },
      V: {
        noise_features: true,
      },
    },
    noise_definition: {
      initial_scale: uniform_2x_scale,
    },
  },
];

local cdf_features = [{ key: 'integral_cdf', end_weight: w } for w in [1.25, 2.5, 5, 10, 20]];
local pi_feature_definitions = [
  {
    key: 'progress',
    raw: [{ key: 'identity' }],
  },
  {
    key: 'inner_feature_ops',
    op_names: [
      'did_grad_clip',
      'sqrt_v_hat_gt_epsilon',
      'mean_log_abs_adam_pre_lr_update',
      'log_two_regimes_scale',
      'log_adam_update_norm_over_param_norm',
      'log_noise_scale',
      'cosine_similarity_grad_momentum',
      'cosine_similarity_grad_param',
      'log_lamb',
      'cosine_similarity_grad_update',
      'logit_cdf_cosine_similarity_grad_momentum',
      'logit_cdf_cosine_similarity_grad_update',
      'cdf_cosine_similarity_grad_param',
    ],
    raw: [{ key: 'identity' }] + cdf_features,
  },
  {
    key: 'action_vector',
  },
  {
    key: 'loss_ratio',
    log_ratio: [{ key: 'is_nan' }, { key: 'tanh' }] + cdf_features,
  },
  {
    key: 'outer_update_ratio',
    log_ratio: [{ key: 'tanh' }] + cdf_features,
  },
  {
    key: 'outer_norm_ratio',
    log_ratio: [{ key: 'tanh' }] + cdf_features,
  },
  {
    key: 'log_loss',
    train_loss: [{ key: 'is_nan' }, { key: 'is_inf' }] + cdf_features,
    eval_loss: [{ key: 'is_nan' }, { key: 'is_inf' }] + cdf_features,
  },
  {
    key: 'restart_checkpoint_loss',
    num_locations: 3,
  },
  {
    key: 'restart_checkpoint_progress',
    num_locations: 3,
  },
  {
    key: 'action_features',
    action_features_key: 'pi',
  },
];

local V_feature_definitions = [
  {
    key: 'task_vector',
  },
  {
    key: 'action_features',
    action_features_key: 'V',
  },
  {
    key: 'reward',
    raw_reward: [{ key: 'identity' }] + cdf_features,
    reward: [{ key: 'identity' }] + cdf_features,
  },
  {
    key: 'log_loss',
    train_loss: [{ key: 'identity' }],
    eval_loss: [{ key: 'identity' }],
  },
  {
    key: 'baseline_reward',
    baseline_max_reward: [{ key: 'identity' }],
    baseline_median_reward: [{ key: 'identity' }],
    baseline_mean_reward: [{ key: 'identity' }],
  },
  {
    key: 'power_curve_fit',
    log_neg_alpha: [{ key: 'tanh', scale: 10 }],
  },
];

local policy = {
  net: {
    hidden_state_sizes: [256],
    pre_lstm_hidden_sizes: [256],
    post_lstm_hidden_sizes: [256],
    chitecture: 'lstm_ln',
    input_normalization: {
      norm_type: 'ewma',
      beta: 0.999,
      per_element_update: false,
      epsilon: 1e-5,
    },
    input_norm_clip: 2,
  },
  value_head_opts: {
    norm_type: 'ewma',
    norm_kwargs: {
      per_element_update: true,
      epsilon: 1e-8,
    },
  },
  actions: {
    head_architecture: 'residual',
    actions: action_definitions,
  },
  features: {
    pi: pi_feature_definitions,
    V: V_feature_definitions,
  },
  polyak_averaging: 0.99,
  initial_inner_hyperparameters: initial_inner_hyperparameters,
};

{
  ppo_hyperparams: {
    clip_param: 0.2,
    ent_coef: 1e-2,
    learning_rate: 2e-4,
    beta2: 0.999,
    max_grad_norm: 5,
    vf_coef: 0.5,
    gae_lambda: 0.95,
    gamma: 1.0,
    batch_size: 128,
    batches_per_iteration: 4,
    buffer_size: 768,
    max_sample_reuse: 4,
    max_staleness: 4,
  },
  policy: policy,
  inner_tasks: train_task_definitions,
}
