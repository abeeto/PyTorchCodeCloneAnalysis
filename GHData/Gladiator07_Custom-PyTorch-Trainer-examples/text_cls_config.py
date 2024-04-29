import ml_collections as mlc

cfg = mlc.ConfigDict()
cfg.project = "Custom_Accelerate_Trainer_Tests" # wandb project name
cfg.log_to_wandb = True
cfg.train_batch_size = 32
cfg.val_batch_size = 32
cfg.model_ckpt = "roberta-base"
cfg.learning_rate = 3e-5

cfg.trainer_args = dict(
    output_dir="./outputs",
    num_train_epochs=5,
    gradient_accumulation_steps=1,
    max_grad_norm=1,
    mixed_precision="fp16",
    scheduler_type="cosine",
    num_warmup_steps=0.1,
    save_best_checkpoint=True,
    save_last_checkpoint=True,
    save_weights_only=True,
    metric_for_best_model="val/accuracy"
)



def get_config():
    return cfg