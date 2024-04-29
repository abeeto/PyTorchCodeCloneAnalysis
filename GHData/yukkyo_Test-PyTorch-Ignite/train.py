from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping


def write_metrics(metrics, writer, mode: str, epoch: int):
    """print metrics & write metrics to log"""
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    print(f"{mode} Results - Epoch: {epoch}  "
          f"Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}")
    writer.add_scalar(f"{mode}/avg_loss", avg_nll, epoch)
    writer.add_scalar(f"{mode}/avg_accuracy", avg_accuracy, epoch)


def score_function(engine):
    """
    ignite.handlers.EarlyStopping では指定スコアが上がると改善したと判定する。
    そのため今回のロスに -1 をかけたものを ignite.handlers.EarlyStopping オブジェクトに渡す
    """
    val_loss = engine.state.metrics['nll']
    return -val_loss


def train(epochs, model, train_loader, valid_loader,
          criterion, optimizer, writer, device, log_interval):
    # device: str であることに注意
    # この時点では Dataloader を与えていないことに注意
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(criterion)},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        i = (engine.state.iteration - 1) % len(train_loader) + 1
        if i % log_interval == 0:
            print(f"Epoch[{engine.state.epoch}] Iteration[{i}/{len(train_loader)}] "
                  f"Loss: {engine.state.output:.2f}")
            # engine.state.output は criterion(model(input)) を表す？
            writer.add_scalar("training/loss", engine.state.output,
                              engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        write_metrics(metrics, writer, 'training', engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        write_metrics(metrics, writer, 'validation', engine.state.epoch)

    # # Checkpoint setting
    # ./checkpoints/sample_mymodel_{step_number}
    # n_saved 個までパラメータを保持する
    handler = ModelCheckpoint(dirname='./checkpoints', filename_prefix='sample',
                              save_interval=2, n_saved=3, create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'mymodel': model})

    # # Early stopping
    handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)
