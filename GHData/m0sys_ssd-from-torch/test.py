import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        ## batch_size=512,
        batch_size=32,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # Init criterion.
    model.priors_cxcy = model.priors_cxcy.to(device)
    loss_fn = loss_fn(priors_cxcy=model.priors_cxcy, device=device).to(device)

    with torch.no_grad():
        det_boxes = []
        det_labels = []
        det_scores = []
        true_boxes = []
        true_labels = []
        true_difficulties = []
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            pred_locs, pred_scores = model(images)

            #
            # save sample images, or do something with output here
            #

            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                pred_locs,
                pred_scores,
                min_score=0.01,
                max_overlap=0.45,
                top_k=200,
                device=device,
            )

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

            # computing loss, metrics on test set
            loss = loss_fn(pred_locs, pred_scores, boxes, labels)
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(
                    det_boxes_batch,
                    det_labels_batch,
                    det_scores_batch,
                    boxes,
                    labels,
                    difficulties,
                )

        for i, metric in enumerate(metric_fns):
            total_metrics[i] += metric(
                det_boxes,
                det_labels,
                det_scores,
                true_boxes,
                true_labels,
                true_difficulties,
            )
    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_fns)
        }
    )
    logger.info(log)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)
    main(config)
