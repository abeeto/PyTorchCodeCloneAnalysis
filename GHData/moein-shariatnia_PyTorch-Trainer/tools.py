import matplotlib.pyplot as plt

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    
    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
        
class AvgMeterVector:
    def __init__(self, num_enteries=3, names=["Metric 1", "Metric 2", "Metric 3"]):
        self.num_enteries = num_enteries
        self.meters = [AvgMeter(names[i]) for i in range(num_enteries)]
    
    def update(self, values, counts):
        for i in range(self.num_enteries):
            meter = self.meters[i]
            meter.update(values[i], counts.get(i, 0))

    def __repr__(self):
        text = [meter for meter in self.meters]
        return str(text)


def _process_metrics(metrics, name):
    new_metrics = {}
    keys = metrics.keys()
    values = [value[name].avg for value in metrics.values()]
    for key, value in zip(keys, values):
        new_metrics[key] = value
    return new_metrics


def process_metrics(old_metrics, name):
    metrics = {"train": {}, "valid": {}}
    train_metrics = old_metrics['train']
    valid_metrics = old_metrics['valid']
    metrics['train'] = _process_metrics(train_metrics, name)
    metrics['valid'] = _process_metrics(valid_metrics, name)
    return metrics


def get_best(data, mode='min'):
    values = data.values()
    best_value = min(values) if mode == "min" else max(values)
    best_key = next(k for k, v in data.items() if v == best_value)
    return best_key, best_value


def plot_statistics(data, name="Loss", mode="min", file_name="stats.png"):
    train_data = data["train"]
    valid_data = data["valid"]
    epochs = train_data.keys()
    train_stats = train_data.values()
    valid_stats = valid_data.values()

    best_train = get_best(train_data, mode=mode)
    best_valid = get_best(valid_data, mode=mode)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plt.sca(ax)
    plt.plot(epochs, train_stats, c="k", label="train")
    plt.plot(epochs, valid_stats, c="r", label="valid")

    x_scale = max(epochs)
    y_scale = max(max(train_stats), max(valid_stats))
    if mode == "max":
        y_scale = - y_scale

    train_x_offset = best_train[0] - x_scale * 0.15
    train_y_offset = best_train[1] + y_scale * 0.3

    valid_x_offset = best_valid[0] + x_scale * 0.15
    valid_y_offset = best_valid[1] + y_scale * 0.3 

    plt.annotate(
        f"Best Valid {name}: {best_valid[1]:.4f}",
        xy=(best_valid[0], best_valid[1]),
        xytext=(valid_x_offset, valid_y_offset),
        ha="center",
        arrowprops=dict(facecolor="red", shrink=0.1),
        fontsize=12,
    )
    plt.annotate(
        f"Best Train {name}: {best_train[1]:.4f}",
        xy=(best_train[0], best_train[1]),
        xytext=(train_x_offset, train_y_offset),
        ha="center",
        arrowprops=dict(facecolor="black", shrink=0.1),
        fontsize=12,
    )

    plt.title(name)
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.legend()
    plt.tight_layout()

    fig.savefig(file_name, facecolor="white")
    return best_valid[1]
