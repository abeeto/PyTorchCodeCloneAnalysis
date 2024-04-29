from sklearn.metrics import precision_score,recall_score,f1_score
import torch

def evaluate_valid(model,data_iter):
    model.eval()
    y_trues = []
    y_preds = []

    with torch.no_grad():
        for input in data_iter:
            inputs = {
                "input_ids": input[0].cuda(),
                "token_type_ids": input[1].cuda(),
                "attention_mask": input[2].cuda()
            }
            logits = model(**inputs)
            predict = torch.max(logits, 1)[1].cpu().numpy()
            label= input[3].data.cpu().numpy()
            y_trues.extend(label.tolist())
            y_preds.extend(predict.tolist())

    p = precision_score(y_trues, y_preds)
    r = recall_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)

    return p, r, f1
def evaluate_test(model,data_iter):

    model.eval()
    y_preds = []

    with torch.no_grad():
        for input in data_iter:
            inputs = {
                "input_ids": input[0].cuda(),
                "token_type_ids": input[1].cuda(),
                "attention_mask": input[2].cuda()
            }
            logits = model(**inputs)
            predict = torch.max(logits, 1)[1].cpu().numpy()
            y_preds.extend(predict.tolist())
    return y_preds