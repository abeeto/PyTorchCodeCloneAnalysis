# -*- coding: utf-8 -*-
"""https://pytorch.org/hub/pytorch_fairseq_roberta/
https://github.com/pytorch/fairseq/
"""
import torch


def main():
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta.eval()  # disable dropout (or leave in train mode to finetune)

    tokens = roberta.encode('Hello world!')
    assert tokens.tolist() == [0, 31414, 232, 328, 2]
    assert roberta.decode(tokens) == 'Hello world!'

    # Extract the last layer's features
    last_layer_features = roberta.extract_features(tokens)
    assert last_layer_features.size() == torch.Size([1, 5, 1024])

    # Extract all layer's features (layer 0 is the embedding layer)
    all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
    assert len(all_layers) == 25
    assert torch.all(all_layers[-1] == last_layer_features)

    # Download RoBERTa already finetuned for MNLI
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
    roberta.eval()  # disable dropout for evaluation

    with torch.no_grad():
        # Encode a pair of sentences and make a prediction
        tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
        prediction = roberta.predict('mnli', tokens).argmax().item()
        assert prediction == 0  # contradiction

        # Encode another pair of sentences
        tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
        prediction = roberta.predict('mnli', tokens).argmax().item()
        assert prediction == 2  # entailment

    roberta.register_classification_head('new_task', num_classes=3)
    logprobs = roberta.predict('new_task',
                               tokens)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)

    return


if __name__ == '__main__':
    main()
    print('main - done')