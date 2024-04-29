import argparse
import sys
import torch
import torch.nn.functional as F


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_file',
        type=str,
        help='path for train file.'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        help='path for test file.'
    )
    # for model
    parser.add_argument(
        '--filter_sizes',
        default=[8, 12, 16, 20, 24, 28, 32, 36],
        type=int,
        nargs='+',
        help='Space seperated list of motif filter lengths. (ex, --filter_sizes 4 8 12)'
    )
    parser.add_argument(
        '--num_filters',
        default=[256, 256, 256, 256, 256, 256, 256, 256],
        type=int,
        nargs='+',
        help='Space seperated list of the number of convolution filters corresponding to length list. (ex, --num_filters 100 200 100)'
    )
    parser.add_argument(
        '--num_hidden',
        type=int,
        default=2000,
        help='Number of neurons in hidden layer.'
    )
    parser.add_argument(
        '--regularizer',
        type=float,
        default=0.001,
        help='(Lambda value / 2) of L2 regularizer on weights connected to last layer (0 to exclude).'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Rate for dropout.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=1000,
        help='Number of classes (families).'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=1000,
        help='Length of input sequences.'
    )
    # for learning
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to write checkpoint file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp',
        help='Directory for log data.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='Interval of steps for logging.'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=100,
        help='Interval of steps for save model.'
    )
    # test
    parser.add_argument(
        '--fine_tuning',
        type=bool,
        default=False,
        help='If true, weight on last layer will not be restored.'
    )
    parser.add_argument(
        '--fine_tuning_layers',
        type=str,
        nargs='+',
        default=["fc2"],
        help='Which layers should be restored. Default is ["fc2"].'
    )
    parser.add_argument(
        '--save_prediction',
        type=str,
        default=None,
        help='Path to save prediction'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=1,
        help='Top k prediction for predict'
    )
    parser.add_argument(
        '--predict_file',
        type=str,
        default=None,
        help='path for predict data.'
    )

    parser.add_argument(
        '--kfold', default=5, type=int, help='kfold'
    )

    try:
        FLAGS, unparsed = parser.parse_known_args()

    except:
        parser.print_help()
        sys.exit(1)

    # check validity
    assert (len(FLAGS.filter_sizes) == len(FLAGS.num_filters))

    return FLAGS


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    # print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def metric(logit, truth):
    with torch.no_grad():
        prob = F.softmax(logit, 1)
        value, top = prob.topk(3, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))
    return correct


def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs


GPCR_label = {'Adenosine': 0, 'Adrenergic': 1, 'Adrenocorticotropic': 2, 'Adrenomedullin': 3, 'Adrenoreceptor': 4,
              'Allatostatin': 5, 'AlphaFac': 6, 'Anaphylatoxin': 7, 'Angiotensin': 8, 'BLT2': 9, 'BOSS': 10,
              'Bombesin': 11, 'Bradykinin': 12, 'BrainSpec': 13, 'C5A': 14, 'Cadherin': 15, 'CalcLike': 16,
              'Calcitonin': 17, 'Cannabinoid': 18, 'Chemokine': 19, 'Cholecystokinin': 20, 'Corticotropin': 21,
              'Dopamine': 22, 'Duffy': 23, 'EMR1': 24, 'Endothelin': 25, 'ExtraCalc': 26, 'FollicleStim': 27,
              'GABA': 28, 'GRHR': 29, 'Galanin': 30, 'Gastric': 31, 'Glucagon': 32, 'GlutaMeta': 33,
              'Gonadotrophin': 34, 'Growth': 35, 'GrowthHorm': 36, 'Histamine': 37, 'Interleukin8': 38, 'Kiss1': 39,
              'Latrophilin': 40, 'LysoEdg2': 41, 'MelaninConc': 42, 'Melanocortin': 43, 'Melanocyte': 44, 'Melaton': 45,
              'Methuselah': 46, 'MuscAcetyl': 47, 'Muscarinicacetylcholine': 48, 'Neuromedin': 49, 'NeuromedinB-U': 50,
              'Neuropeptide': 51, 'NeuropeptideFF': 52, 'Neurotensin': 53, 'Octopamine': 54, 'Olfactory': 55,
              'Opoid': 56, 'Orexin': 57, 'Oxytocin': 58, 'PACAP': 59, 'Parathyroid': 60, 'Pheromone': 61,
              'Platelet': 62, 'Prokineticin': 63, 'Prolactin': 64, 'Prostacyclin': 65, 'Prostaglandin': 66,
              'Proteinase': 67, 'Purinergic': 68, 'PutPher': 69, 'Secretin': 70, 'Serotonin': 71, 'Somatostatin': 72,
              'SubstanceK': 73, 'SubstanceP': 74, 'Tachykinin': 75, 'Taste': 76, 'Thrombin': 77, 'Thyro': 78,
              'Thyrotropin': 79, 'Traceamine': 80, 'UrotensinII': 81, 'Vasoactive': 82, 'Vasopressin': 83,
              'Vasotocin': 84, 'cAMP': 85}

GPCR_family = {'Adenosine': 'ClassA', 'Adrenergic': 'ClassA', 'Adrenocorticotropic': 'ClassA',
               'Adrenomedullin': 'ClassA', 'Adrenoreceptor': 'ClassA', 'Allatostatin': 'ClassA', 'AlphaFac': 'ClassD',
               'Anaphylatoxin': 'ClassA', 'Angiotensin': 'ClassA', 'BLT2': 'ClassA', 'BOSS': 'ClassC',
               'Bombesin': 'ClassA', 'Bradykinin': 'ClassA', 'BrainSpec': 'ClassB', 'C5A': 'ClassA',
               'Cadherin': 'ClassB', 'CalcLike': 'ClassC', 'Calcitonin': 'ClassB', 'Cannabinoid': 'ClassA',
               'Chemokine': 'ClassA', 'Cholecystokinin': 'ClassA', 'Corticotropin': 'ClassB', 'Dopamine': 'ClassA',
               'Duffy': 'ClassA', 'EMR1': 'ClassB', 'Endothelin': 'ClassA', 'ExtraCalc': 'ClassC',
               'FollicleStim': 'ClassA', 'GABA': 'ClassC', 'GRHR': 'ClassA', 'Galanin': 'ClassA', 'Gastric': 'ClassB',
               'Glucagon': 'ClassB', 'GlutaMeta': 'ClassC', 'Gonadotrophin': 'ClassA', 'Growth': 'ClassA',
               'GrowthHorm': 'ClassB', 'Histamine': 'ClassA', 'Interleukin8': 'ClassA', 'Kiss1': 'ClassA',
               'Latrophilin': 'ClassB', 'LysoEdg2': 'ClassA', 'MelaninConc': 'ClassA', 'Melanocortin': 'ClassA',
               'Melanocyte': 'ClassA', 'Melaton': 'ClassA', 'Methuselah': 'ClassB', 'MuscAcetyl': 'ClassA',
               'Muscarinicacetylcholine': 'ClassA', 'Neuromedin': 'ClassA', 'NeuromedinB-U': 'ClassA',
               'Neuropeptide': 'ClassA', 'NeuropeptideFF': 'ClassA', 'Neurotensin': 'ClassA', 'Octopamine': 'ClassA',
               'Olfactory': 'ClassA', 'Opoid': 'ClassA', 'Orexin': 'ClassA', 'Oxytocin': 'ClassA', 'PACAP': 'ClassB',
               'Parathyroid': 'ClassB', 'Pheromone': 'ClassC', 'Platelet': 'ClassA', 'Prokineticin': 'ClassA',
               'Prolactin': 'ClassA', 'Prostacyclin': 'ClassA', 'Prostaglandin': 'ClassA', 'Proteinase': 'ClassA',
               'Purinergic': 'ClassA', 'PutPher': 'ClassC', 'Secretin': 'ClassB', 'Serotonin': 'ClassA',
               'Somatostatin': 'ClassA', 'SubstanceK': 'ClassA', 'SubstanceP': 'ClassA', 'Tachykinin': 'ClassA',
               'Taste': 'ClassC', 'Thrombin': 'ClassA', 'Thyro': 'ClassA', 'Thyrotropin': 'ClassA',
               'Traceamine': 'ClassA', 'UrotensinII': 'ClassA', 'Vasoactive': 'ClassB', 'Vasopressin': 'ClassA',
               'Vasotocin': 'ClassA', 'cAMP': 'ClassE'}

GPCR_subfamily = {'Adenosine': 'ClassA_Nucleotide', 'Adrenergic': 'ClassA_Adrenergic',
                  'Adrenocorticotropic': 'ClassA_Peptide', 'Adrenomedullin': 'ClassA_Peptide',
                  'Adrenoreceptor': 'ClassA_Amine', 'Allatostatin': 'ClassA_Peptide', 'AlphaFac': 'ClassD_Pheromone',
                  'Anaphylatoxin': 'ClassA_Anaphylatoxin', 'Angiotensin': 'ClassA_Peptide', 'BLT2': 'ClassA_Leuko',
                  'BOSS': 'ClassC_BOSS', 'Bombesin': 'ClassA_Peptide', 'Bradykinin': 'ClassA_Peptide',
                  'BrainSpec': 'ClassB_BrainSpec', 'C5A': 'ClassA_Peptide', 'Cadherin': 'ClassB_Cadherin',
                  'CalcLike': 'ClassC_CalcSense', 'Calcitonin': 'ClassB_Calcitonin',
                  'Cannabinoid': 'ClassA_Cannabinoid', 'Chemokine': 'ClassA_Peptide',
                  'Cholecystokinin': 'ClassA_Peptide', 'Corticotropin': 'ClassB_Corticotropin',
                  'Dopamine': 'ClassA_Amine', 'Duffy': 'ClassA_Peptide', 'EMR1': 'ClassB_EMR1',
                  'Endothelin': 'ClassA_Peptide', 'ExtraCalc': 'ClassC_CalcSense', 'FollicleStim': 'ClassA_Hormone',
                  'GABA': 'ClassC_GABA', 'GRHR': 'ClassA_GRHR', 'Galanin': 'ClassA_Peptide',
                  'Gastric': 'ClassB_Gastric', 'Glucagon': 'ClassB_Glucagon', 'GlutaMeta': 'ClassC_GlutaMeta',
                  'Gonadotrophin': 'ClassA_Hormone', 'Growth': 'ClassA_Thyro', 'GrowthHorm': 'ClassB_GrowthHorm',
                  'Histamine': 'ClassA_Amine', 'Interleukin8': 'ClassA_Interleukin8', 'Kiss1': 'ClassA_Peptide',
                  'Latrophilin': 'ClassB_Latrophilin', 'LysoEdg2': 'ClassA_Lyso', 'MelaninConc': 'ClassA_Peptide',
                  'Melanocortin': 'ClassA_Peptide', 'Melanocyte': 'ClassA_Peptide', 'Melaton': 'ClassA_Melaton',
                  'Methuselah': 'ClassB_Methuselah', 'MuscAcetyl': 'ClassA_Amine',
                  'Muscarinicacetylcholine': 'ClassA_Amine', 'Neuromedin': 'ClassA_Peptide',
                  'NeuromedinB-U': 'ClassA_Peptide', 'Neuropeptide': 'ClassA_Peptide',
                  'NeuropeptideFF': 'ClassA_Peptide', 'Neurotensin': 'ClassA_Peptide', 'Octopamine': 'ClassA_Amine',
                  'Olfactory': 'ClassA_Olfactory', 'Opoid': 'ClassA_Peptide', 'Orexin': 'ClassA_Peptide',
                  'Oxytocin': 'ClassA_Peptide', 'PACAP': 'ClassB_PACAP', 'Parathyroid': 'ClassB_Parathyroid',
                  'Pheromone': 'ClassC_CalcSense', 'Platelet': 'ClassA_Platelet', 'Prokineticin': 'ClassA_Peptide',
                  'Prolactin': 'ClassA_Peptide', 'Prostacyclin': 'ClassA_Prostanoid',
                  'Prostaglandin': 'ClassA_Prostanoid', 'Proteinase': 'ClassA_Peptide',
                  'Purinergic': 'ClassA_Nucleotide', 'PutPher': 'ClassC_PutPher', 'Secretin': 'ClassB_Secretin',
                  'Serotonin': 'ClassA_Amine', 'Somatostatin': 'ClassA_Peptide', 'SubstanceK': 'ClassA_Peptide',
                  'SubstanceP': 'ClassA_Peptide', 'Tachykinin': 'ClassA_Peptide', 'Taste': 'ClassC_Taste',
                  'Thrombin': 'ClassA_Peptide', 'Thyro': 'ClassA_Thyro', 'Thyrotropin': 'ClassA_Hormone',
                  'Traceamine': 'ClassA_Amine', 'UrotensinII': 'ClassA_Peptide', 'Vasoactive': 'ClassB_Vasocactive',
                  'Vasopressin': 'ClassA_Peptide', 'Vasotocin': 'ClassA_Peptide', 'cAMP': 'ClassE_cAMP'}
