from pytorchcv.model_provider import get_model as ptcv_get_model


def setupModel(chosenModel):
    if chosenModel == 'resnet20':
        return ptcv_get_model('resnet20_cifar10', pretrained=True)
    elif chosenModel == 'resnet110':
        return ptcv_get_model('resnet110_cifar10', pretrained=True)
    else:
        raise Exception("其他模型暂不支持! 请使用README中所述的已支持模型！")
