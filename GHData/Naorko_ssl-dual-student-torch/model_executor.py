import mean_teacher_baseline
import multiple_student


def execute_ms(args, context, train_loader, eval_loader):
    multiple_student.args = args

    return multiple_student.main(context, train_loader, eval_loader)


def execute_mt(args, context, train_loader, eval_loader):
    mean_teacher_baseline.args = args

    return mean_teacher_baseline.main(context, train_loader, eval_loader)


def execute_model(args, context, train_loader, val_loader):
    if args.model_arch in ['ms', 'msi']:
        return execute_ms(args, context, train_loader, val_loader)
    elif args.model_arch == 'mt':
        return execute_mt(args, context, train_loader, val_loader)
    else:
        raise Exception('Unknown model architecture')
