"""
命令行执行模块
"""
import argparse

from torchrec.task.ITask import ITask
from torchrec.task.tasks import task_name_list, get_task_type


def main() -> None:
    """
    命令行运行主函数
    """

    # 初始参数
    init_parser = argparse.ArgumentParser(description='命令行主函数', add_help=False)
    init_parser.add_argument('--task_name', type=str, default='normal', help='任务类型', choices=task_name_list)
    init_args, init_extras = init_parser.parse_known_args()

    task: ITask = get_task_type(init_args.task_name.lower()).create_from_console()

    task.run()

    # # 如果日志文件、结果文件、模型文件参数为空，生成预定的路径
    # paras = sorted(vars(origin_args).items(), key=lambda kv: kv[0])
    # log_name_exclude = ['dataset', 'random_seed']
    # log_name_exclude.extend(runner_name.log_name_exclude_args)
    # log_name_exclude.extend(data_reader_name.log_name_exclude_args)
    # log_name_exclude.extend(model_name.log_name_exclude_args)
    # log_file_name = [origin_args.model_name, origin_args.dataset, str(origin_args.random_seed)] + \
    #                 [p[0].replace('_', '')[:3] + str(p[1] if p[0] != 'metrics' else p[1][0])
    #                  for p in paras if p[0] not in log_name_exclude]
    # log_file_name = '_'.join(item.replace(' ', '-').replace('_', '-').replace("'", '')
    #                          for item in log_file_name)
    # if origin_args.log_file == '':
    #     origin_args.log_file = os.path.join(LOG_DIR, '%s/%s.txt' % (init_args.model_name, log_file_name))
    # check_dir_and_mkdir(origin_args.log_file)
    # if origin_args.result_file == '':
    #     origin_args.result_file = os.path.join(RESULT_DIR, '%s/%s.npy' % (init_args.model_name, log_file_name))
    # check_dir_and_mkdir(origin_args.result_file)
    # if origin_args.model_file == '':
    #     origin_args.model_file = os.path.join(MODEL_DIR, '%s/%s.pt' % (init_args.model_name, log_file_name))
    # check_dir_and_mkdir(origin_args.model_file)


if __name__ == '__main__':
    main()
