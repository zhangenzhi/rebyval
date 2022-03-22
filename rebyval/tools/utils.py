import os
import sys
import json
import tempfile
import string
import random
import ruamel.yaml as yaml
from colorama import Fore

from .constants import ERROR_INFO, NORMAL_INFO, WARNING_INFO

def check_mkdir(path):
    if not os.path.exists(path=path):
        print_warning("no such path: {}, but we made.".format(path))
        os.makedirs(path)
        

def get_format_time(cost_time):
    minute, second = divmod(cost_time, 60)
    hour, minute = divmod(minute, 60)
    formated_time = '{:d}h：{:d}m：{:.2f}s'.format(int(hour), int(minute),
                                                 second)
    return formated_time


def get_yml_content(file_path):
    '''Load yaml file content'''
    try:
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.Loader)
    except yaml.scanner.ScannerError as err:
        print_error('yaml file format error!')
        print_error(err)
        exit(1)
    except Exception as exception:
        print_error(exception)
        exit(1)
        
def save_yaml_contents(file_path, contents):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(contents, file)
    except Exception as exception:
        print_error(exception)
        exit(1)
        


def get_json_content(file_path):
    '''Load json file content'''
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except TypeError as err:
        print_error('json file format error!')
        print_error(err)
        return None


def print_error(*content):
    '''Print error information to screen'''
    print(Fore.RED + ERROR_INFO + ' '.join([str(c)
                                            for c in content]) + Fore.RESET)

def print_red(*content):
    '''Print information to screen in red'''
    print(Fore.RED + ' '.join([str(c) for c in content]) + Fore.RESET)

def print_green(*content):
    '''Print information to screen in green'''
    print(Fore.GREEN + ' '.join([str(c) for c in content]) + Fore.RESET)


def print_normal(*content):
    '''Print error information to screen'''
    print(NORMAL_INFO, *content)


def print_warning(*content):
    '''Print warning information to screen'''
    print(Fore.YELLOW + WARNING_INFO + ' '.join([str(c) for c in content]) +
          Fore.RESET)


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print_green('\t' * (indent) + key + ": ")
            print_dict(value, indent + 1)
        else:
            try:
                print('\t' * (indent) + str(key) + ": " + str(value))
            except:
                print_error("value can not print: ", type(value))
                raise


def generate_temp_dir():
    '''generate a temp folder'''

    def generate_folder_name():
        return os.path.join(
            tempfile.gettempdir(), 'rebyval',
            ''.join(random.sample(string.ascii_letters + string.digits, 8)))

    temp_dir = generate_folder_name()
    while os.path.exists(temp_dir):
        temp_dir = generate_folder_name()
    os.makedirs(temp_dir)
    return temp_dir


def check_tensorboard_version():
    try:
        import tensorboard
        return tensorboard.__version__
    except:
        print_error('import tensorboard error!')
        exit(1)


def get_cvr_sample_num(dir_list):
    total_num = 0
    for part_dir in dir_list:
        part_dir_up = os.path.split(part_dir)[0]
        num_file = os.path.join(part_dir_up, 'filter_cnt', 'part-00000')
        with open(num_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                num = (line.split('\t')[1]).split('\n')[0]
                total_num += int(num)
    return total_num


def auto_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_train_valid_test_dir_local(root_dir, date='20200721'):
    train_dirs = []
    valid_dirs = []
    test_dirs = []
    path = os.path.join(root_dir, date, 'online_sample', 'tfRecord')
    train_dirs.append(path)
    valid_dirs.append(path)
    test_dirs.append(path)

    return train_dirs, valid_dirs, test_dirs


def write_log(file, msg):
    with open(file, 'a+') as f:
        f.write(msg)
        f.write('\n')


# Copied from CTR group
def calculate_auc(label_list, pred_list):
    """
    AUC 为 ROC 曲线下方的面积,
    ROC 曲线纵轴为 TPR, 即将正样本预测为正样本的概率,
    ROC 曲线横轴为 FPR, 即将负样本预测为正样本的概率,
    如果分类器对正负样本没有区分能力, TPR = FPR, 则 AUC = 0.5,
    我们希望分类器的 TPR 大于 FPR, AUC > 0.5.

    ROC 曲线的趋势:
    随着选取阈值的减小, 越来越多的样本被划分成正样本, TPR 与 FPR 同时增大, 直至 (1, 1),
    反之 TPR 与 FPR 同时减小, 直至 (0, 0), 或者 (0, b),
    因为如果模型性能较好, 当阈值较大时, 0 < TRP <= 1, 而 FPR = 0, 即没有负样本被预测为正样本,
    此时 ROC 曲线的左侧起点不为 (0, 0), 而是 (0, b) 点.
    理想目标: 可以找到一个完美的阈值, 划分正样本与负样本, 此时 ROC 曲线为 TPR = 1, 则 AUC = 1

    :param label_list: 真实类别
    :param pred_list: 预测类别, 预测类别为 1 的为 Positive (阳性), 预测类别为 0 的为 Negative (阴性)
    :return: AUC
    """

    # 将预测值降序排列, 得到新序列的索引值
    i_sorted = sorted(range(len(pred_list)),
                      key=lambda i: pred_list[i],
                      reverse=True)

    auc_temp = 0.0
    logloss_temp = 0.0
    rmse_temp = 0.0

    # 初始化
    tp = 0  # sgn(pred-thr) == 1 == label, 为 True Positive 真阳
    fp = 0  # sgn(pred-thr) == 1 != label, 为 False Positive 假阳

    # ROC 曲线中上一次出现过的新值
    tp_prev = 0
    fp_prev = 0
    threshold = -sys.maxsize

    for i in i_sorted:
        if threshold != pred_list[i]:
            # 使用梯形法则计算连接两个输入点的线下面积, (上底+下底)*高/2
            auc_temp += (tp + tp_prev) * (fp - fp_prev) / 2.0
            tp_prev = tp
            fp_prev = fp
            threshold = pred_list[i]

        if label_list[i] == 1:
            tp = tp + 1
        else:
            fp = fp + 1

    auc_temp += (fp - fp_prev) * (tp + tp_prev) / 2.0

    # 将频数转化为频率, 如果 tp fp 其中之一为零, 则会报错
    auc = auc_temp / (tp * fp)

    return auc


if __name__ == '__main__':
    print_error("lalalal")
