#-*- coding : utf-8-*-
# coding:unicode_escape
from optparse import OptionParser
import os
import sys
sys.path.append(sys._MEIPASS)
AL_PY_DEPS = os.environ['AL_PY_DEPS']
assert os.path.exists(AL_PY_DEPS), 'AL_PY_DEPS must be set up correctly'
sys.path.append(AL_PY_DEPS)
from ctypes.wintypes import DWORD, BOOL, HANDLE
import torch
from pydicom import config
config.settings.reading_validation_mode = config.IGNORE

from src.IMSE import IMSE_img2img

if __name__ == '__main__':
    p = OptionParser()
    p.add_option('--source_path', '-s', action="store", type="string", dest="source_path", help="source path!")
    p.add_option('--target_path', '-t', action="store", type="string", dest="target_path", help="target path!")
    p.add_option('--output_path', '-o', action="store", type="string", dest="output_path", help="out path!")
    p.add_option('--log_path', '-l', action="store", type="string", dest="log_path", help="log path!")
    p.add_option('--weight_path', '-w', action="store", type="string", dest="weight_path", help="weight path!")
    p.add_option('--use_roi', '-r', action="store", type="string", dest="use_roi", help="rigitd rot matrix")
    p.add_option('--devices', '-d',  type=int, default= 0,dest="devices", help="GPU number")

    options, arguments = p.parse_args()

    for t in ['source_path', 'target_path', 'weight_path']:
        t_value = getattr(options, t)
        if t_value is None:
            raise ValueError(f'missing required parameter: {t}')
        if not os.path.exists(t_value):
            raise ValueError(f'{t}: [{t_value}] is not valid')
    for t in ['output_path', 'log_path']:
        t_value = getattr(options, t)
        if t_value is None:
            if t == 'log_path':
                continue
            raise ValueError(f'missing required parameter: {t}')
        if not os.path.exists(os.path.dirname(t_value)):
            raise ValueError(f'{t}: [{t_value}] is not valid')


    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.devices)
    if options.log_path is not None:
        import logging,time
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

        fh = logging.FileHandler(options.log_path, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        try:
            IMSE_img2img(options.source_path,options.target_path,options.output_path,options.use_roi,options.weight_path,options.log_path)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception:
            logger.error('Failed to open file', exc_info=True)

    else:
        IMSE_img2img(options.source_path,options.target_path,options.output_path,options.use_roi,options.weight_path,options.log_path)