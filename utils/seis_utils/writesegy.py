import os

import segyio
import numpy as np

def writesegy(src_data_dir, dst_data_dir, src_file, dst_file, data, sampling_interval):
    src_filename = os.path.join(src_data_dir, src_file)
    ###重新构建header的text########################
    ##读取原SEGY数据
    f = segyio.open(src_filename, ignore_geometry=True)
    aaa = str(f.text[0])
    ##将地震头信息保存为文件
    fh = open('sourceHeader.txt', 'w', encoding='utf-8')
    fh.write(aaa)
    fh.close()
    ###手动修改地震头文本信息，然后读取地震头信息为变量
    fd = open('sourceHeader.txt')
    content = fd.read()

    ###构建新的SEGY数据########################
    '''
    tracecount=500
    samples=4ms
    TRACE_SAMPLE_COUNT: 500
    '''
    with segyio.open(src_filename, ignore_geometry=True) as src:
        spec = segyio.spec()
        # filename = 'E:\VIRI\mycode\output\mat20220318\\clean.sgy'
        dst_filename=os.path.join(dst_data_dir, dst_file)
        spec.sorting = 2  # 1: TraceSortingFormat.CROSSLINE_SORTING,2: TraceSortingFormat.INLINE_SORTING
        spec.format = 5  # 1 = IBM float, 5 = IEEE float
        spec.samples = np.arange(0, data.shape[0]*sampling_interval, sampling_interval)  # 纵向采样点 4*500=2000
        spec.tracecount = data.shape[1]  # 道数

        with segyio.create(dst_filename, spec) as dst:
            # dst.trace = np.asarray([np.copy(x) for x in src.trace[:]])[:dataMat.shape[0], :dataMat.shape[1]]  # 截取了部分原始数据
            dst.trace =data.T
            #        dst.text[0] = src.text[0]
            dst.text[0] = content  # 将自己构造的地震头信息保存到地震数据
            dst.header = src.header[:data.shape[1]]

            for x in dst.header[:]:  # 将所有地震道头的TRACE_SAMPLE_COUNT改为新的数值
                x[115] = data.shape[0]  ##115 means TRACE_SAMPLE_COUNT
            dst.bin.update()