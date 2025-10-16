import numpy as np

onnx_path = r'E:\mmdetection320\work_dirs\vitdet_faster-rcnn_vit-b-mae_lsj-100e\iter_427920.onnx'
onnx = open(onnx_path,'rb')
org_bytes = onnx.read()
onnx.close()

fname = onnx_path.replace('.onnx','.chongwang2')

ls = []

for i in range(len(org_bytes)):
    ls.append((org_bytes[i]*2+10))

ls = np.array(ls,dtype=np.int16)
# np.save(fname,ls)  ##一样的
fo = open(fname,'wb')
fo.write(bytearray(ls))
fo.close()

recovery_int = np.fromfile(fname,dtype=np.int16)
recovery_bytes = np.array(np.around((recovery_int-10)/2),dtype=np.int8).tobytes()

with open(onnx_path.replace('.onnx','_jiemi.onnx'),'wb') as f:
    f.write(recovery_bytes)

