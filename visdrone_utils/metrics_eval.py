import motmetrics as mm
import pandas as pd
from pathlib import Path

# 你要比對的序列
seq_name = 'uav0000009_03358_v'

# 修改成你實際的 GT 與 tracking txt 檔案位置
gt_path = f'D:/YOLOv8-DeepSORT-Object-Tracking-main/eval/gt/{seq_name}.txt'
pred_path = f'D:/YOLOv8-DeepSORT-Object-Tracking-main/eval/train_gt/{seq_name}.txt'

# 只讀前 6 欄：frame_id, id, x, y, w, h
def read_mot_txt(path):
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :6]
    df.columns = ['FrameId', 'Id', 'X', 'Y', 'W', 'H']
    return df

# 載入 GT 和預測結果
gt = read_mot_txt(gt_path)
pred = read_mot_txt(pred_path)

# 建立 MOT accumulator
acc = mm.MOTAccumulator(auto_id=True)

# 處理每一幀資料
for frame_id in sorted(gt['FrameId'].unique()):
    gt_frame = gt[gt['FrameId'] == frame_id]
    pred_frame = pred[pred['FrameId'] == frame_id]

    gt_boxes = gt_frame[['X', 'Y', 'W', 'H']].values
    pred_boxes = pred_frame[['X', 'Y', 'W', 'H']].values

    distance_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

    acc.update(
        gt_frame['Id'].values,
        pred_frame['Id'].values,
        distance_matrix
    )

# 計算指標
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'idf1', 'num_switches'], name=seq_name)

# 顯示結果
print(summary)
