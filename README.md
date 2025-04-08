# YOLO-v4
- 該程式碼主要實現了基於 YOLOv4 的物件檢測與座標提取，並將檢測結果進行後處理與儲存
## 程式碼架構
1. ImportYoloGraph 類別：
  - 初始化 (__init__)：載入 YOLOv4 的權重檔案（.weights）、配置檔案（.cfg）以及標籤檔案（.names）
  - get_output_layers 方法：獲取 YOLO 模型的輸出層名稱
  - detectGetCoordinates 方法：對輸入影像進行物件檢測，返回檢測框的座標、信心分數、中心點以及類別
2. 主程式邏輯 (__main__)：
  - 初始化 YOLO 模型
  - 定義輸入影像資料夾與輸出結果資料夾
  - 遍歷多個月份的影像資料，對每張影像進行物件檢測
  - 使用非極大值抑制（NMS）過濾檢測框，並將結果儲存為影像與裁剪的物件
## 關鍵技術
1. YOLOv4 模型：
  - 使用 OpenCV 的 cv2.dnn 模組載入 YOLOv4 模型進行物件檢測
  - 配置檔案（yolov4.cfg）定義了 YOLOv4 的網路結構，包含卷積層、池化層、激活函數等
  - 權重檔案（yolov4_best.weights）包含了訓練好的模型參數
2. 非極大值抑制（NMS）：
  - 使用 OpenCV 的 cv2.dnn.NMSBoxes 方法，過濾掉重疊的檢測框，保留高信心分數的框
3. 影像處理：
  - 使用 OpenCV 進行影像讀取、縮放（cv2.resize）、框選（cv2.rectangle）以及裁剪
