# Week 1: Video Clip and Frame Generation
# Keyword: Shot Change Detection

import cv2

# 開啟影片檔案
cap = cv2.VideoCapture('./soccer.mp4')

# 檢查影片是否成功開啟
if not cap.isOpened():
    print("Cannot open video")
    exit()

# 循循環讀取影片幀
while True:
    ret, frame = cap.read()             # 讀取影片的每一幀

    if not ret:
        print("Cannot receive frame")   # 如果沒有幀可讀，退出循環
        break

    cv2.imshow('Frame', frame)          # 如果讀取成功，在視窗顯示該幀的畫面
    
    if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，偵測按下'q'退出循環
        break

# 釋放資源
cap.release()                           # 所有作業都完成後，釋放資源
cv2.destroyAllWindows()                 # 結束所有視窗

# 參考資料: https://steam.oxxostudio.tw/category/python/ai/opencv-read-video.html