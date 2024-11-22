from PIL import Image
import cv2

img = cv2.imread("/home/eric/data/military_data/tanks_20/image/1.png")

#img = Image.open("/home/eric/data/military_data/tanks_20/image/1.png")
# 이미지 창에 표시
cv2.imshow("Loaded Image", img)

# 키 입력 대기
cv2.waitKey(0)  # 아무 키나 누르면 창이 닫힘
cv2.destroyAllWindows()