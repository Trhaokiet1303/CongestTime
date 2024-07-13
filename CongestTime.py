import numpy as np
import cv2
from math import sqrt, pow
import time

#########################################################################################
# Link video : https://drive.google.com/drive/folders/1pDslmmSyhkXa4-HfEAzwkA_3PPRIDAlM #
#########################################################################################

# Tính khoảng cách giữa hai tọa độ cũ và mới bằng Euclidean
def dist(x1, y1, x2, y2):
    return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))

# Tính tốc độ và thời gian
def get_speed(ox, oy, nx, ny, lane=0):
    if oy > measurement_region[lane][0][1] and oy < measurement_region[lane][1][1] and ox < 1500 and ox > 1000:
        pixeldistance = dist(ox, oy, nx, ny)
        pixelspersecond = pixeldistance * fps
        meterspersecond = pixelspersecond / pixelspermeter
        # return kmph
        speed = round(meterspersecond * 3.6, 1)

        tiMe = 0

        if speed > 1:
            tiMe = round((500 / pixelspermeter / meterspersecond), 1)

        return speed, tiMe
    return 0, 0

# Tính chỉ số Travel Time Index (TTI), kẹt xe
def get_congestion(actual_time, minimum_time):
    travel_time_index = actual_time / minimum_time
    congestion_index = (actual_time - minimum_time) / minimum_time * 100
    return travel_time_index, congestion_index

# Đặt lại tốc độ, thời gian, chỉ số kẹt xe của từng làn
def resetlanespeed():
    for i in range(len(lane_speed)):
        lane_speed[i] = 0
        travel_time[i] = 0
        congestion_time[i] = 0
        travel_time_index[i] = 0

###################################################################


video_file = "FullLane.mp4"

# Vẽ vùng đo lường dài 500 pixel
measurement_region = [
    np.array([[1000,  30], [1000, 165], [1500, 165], [1500,  30]]),
    np.array([[1000, 185], [1000, 240], [1500, 240], [1500, 185]]),
    np.array([[1000, 245], [1000, 320], [1500, 320], [1500, 245]]),
    np.array([[1000, 324], [1000, 390], [1500, 390], [1500, 324]]),
    np.array([[1000, 395], [1000, 460], [1500, 460], [1500, 395]]),
    np.array([[1000, 485], [1000, 560], [1500, 560], [1500, 485]]),
    np.array([[1000, 565], [1000, 630], [1500, 630], [1500, 565]]),
    np.array([[1000, 636], [1000, 760], [1500, 760], [1500, 636]]),
    np.array([[1000, 785], [1000, 915], [1500, 915], [1500, 785]]),
    np.array([[1000, 930], [1000, 1050], [1500, 1050], [1500, 930]])
    ]

lane_speed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
travel_time = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#Tốc độ bình thường trên đường 80 km/h
minimum_time = 0.92676

congestion_time = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
travel_time_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

TTI_full = 0
traffic_full = ""


cap = cv2.VideoCapture(video_file)

# Quy đổi giữa pixel và met
lanemarkingpixel = 768-694
pixelspermeter = lanemarkingpixel / 3.048
fps = round(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS: {fps}")

###################################################################

# Thuật toán phát hiện góc Shitomasi

# Khai báo các thông số
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.01,
                       minDistance = 20,
                       blockSize = 3 )

# Các thông số cho dòng quang lucas kanade
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS |
                            cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Tạo mảng màu ngẫu nhiên
color = np.random.randint(0, 255, (100, 3))

# Lấy khung hình đầu tiên và tìm các góc
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Tạo hình ảnh mặt nạ cho các chỉ số
mask = np.zeros_like(old_frame)

feature_refresh_counter = 0
start_time = time.time()

###################################################################

while(True):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_copy = frame.copy()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    cl1 = clahe.apply(v)
    hsv[:,:,2] = cl1
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    clahe_img = frame.copy()

    # Tạo màu cho vùng đo lường
    cv2.fillPoly(mask, [measurement_region[0]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[1]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[2]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[3]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[4]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[5]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[6]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[7]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[8]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[9]], (0,0,50))

    # Thời gian phát của video khi bắt đầu phát
    elapsed_time = time.time() - start_time

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Tính toán lại điểm đặc trưng tốt
    if feature_refresh_counter % 20 == 0:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


    # Tính dòng quang bằng thuật toán Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Chọn điểm tốt
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    # Nhận dạng xe
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        for i in range(len(lane_speed)):
            speed, time_to_travel = get_speed(a, b, c, d, lane=i)
            travel, congest_time = get_congestion(time_to_travel, minimum_time)

            if speed > 1:
                lane_speed[i] = speed
                travel_time[i] = time_to_travel
                congestion_time[i] = congest_time
                travel_time_index[i] = travel

                TTI_full = sum(travel_time_index) / len(travel_time_index)
                if TTI_full >= 1 and TTI_full <= 1.7:
                    traffic_full = "Smooth Traffic"
                elif TTI_full > 1.7:
                    traffic_full = "Traffic Jam"

                
        # Vẽ tọa độ cũ và mới lên
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0,255,0), -1)
        frame = cv2.circle(frame, (int(c), int(d)), 5, (0,0,255), -1)

    img = cv2.add(frame, mask)

    ###################################################################

    # Ghi các thông tin lên video

    # Thời gian đo lường và FPS 
    cv2.putText(img, 'Time: {:.1f} s | FPS: {:.0f} | Traffic Assessment: {}'.format(elapsed_time, fps, traffic_full), (20,42), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)

    # Các thông tin của từng làn: tốc độ, thời gian qua vùng đo lường
    cv2.putText(img, 'LANE1: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[0], travel_time[0]), (20,72), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'LANE2: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[1], travel_time[1]), (20,200), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'LANE3: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[2], travel_time[2]), (20,265), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'LANE4: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[3], travel_time[3]), (20,340), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'LANE5: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[4], travel_time[4]), (20,420), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'LANE6: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[5], travel_time[5]), (20,500), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'LANE7: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[6], travel_time[6]), (20,580), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'LANE8: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[7], travel_time[7]), (20,690), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'LANE9: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[8], travel_time[8]), (20,840), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'LANE9: Speed: {:.1f} km/h | Travel time: {:.1f} s'.format(lane_speed[9], travel_time[9]), (20,985), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)

    # Thông tin của từng làn: Chỉ số kẹt xe, chỉ số Travel time index
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[0], travel_time_index[0]), (107,92), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[1], travel_time_index[1]), (107,220), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[2], travel_time_index[2]), (107,290), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[3], travel_time_index[3]), (107,360), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[4], travel_time_index[4]), (107,440), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[5], travel_time_index[5]), (107,520), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[6], travel_time_index[6]), (107,600), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[7], travel_time_index[7]), (107,710), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[8], travel_time_index[8]), (107,860), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, 'Congestion time: {:.2f} % | TTI: {:.2f}'.format(congestion_time[9], travel_time_index[9]), (107,1005), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    ###################################################################


    cv2.imshow('frame', img)
    
    
    key = cv2.waitKey(fps)
    if key == ord("q"):
        break
    
    resetlanespeed()
    
    # cập nhật khung trước với các điểm trước đó
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    feature_refresh_counter += 1
    
cv2.destroyAllWindows()

# Link source: https://github.com/Shubhamtakbhate1998/Realtime-highway-traffic-speed-measurement-
# Link video: https://mixkit.co/free-stock-video/highway-traffic-seen-through-drone-611/