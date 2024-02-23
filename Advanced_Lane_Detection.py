#!/usr/bin/env python3
# # Advanced Lane Detection

# ## Pipeline of this project:
# 
# 1) Compute the camera caliberation matrix and distortion coefficient from chessboard images. <br/>
# 2) Apply distortion correction to raw images. <br/>
# 3) Use color gradient to create binary threshholded image. <br/>
# 4) Apply perspective transform to binary threshholded image to get top view. <br/>
# 5) Detect pixel lane and fit to find lane boundary. <br/>
# 6) Determine lane curvature and vehicle position wrt centre. <br/>
# 7) Warp the detected boundaries back to original image. <br/>


import numpy as np  
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import time

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# # Camera Caliberation

CAL_IMGS = "/home/hi/문서/Self_Driving_Car-master/CarND-Advanced-Lane-Lines/camera_cal"

calib_files = os.listdir(CAL_IMGS) #주어진 경로에 잇는 파일 및 디렉토리 이름을 리스트 형태로 반환
assert(len(calib_files) > 0)

def draw_imgs(lst, rows, cols=2, figsize=(10, 25), dosave= False, save_dir=""): #save_dir= ""이렇게 비어져 있으면 현재 작업디렉토리가 경로
    assert(len(lst) > 0) #함수 호출시 이 값을 넣어줄거다 
    assert(rows > 0)
    if dosave: #dosave true인 경우 아래 내용 실행, 저장경로 유효성확인
        assert(os.path.exists(save_dir)) #save_dir 경로에  파일이 실제로 존재 하는진 확인후 참인 경우만 계속 코드 진행행
    fig = plt.figure(figsize=figsize)
    fig.tight_layout() #서브플롯 간의 공간을 자동으로 조정하여 겹치지 않게게
    for i in range(1, rows * cols +1): 
        fig.add_subplot(rows, cols, i)
        img = mpimg.imread(CAL_IMGS + "/"+calib_files[i-1])
        plt.imshow(img)
    #plt.show()
    if dosave:#위에서 여러 이미지 파일을 여러 그리드로 합쳐진 하나의 이미지를 저장 
        fig.savefig(save_dir + "/op_" + str(time.time()) + ".png")
    plt.show()

def create_dir(dir_name):#폴더 이름 같은거 없으면 폴더 만드는 함수
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Create directory to save output directory
OUTDIR = "output_images_mj/"   #그리드로 나눈 거 결과물 저장할 폴더 이름
create_dir(OUTDIR)

# Just checking the image,위에서 만든 draw_image 잘되는지 테스트
draw_imgs(calib_files, len(calib_files)//2, dosave=True, save_dir=OUTDIR)



# # Caliberation
# 
# As can be seen in above images there are 9 corners in rows and 6 corners in columns. Lets go ahead and find corners.<br/>
# There are 3 images for which corners = 9 * 6 doesn't work. But 17 images are enough for caliberation

nx = 9
ny = 6 #체스보드의 코너 수입니다. 각각 행과 열에 대한 코너 수를 나타냅니다.

objp = np.zeros((ny * nx, 3), np.float32) #실세계에서 체스보드 코너의 3D 좌표를 저장합니다. 각 코너는 (x, y, z) 형태의 좌표를 가지며, z는 항상 0입니다 54행 3열
objp[:,:2] = np.mgrid[:nx, :ny].T.reshape(-1, 2) #(nx, ny ,2) 형태 배열을 (nx*ny,2) 형태로 변환 즉 각 행이 하나의 점을 나타내며 두열은 각 각 x와 y좌표를 의미

objpoints = [] # 3d points in real world space, 모든 이미지에 대해 감지된 코너들의 3D 위치를 저장하는 리스트
imgpoints = [] # 2d points in image plane.,이미지 평면에서 감지된 코너들의 2D 위치를 저장하는 리스트

failed =[] #코너 감지에 실패한 이미지 파일 이름을 저장하는 리스트

for idx, name in enumerate(calib_files): #enumerate  인덱스와 함게 해당항목을 반환하는 함수, 튜플로 반환
    img = cv2.imread(CAL_IMGS + "/"+ name)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    #    cv2.TERM_CRITERIA_EPS: 이 조건은 알고리즘의 원하는 정확도(epsilon)에 도달했을 때 알고리즘을 종료하도록 합니다. 즉, 알고리즘이 설정된 epsilon 값보다 작은 변화가 발생하면 종료됩니다.
    # v2.TERM_CRITERIA_MAX_ITER: 이 조건은 알고리즘이 설정된 최대 반복 횟수에 도달했을 때 종료하도록 합니다. 알고리즘이 설정된 반복 횟수를 초과하면, 그 시점에서 종료됩니다.
    # 이 두 조건은 + 연산자를 사용하여 결합될 수 있으며, 이는 알고리즘이 두 조건 중 하나라도 만족할 때 종료되도록 설정됨을 의미합니다.

    '''튜플의 이러한 구조는 OpenCV의 특정 함수들이 종료 조건을 해석하는 방식과 직접적으로 연결되어 있습니다. 
    예를 들어, cv2.findChessboardCorners, cv2.cornerSubPix, cv2.calibrateCamera와 같은 함수들은 이 튜플을 인자로 받아, 두 번째 값으로 최대 반복 횟수를, 세 번째 값으로 원하는 정확도를 해석합니다.
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #(종료 조건 유형, 최대 반복 횟수, 원하는 정확도)의 형식을 따라야 합니다.
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None) #이 함수는 성공여부 불리언값(ret)값, 찾아낸 코너 위치정보 순서의 튜플로 반환하기에 순서 바꾸면 안된다.
    
    if ret == True: #코너 검출에 성공하면
        objpoints.append(objp) #빈 리스트인 objpoints뒤에 위에서 만든 격자(gird)들의 좌표 값인 objp 값을 추가
        
        #cv2.findChessboardCorners와 같은 함수로 초기 코너 위치를 대략적으로 찾은 후, 그 결과를 더욱 정밀하게 조정하는 데 사용
        #gray 이미지에서 이미 검출된 corners의 위치를 (11, 11) 크기의 윈도우를 사용하여 미세 조정합니다. (-1, -1)은 중심점 주변의 노이즈를 방지하기 위해 사용되며, 
        #criteria는 미세 조정 과정의 종료 조건을 정의합니다. 
        #이 과정을 통해, 초기에 검출된 코너의 위치를 보다 정확한 서브픽셀 단위의 위치로 조정할 수 있습니다
        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        
        imgpoints.append(corners) #코너좌표 보정된 좌표 값을 빈 2d 좌표로 활용될 imgpoints에 추가한다. 아때 빈 리스트 이므로 그냥 넣어진다.
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)  #img에 있는 원본이미지를 배경으로  코너좌표보정된 값인 corners에 그림 그린다 
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8)) #1행2열의 서브플롯에 대한 축 객체 ax1, ax2로 지정
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(cv2.imread(CAL_IMGS + "/"+ name), cv2.COLOR_BGR2RGB)) #현재 img는 코너에 표시된 그림으로 바뀌었으니 원본파일을 다시 불러와서 오리지날이미지를 그래프에 나타낸다
        ax1.set_title("Original:: " + name, fontsize=18)
        ax2.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))# 위에 cv2.drawChessboardCorners를 통해서 변환된img를 그래프에 나타낸다.
        #matplotlib는 RGB포멧을 쓰는데 opencv는 BGR 포멧을 사용 하기에,
        ax2.set_title("Corners:: "+ name, fontsize=18)
        f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
        
    else:
        failed.append(name)
        
print("Failed for images: [")
print(failed)
print("]")

# # Distortion correction
# 
# Using object and image points calculated in step 1 to caliberate the camera and compute the camera matrix and distortion coefficients.<br/>
# Then use these camera matrix and distortion cofficients to undistort images

def undistort(img_name, objpoints, imgpoints):
    img = cv2.imread(img_name)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    #카메라 켈리브레이션 수행, 실제 objpoints점(실제 세계의 점)과 imgpoints(이미지상에 매핑된점) 사용하여 내부매개변수 mtx, 왜곡계수 dist 계산
    undist= cv2.undistort(img, mtx, dist, None, mtx)
    return undist

#ndistort 함수는 이미지 파일 이름을 입력으로 받아 파일을 먼저 읽은 다음 처리하는 반면, 
#undistort_no_read 함수는 이미 불러온 이미지 데이터를 직접 입력으로 사용합니다.
def undistort_no_read(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist= cv2.undistort(img, mtx, dist, None, mtx)
    return undist


 #10번이미지 예시 출력, 오리지날 이미지와 왜곡수정한 이미지 나란히 두고 비교 하는 블록
undist = undistort(CAL_IMGS+"/calibration10.jpg", objpoints, imgpoints)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(cv2.imread(CAL_IMGS+"/calibration10.jpg"), cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: calibration10.jpg" , fontsize=18)
ax2.imshow(cv2.cvtColor(undist,cv2.COLOR_BGR2RGB))
ax2.set_title("Undistorted:: calibration10.jpg", fontsize=18)
f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")


#jpg 파일의 경로명을 리스트 형태로 images 변수에 저장합니다.
#image 변수는 리스트 images의 다음 요소(이미지 파일 경로)를 차례대로 가리킵니다. 따라서 루프의 본문에서는 image를 사용하여 각 이미지 파일에 대한 작업을 수행할 수 있습니다.
images = glob.glob('test_images/test*.jpg') #glob.glob('test_images/test*.jpg')를 사용하여 특정 패턴("test*.jpg")에 일치하는 모든 이미지 파일의 경로를 가져옵니다.
for image in images:
    undist = undistort(image, objpoints, imgpoints)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()

    # 전체 경로에서 파일 이름만 추출
    """ filename = os.path.basename(image)
    ax1.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + filename , fontsize=18)
    ax2.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    ax2.set_title("Undistorted:: "+ filename, fontsize=18)
    f.savefig(OUTDIR1 + "/op_" + str(time.time()) + ".png")
 """
    ax1.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    ax2.set_title("Undistorted:: "+ image, fontsize=18)
    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")

# # Gradient and color transform
# 
# 
# We'll use sobel filter in both x and y direction to get gradient change in both axes to generate binary threshhold image. <br/>
# We'll also use color space HLS to get color transformed binary threshold image. <br/>
# We'll combine both these outputs to get final binary threshold image.<br/>


#sobel filter란 주파수의 방향성에 따라 필터링을 해주는 필터이다. 방향에 따라 얼마나 변하는지(x,y 방향 미분) 경계선 찾아준다
#x방향 조건을 넣을땐 x방향의 경계 변화량에 따른 검출 행렬 생성하고 y방향 조건을 넣을때도 같이 , 즉 하나에 한번씩 하고
#  def mag_thresh에서 x y 성분 합쳐서 검출 행렬 생성
def abs_thresh(img, sobel_kernel=3, mag_thresh=(0,255), return_grad= False, direction ='x'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB로 변환해놓고 표준화
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad = None
    scaled_sobel = None
    
    # Sobel x, .lower()는 문자열내의 모든 대문자를 소문자로 변환하는 함수, 즉 대문자 소문자 아무거나 입력해도 소문자로 처리 하도록
    if direction.lower() == 'x':
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x       
    # Sobel y
    else:
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
        
    if return_grad == True:
        return grad
    #미분결과가 양수(경계가 밝아지는 방향)인지 음수(어두워지는 방향)인지 보다 경계가 존재하는지 안하는지 미분강도만을 강조할 수 있으므로 절대값취한다.    
    abs_sobel = np.absolute(grad) # Absolute x derivative to accentuate lines away from horizontal,
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) # map함수처럼 0에서 255로 범위를 치환해주는 것

    grad_binary = np.zeros_like(scaled_sobel) # scaled_sobel 이미지와 동일한 형태와 크기를 가지면서, 모든 값을 0으로 채운 새로운 배열을 생성, 검은판 그리고 
    grad_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1 #설정한 임계값이상인 픽셀에만 표시하여 경계선 명확하게 추출, 함수 선언시  mag_thresh 튜플 불러오는것 0과 255
    return grad_binary

img = undistort(images[0], objpoints, imgpoints) #위 glob을 통해서 images를 받아왔다 리스트 형태로 이건 BGR

#X방향 변화량 검출
combined_binary = abs_thresh(img, sobel_kernel=3, mag_thresh=(30, 100), direction='x') 
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)

img = undistort(images[0], objpoints, imgpoints)

#Y방향 변화량 검출
combined_binary = abs_thresh(img, sobel_kernel=3, mag_thresh=(30, 120), direction='y')
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)): #x,y 방향 변화량 통합하여 mag_thresh 범위에 해당하는지에 따라 검출출   
    xgrad =  abs_thresh(img, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh, return_grad=True)
    ygrad =  abs_thresh(img, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh, return_grad=True, direction='y')
    
    magnitude = np.sqrt(np.square(xgrad)+np.square(ygrad)) #제곱합에 제곱근하여 x y 방향 경사값  합의 크기 계산
    abs_magnitude = np.absolute(magnitude)#불필요하다고 생각 위에서 이미 크기만을 나타내고 있기에
    scaled_magnitude = np.uint8(255*abs_magnitude/np.max(abs_magnitude))
    mag_binary = np.zeros_like(scaled_magnitude)
    mag_binary[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude < mag_thresh[1])] = 1
    
    return mag_binary

# X ,Y 두 방향 mag_threshold로 검출
img = undistort(images[0], objpoints, imgpoints)
    
combined_binary = mag_threshold(img, mag_thresh=(30, 100))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)): #차선기울기 , 방향성, 아크탄젠트로 게산
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    xgrad =  cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    ygrad =  cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    xabs = np.absolute(xgrad)
    yabs = np.absolute(ygrad)
    
    grad_dir = np.arctan2(yabs, xabs)
    
    binary_output = np.zeros_like(grad_dir).astype(np.uint8)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir < thresh[1])] = 1
    return binary_output
#매개변수로 정한 채널 한개에 대해서만 범위에 따른 검출
def get_rgb_thresh_img(img, channel='R', thresh=(0, 255)):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if channel == 'R':
        bin_img = img1[:, :, 0]
    if channel == 'G' :른
        bin_img = img1[:, :, 1]
    if channel == 'B' :
        bin_img = img1[:, :, 2]
        
    binary_img = np.zeros_like(bin_img).astype(np.uint8) 
    binary_img[(bin_img >= thresh[0]) & (bin_img < thresh[1])] = 1
    
    return binary_img

img = undistort(images[0], objpoints, imgpoints)
#R채널 검출 테스트
combined_binary = get_rgb_thresh_img(img, thresh=(230, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)

img = undistort(images[0], objpoints, imgpoints)
#G채널 검출 테스트
combined_binary = get_rgb_thresh_img(img, thresh=(200, 255), channel='G')
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)

img = undistort(images[0], objpoints, imgpoints)
#B채널 검출 테스트
combined_binary = get_rgb_thresh_img(img, thresh=(185, 255), channel='B')
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)
#HLS 이미지에서 Lightness 채널을 선택합니다. Lightness 채널은 이미지의 밝기 정보를 담고 있으며, 
#1은 HLS 색상 공간에서 두 번째 채널을 의미합니다
def get_hls_lthresh_img(img, thresh=(0, 255)):
    hls_img= cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = hls_img[:, :, 1]

    binary_output = np.zeros_like(L).astype(np.uint8)    
    binary_output[(L >= thresh[0]) & (L < thresh[1])] = 1
    
    return binary_output

img = undistort(images[0], objpoints, imgpoints)
    
combined_binary = get_hls_lthresh_img(img, thresh=(201, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)
#환된 HLS 이미지에서 Saturation 채널을 선택합니다. Saturation 채널은 이미지의 채도 정보를 담고 있으며,
# 여기서 2는 HLS 색상 공간에서 세 번째 채널을 의미합니다.
def get_hls_sthresh_img(img, thresh=(0, 255)):
    hls_img= cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls_img[:, :, 2]

    binary_output = np.zeros_like(S).astype(np.uint8)    
    binary_output[(S >= thresh[0]) & (S < thresh[1])] = 1
    
    return binary_output

img = undistort(images[0], objpoints, imgpoints)
    
combined_binary = get_hls_sthresh_img(img, thresh=(150, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)




#LAB 색상 공간은 L 채널(Lightness, 밝기), 
#A 채널(녹색에서 마젠타 사이 색상), 
#B 채널(파란색에서 노란색 사이 색상)로 구성됩니다
def get_lab_athresh_img(img, thresh=(0,255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = lab_img[:, :, 1]
    #A 채널을 선택합니다. A 채널은 이미지의 색상 정보 중 녹색에서 마젠타 색상 범위를 나타내며, 색상의 차이를 감지하는 데 유용합니다.
    bin_op = np.zeros_like(A).astype(np.uint8)
    bin_op[(A >= thresh[0]) & (A < thresh[1])] = 1
    
    return bin_op

def get_lab_bthresh_img(img, thresh=(0,255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    B = lab_img[:, :, 2]
    
    bin_op = np.zeros_like(B).astype(np.uint8)
    bin_op[(B >= thresh[0]) & (B < thresh[1])] = 1
    
    return bin_op

img = undistort(images[0], objpoints, imgpoints)
    
combined_binary = get_lab_bthresh_img(img, thresh=(147, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)
 #hsv 색공간으로 변경
 #소벨필터를 기반한 이진화 거친 레이어(sbinary)를 생성하고 
#앞으로 여러 알고리즘을 통해 이진화의 통합본 레이어인 combined 레이어도 만들고 여기에 우선 소벨필터결과만 반영해둔다.
def get_bin_img(img, kernel_size=3, sobel_dirn='X', sobel_thresh=(0,255), r_thresh=(0, 255), 
                s_thresh=(0,255), b_thresh=(0, 255), g_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    if sobel_dirn == 'X':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = kernel_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = kernel_size)
        
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    #sbiinary는 소벨필터 결과 하여 이진화여 나타낸 검은색 바탕에 가장자리 검출한 레이어
    #combined에 복사하는 이유는 나중에 다른 과정(색상 기반 이진화)을 거친 이진화 처리 결과들도 combined에 합치고
    #sbinary는 나중에 다시 소벨필터를 통한 이진화만 된결과로 사용할 수 있게 
    combined = np.zeros_like(sbinary)
    combined[(sbinary == 1)] = 1

    # Threshold R color channel
    #함수는 원본 이미지와 R 채널에 적용할 임계값을 매개변수로 받아, 해당 조건을 만족하는 픽셀을 1로, 그렇지 않은 픽셀을 0으로 하는 이진화 이미지를 반환
    r_binary = get_rgb_thresh_img(img, thresh= r_thresh)
    
    # Threshhold G color channel
    g_binary = get_rgb_thresh_img(img, thresh= g_thresh, channel='G')
    
    # Threshhold B in LAB
    #B 채널(파란색) 이진화는 LAB 색공간에서 수행됩니다. LAB 색공간은 다른 색공간과 달리 인간의 시각과 더 유사한 방식으로 색상을 구분합니다. 
    #여기서 B 채널에 대한 임계값을 적용하여 이진화 이미지를 생성
    b_binary = get_lab_bthresh_img(img, thresh=b_thresh)
    
    # Threshold color channel
    #S 채널(채도)에 임계값을 적용합니다. 채도는 색상의 강렬함을 나타내며, 
    #이를 통해 색상이 뚜렷한 차선을 잘 감지할 수 있습니다.
    s_binary = get_hls_sthresh_img(img, thresh=s_thresh)

    # If two of the three are activated, activate in the binary image
    #각 이진화를 거치고 나서 하나라도 1(즉, 조건을 만족하는 픽셀)인 경우, 최종 이진화 이미지(combined_binary)에서 해당 픽셀을 1로 설정합니다.
    combined_binary = np.zeros_like(combined)
    combined_binary[(r_binary == 1) | (combined == 1) | (s_binary == 1)| (b_binary == 1) | (g_binary == 1)] = 1

    return combined_binary

# Testing the threshholding
kernel_size = 5
mag_thresh = (30, 100)
r_thresh = (235, 255)
s_thresh = (165, 255)
b_thresh = (160, 255)
g_thresh = (210, 255)

for image_name in images:
    img = undistort(image_name, objpoints, imgpoints)
    
    combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh, 
                                  s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(combined_binary, cmap='gray')
    ax2.set_title("Threshold Binary:: "+ image, fontsize=18)
    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")

# # Perspective Transform
# 
# Perspective transform maps the points in given image to different perspective. <br/>
# We are here looking for bird's eye view of the road <br/>
# This will be helpful in finding lane curvature. <br/>
# Note that after perspective transform the lanes should apear aproximately parallel <br/>

def transform_image(img, offset=250, src=None, dst=None):    # src: 변환을 정의하는데 필요한 소스, dst: 목적지
    #.shape 속성: cv2.imread 해서 이미지를 읽어 오면 numpy 형태로 저장 되는데  이때 다양한 속성과 메서드가 생성되는데 이중 하나가 .shape 
    #.shpae는 배열의 각 차원의 크기를 나타냅니다. 이미지 데이터에서는 (높이, 너비, 채널 수) 형태의 정보를 제공합니다.
    #예를 들어, (640, 480, 3)은 높이가 640, 너비가 480, 색상 채널이 3(RGB)인 이미지를 의미
    img_size = (img.shape[1], img.shape[0]) # 너비 높이 순으로 튜플 저장, 나중에 관전 변환 후 이미지 크기를 미리 지정

    
    out_img_orig = np.copy(img) #원본 이미지의 복사본을 생성합니다. 이는 변환 전후 이미지를 비교하기 위해 사용
    #원본 이미지에서 변환할 네 모서리 점
    leftupper  = (585, 460)
    rightupper = (705, 460)
    leftlower  = (210, img.shape[0])
    rightlower = (1080, img.shape[0])
    
    #변환 이미지에서의 위 4개의 점위치
    warped_leftupper = (offset,0)
    warped_rightupper = (offset, img.shape[0]) 
    warped_leftlower = (img.shape[1] - offset, 0) 
    warped_rightlower = (img.shape[1] - offset, img.shape[0])
    
    color_r = [0, 0, 255] #컬러 코드 미리 리스트 형태로 만든 것
    color_g = [0, 255, 0]
    line_width = 5
    
    if src is not None: #src가 있다면 그걸 쓰고 
        src = src
    else: #없다면 아 4좌표를 src 로 쓴다
        src = np.float32([leftupper, leftlower, rightupper, rightlower])
        
    if dst is not None:
        dst = dst
    else:
        dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])
    
    cv2.line(out_img_orig, leftlower, leftupper, color_r, line_width)
    cv2.line(out_img_orig, leftlower, rightlower, color_r , line_width * 2)
    cv2.line(out_img_orig, rightupper, rightlower, color_r, line_width)
    cv2.line(out_img_orig, rightupper, leftupper, color_g, line_width)
    
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)#원본 소스와 bird eye view로 변환한 좌표사이의 변환행렬
    minv = cv2.getPerspectiveTransform(dst, src)# 변환한 좌표에서 다시 원본 소스의 좌표로 돌리는 변환행렬, 역행렬
    
    # Warp the image
    #flags (선택적): 변환을 수행하는 방법을 지정하는 플래그
    #cv2.WARP_FILL_OUTLIERS: 변환 과정에서 발생할 수 있는 외곽선 부분을 채우는 데 사용됩니다.
    #cv2.INTER_CUBIC: 이미지를 확대하거나 축소할 때 사용하는 보간법 중 하나로, 픽셀 값들 사이를 부드럽게 만드는 데 유용합니다. 이는 확대 또는 회전된 이미지의 품질을 향상시키기 위해 사용됩니다.
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC) #변환한 이미지지
    out_warped_img = np.copy(warped)
    #변환한 이미지 위에 선그리기
    cv2.line(out_warped_img, warped_rightupper, warped_leftupper, color_r, line_width)
    cv2.line(out_warped_img, warped_rightupper, warped_rightlower, color_r , line_width * 2)
    cv2.line(out_warped_img, warped_leftlower, warped_rightlower, color_r, line_width)
    cv2.line(out_warped_img, warped_leftlower, warped_leftupper, color_g, line_width)
    
    return warped, M, minv, out_img_orig, out_warped_img

for image in images:
    img = cv2.imread(image)
    warped, M, minv, out_img_orig, out_warped_img = transform_image(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(out_img_orig, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(cv2.cvtColor(out_warped_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Warped:: "+ image, fontsize=18)
    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")

for image in images:
    img = undistort(image, objpoints, imgpoints)
    combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, 
                                  r_thresh=r_thresh, s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(warped, cmap='gray')
    ax2.set_title("Transformed:: "+ image, fontsize=18)
    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")

# # Lane line pixel detection and polynomial fitting
# 
# With binary image where lane lines are clearly visible, now we have to decide lane pixels <br/>
# Also we need to decide pixels from left lane and pixels from right lane. <br/>
# <br/>
# The threshold image pixels are either 0 or 1, so if we take histogram of the image <br/>
# the 2 peaks that we might see in histogram might be good position to start to find lane pixels <br/>
# We can then use sliding window to find further pixels<br/>

def find_lines(warped_img, nwindows=9, margin=80, minpix=40):
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
        
    # Create an output image to draw on and visualize the result
    #dstack은 마지막 차원(여기서는 깊이 차원)방향으로 배열을 쌓아서 하비는 기능 수행,
    #여기서는 gray 인 warped_img 2차원 배열 즉 을 3채널로 배값치하여 r, g, b처럼 3채널에 배치 한다.
    #이 때 gray는 0과 1 사이 값이므로 255 를 곱해줘서 범위를 0~255범위로 바꿔서 컬러로 바꿀준비
    out_img = np.dstack((warped_img, warped_img, warped_img)) * 255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped_img.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window+1)*window_height
        win_y_high = warped_img.shape[0] - window*window_height
        
        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin  
        win_xright_low =  rightx_current - margin 
        win_xright_high = rightx_current + margin  
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) &\
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) &\
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img

def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50, show=True):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img \
        = find_lines(binary_warped, nwindows=nwindows, margin=margin, minpix=minpix)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    if show == True:
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, out_img

# # Skip the sliding windows step once you've found the lines
# 
# Once lines are found, we don't need to do blind search , but we can search around existing line with some margin. As the lanes are not going to shift much between 2 frames of video

def search_around_poly(binary_warped, left_fit, right_fit, ymtr_per_pixel, xmtr_per_pixel, margin=80):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit second order polynomial to for for points on real world   
    left_lane_indices = np.polyfit(lefty*ymtr_per_pixel, leftx*xmtr_per_pixel, 2)
    right_lane_indices = np.polyfit(righty*ymtr_per_pixel, rightx*xmtr_per_pixel, 2)
    
    return left_fit, right_fit, left_lane_indices, right_lane_indices

left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped, nwindows=20)
plt.imshow(out_img)
plt.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")

# # Radius of curvature
# 
# We can fit the circle that can approximately fits the nearby points locally <br/>
# 
# ![alt text](radius_curvature1.png)
# 
# The radius of curvature is radius of the circle that fits the curve<br/>
# The radius of curvature can be found out using equation: <br/>
# <br/>
# ![alt text](eq1.gif)
# <br/>
# For polynomial below are the equation: <br/>
# ![alt text](eq2.gif)

def radius_curvature(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    y_eval = np.max(ploty)
    
    left_fit_cr = np.polyfit(ploty*ymtr_per_pixel, left_fitx*xmtr_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty*ymtr_per_pixel, right_fitx*xmtr_per_pixel, 2)
    
    # find radii of curvature
    left_rad = ((1 + (2*left_fit_cr[0]*y_eval*ymtr_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_rad = ((1 + (2*right_fit_cr[0]*y_eval*ymtr_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return (left_rad, right_rad)

def dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ## Image mid horizontal position 
    #xmax = img.shape[1]*xmtr_per_pixel
    ymax = img.shape[0]*ymtr_per_pixel
    
    center = img.shape[1] / 2
    
    lineLeft = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    lineRight = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
    
    mid = lineLeft + (lineRight - lineLeft)/2
    dist = (mid - center) * xmtr_per_pixel
    if dist >= 0. :
        message = 'Vehicle location: {:.2f} m right'.format(dist)
    else:
        message = 'Vehicle location: {:.2f} m left'.format(abs(dist))
    
    return message

def draw_lines(img, left_fit, right_fit, minv):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    # Find left and right points.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix 
    unwarp_img = cv2.warpPerspective(color_warp, minv, (img.shape[1], img.shape[0]), flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
    return cv2.addWeighted(img, 1, unwarp_img, 0.3, 0)

def show_curvatures(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):
    (left_curvature, right_curvature) = radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    dist_txt = dist_from_center(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    
    out_img = np.copy(img)
    avg_rad = round(np.mean([left_curvature, right_curvature]),0)
    cv2.putText(out_img, 'Average lane curvature: {:.2f} m'.format(avg_rad), 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(out_img, dist_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    return out_img

for image in images:    
    img = undistort(image, objpoints, imgpoints)
    
    combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh, 
                                  s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
    
    xmtr_per_pixel=3.7/800
    ymtr_per_pixel=30/720
    
    left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped, nwindows=12, show=False)
    lane_img = draw_lines(img, left_fit, right_fit, minv)
    out_img = show_curvatures(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Lane:: "+ image, fontsize=18)
    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")

# # Pipeline for video

class Lane():
    def __init__(self, max_counter):
        self.current_fit_left=None
        self.best_fit_left = None
        self.history_left = [np.array([False])] 
        self.current_fit_right=None
        self.best_fit_right = None
        self.history_right = [np.array([False])] 
        self.counter = 0
        self.max_counter = 1
        self.src = None
        self.dst = None
        
    def set_presp_indices(self, src, dest):
        self.src = src
        self.dst = dst
        
    def reset(self):
        self.current_fit_left=None
        self.best_fit_left = None
        self.history_left =[np.array([False])] 
        self.current_fit_right = None
        self.best_fit_right = None
        self.history_right =[np.array([False])] 
        self.counter = 0
        
    def update_fit(self, left_fit, right_fit):
        if self.counter > self.max_counter:
            self.reset()
        else:
            self.current_fit_left = left_fit
            self.current_fit_right = right_fit
            self.history_left.append(left_fit)
            self.history_right.append(right_fit)
            self.history_left = self.history_left[-self.max_counter:] if len(self.history_left) > self.max_counter else self.history_left
            self.history_right = self.history_right[-self.max_counter:] if len(self.history_right) > self.max_counter else self.history_right
            self.best_fit_left = np.mean(self.history_left, axis=0)
            self.best_fit_right = np.mean(self.history_right, axis=0)
        
    def process_image(self, image):
        img = undistort_no_read(image, objpoints, imgpoints)
        
        combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh,
                                      r_thresh=r_thresh, s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    
        if self.src is not None or self.dst is not None:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, src=self.src, dst= self.dst)
        else:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
    
        xmtr_per_pixel=3.7/800
        ymtr_per_pixel=30/720
    
        if self.best_fit_left is None and self.best_fit_right is None:
            left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped, nwindows=15, show=False)
        else:
            left_fit, right_fit, left_lane_indices, right_lane_indices= search_around_poly(warped, self.best_fit_left, self.best_fit_right, xmtr_per_pixel, ymtr_per_pixel)
            
        self.counter += 1
        
        lane_img = draw_lines(img, left_fit, right_fit, unwarp_matrix)
        out_img = show_curvatures(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)
        
        self.update_fit(left_fit, right_fit)
        
        return out_img

clip1 = VideoFileClip("project_video.mp4")
img = clip1.get_frame(0)

leftupper  = (585, 460)
rightupper = (705, 460)
leftlower  = (210, img.shape[0])
rightlower = (1080, img.shape[0])
    
color_r = [255, 0, 0]
color_g = [0, 255, 0]
line_width = 5
    
src = np.float32([leftupper, leftlower, rightupper, rightlower])

cv2.line(img, leftlower, leftupper, color_r, line_width)
cv2.line(img, leftlower, rightlower, color_r , line_width * 2)
cv2.line(img, rightupper, rightlower, color_r, line_width)
cv2.line(img, rightupper, leftupper, color_g, line_width)

plt.imshow(img)

lane1 = Lane(max_counter=5)

leftupper  = (585, 460)
rightupper = (705, 460)
leftlower  = (210, img.shape[0])
rightlower = (1080, img.shape[0])
    
warped_leftupper = (250,0)
warped_rightupper = (250, img.shape[0])
warped_leftlower = (1050, 0)
warped_rightlower = (1050, img.shape[0])

src = np.float32([leftupper, leftlower, rightupper, rightlower])
dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])

lane1.set_presp_indices(src, dst)

output = "test_videos_output/project.mp4"
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(lane1.process_image)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))

# ## 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
# 
# 

# While testing on challenge video and harder challenge video, the problems encountered are mostly due to lighting condition, shadows and road conditions ( edge visible on road other than lane marking). Although HLS space works well for simple video, it activates noisy areas more. ( that's visible in video 2 and 3). I may try LAB color space which separates yellow color better.<br/>
# <br/>
# The averaging of lane works well to smoothen the polynomial output. Harder challenge also poses a problem with very steep curves too. May be we need to fit higher polynomial to these steep curves<br/>
# <br/>
# Also, still these algorithms relie a lot on lane being visible, video being taken from certain angle, light condition and still feels like hand crafted. There might be better way based off RNN(ReNet) or Instance Segmentation (https://arxiv.org/pdf/1802.05591.pdf) or spatial CNN (https://arxiv.org/pdf/1712.06080.pdf) <br/>
# 
# I wonder what the tesla is using  that they displayed in recent autonomy day video.
# <br/>

