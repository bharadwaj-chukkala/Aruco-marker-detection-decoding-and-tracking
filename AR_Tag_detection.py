import numpy as np
import cv2    
import scipy.fftpack
import matplotlib.pyplot as plt


##### Defining a Gaussian Mask to Blur the image######
#
#   A Gaussian mask upon multiplying with the fft of grey image will give a blurred output
#
def Gaussian_Blur_Mask(grey_image, sigma_x, sigma_y):
    cols,rows = grey_image.shape
    center_x, center_y = rows / 2, cols / 2
    x = np.linspace(0, rows, rows)
    y = np.linspace(0, cols, cols)
    X, Y = np.meshgrid(x,y)
    mask = np.exp(-(np.square((X - center_x)/sigma_x) + np.square((Y - center_y)/sigma_y)))
    return mask


##### Defining a function to perform fft and multiply gaussian mask ######
#
#   Calculate the FFT for the grey image and mutliply with Gaussian and then perform inverse fft
#
def BlurImage_FFT(grey_image):

    fft_image = scipy.fft.fft2(grey_image, axes = (0,1))
    fft_image_shifted = scipy.fft.fftshift(fft_image)
    
    Gmask = Gaussian_Blur_Mask(grey_image,40,40)
    fft_image_blur = fft_image_shifted * Gmask

    img_shifted_back = scipy.fft.ifftshift(fft_image_blur)
    img_back_blur = scipy.fft.ifft2(img_shifted_back)
    img_back_blur = np.abs(img_back_blur)
    img_blur = np.uint8(img_back_blur)

    return img_blur


##### Defining a Circular Mask to single out the high frequencies (edges) from the image #####
#
#   Using a high band pass filter, we shall reduce the noisy frequencies in the image to detect edges
#
def High_band_pass_filter(image_size, radius, high_band_pass = True):
    rows, cols = image_size
    center_x, center_y = int(rows / 2), int(cols / 2)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center_x) ** 2 + (y - center_y) ** 2 <= np.square(radius)

    if high_band_pass:
        mask = np.ones((rows, cols)) 
        mask[mask_area] = 0
    else:
        mask = np.zeros((rows, cols)) 
        mask[mask_area] = 1

    return mask


###### Defining a function to perform fft on blurred image and multiply it with a circular mask ####
#
#   This Circular mask will have a threshold value, which will negate all the lower frequencies that are unneeded
#
def Edges_using_fft(thresh):

    fft_thresh_img = scipy.fft.fft2(thresh, axes = (0,1))          #Fast Fourier Transform of blurred image
    fft_thresh_img_shifted = scipy.fft.fftshift(fft_thresh_img)

    Cmask = High_band_pass_filter(thresh.shape, 125, True)         
    fft_edge_img = fft_thresh_img_shifted * Cmask                  #Multiplying with the Mask to get higher frequencies
    
    edge_img_back_shifted = scipy.fft.ifftshift(fft_edge_img)
    img_back_edge = scipy.fft.ifft2(edge_img_back_shifted)         #Inverse FFT to get output in original domain (edges)
    img_back_edge = np.abs(img_back_edge)

    return img_back_edge


###### Defining a function to do the image processing that is needed on every frame in the video ######
#
#  This fucntion will perform smoothening and edge detection on every frame by using the fucntions i previously defined
#
def Process_image(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                #Convert to grayscaele
    image_blur = BlurImage_FFT(grey_image)                              #Image blurring
    ret,thresh = cv2.threshold(image_blur, 220 ,255,cv2.THRESH_BINARY)  #Thresholding the Binaryimage
    image_edge = Edges_using_fft(thresh)                                #Detect Edges
    return grey_image,image_blur, thresh,image_edge


##### Defining a function to detect the corners #####
#
#   Using the Harris Corner Detection method and a morphological process called erosion we detect the corners
#   We use a kernel to smoothen the image and we assign a criteria to refine the corners
#   Shi- Tomasi execution, didnt work as expected. Code commented for reference purposes
#
def features(grey_image,image):
    
#### Shi- Tomasi ####

    # kernel = np.ones((5,5),np.uint8)
    # kernel = np.ones((5,5),np.uint8)
    # image_dilated = cv2.dilate(grey_image,kernel,iterations = 1)
    # erosion = cv2.erode(image_dilated,kernel,iterations = 1)
    # corners = cv2.goodFeaturesToTrack(erosion, 10,0.1,100)
    # corners = np.int0(corners)

#### Harris Corner Detection ####
    kernel = np.ones((11,11),np.uint8)                       #Kernel to smoothen the image
    erosion = cv2.erode(grey_image,kernel,iterations = 1)    #Morphology to get specific features
    dst = cv2.cornerHarris(erosion,3,3,0.05)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    #find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    #Criterion to stop at each frame and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(grey_image,np.float32(centroids),(5,5),(-1,-1),criteria)

    if len(corners) > 8:

        x = []
        y = []

        for i in range(0,len(corners)):
            a = corners[i]
            x.append(int(a[0]))
            y.append(int(a[1]))

#### Detecting the sheet corners ####
        X_min = x.index(min(x))
        min_x = x.pop(X_min)
        min_x_y = y.pop(X_min)

        X_max = x.index(max(x))
        max_x = x.pop(X_max)
        max_x_y = y.pop(X_max)

        Y_min = y.index(min(y))
        min_y = y.pop(Y_min)
        min_y_x = x.pop(Y_min)

        Y_max = y.index(max(y))
        max_y = y.pop(Y_max)
        max_y_x = x.pop(Y_max)

        image = cv2.line(image,(min_x,min_x_y),(min_y_x,min_y),(0,120,0),2)
        image = cv2.line(image,(min_x,min_x_y),(max_y_x,max_y),(0,120,0),2)
        image = cv2.line(image,(max_y_x,max_y),(max_x,max_x_y,),(0,120,0),2)
        image = cv2.line(image,(min_y_x,min_y),(max_x,max_x_y),(0,120,0),2)

#### Detecting the Tag corners ####
        X_min = x.index(min(x))
        min_x = x.pop(X_min)
        min_x_y = y.pop(X_min)

        X_max = x.index(max(x))
        max_x = x.pop(X_max)
        max_x_y = y.pop(X_max)

        Y_min = y.index(min(y))
        min_y = y.pop(Y_min)
        min_y_x = x.pop(Y_min)

        Y_max = y.index(max(y))
        max_y = y.pop(Y_max)
        max_y_x = x.pop(Y_max)

        image = cv2.line(image,(min_x,min_x_y),(min_y_x,min_y),(190,0,255),2)
        image = cv2.line(image,(min_x,min_x_y),(max_y_x,max_y),(190,0,255),2)
        image = cv2.line(image,(max_y_x,max_y),(max_x,max_x_y,),(190,0,255),2)
        image = cv2.line(image,(min_y_x,min_y),(max_x,max_x_y),(190,0,255),2)

        corner_points = np.array(([min_y_x,min_y],[min_x,min_x_y],[max_y_x,max_y],[max_x,max_x_y]))

        desirable_coordinates = np.array([ [0, tag_dimension-1], [tag_dimension-1, tag_dimension-1], [tag_dimension-1, 0], [0, 0]])

        return image,corner_points,desirable_coordinates,min_y,max_y,min_x,max_x

    return image,None,None,None,None,None,None


##### Function to extract the inner grid of the AR Tag #####
#
#   Extracting the inner grid of the tag by decomposing the tag into an 8 x 8 grid
#   This grid will give us orientation information and the significance values of innermost bits
#
def Inner_grid(ref_tag_image):
    tag_dimension = 160
    AR_tag_gray = cv2.cvtColor(ref_tag_image, cv2.COLOR_BGR2GRAY)
    AR_tag_Threshold = cv2.threshold(AR_tag_gray, 230 ,255,cv2.THRESH_BINARY)[1]
    Resized_AR_tag = cv2.resize(AR_tag_Threshold, (tag_dimension, tag_dimension))
    grid_size = 8
    block = int(tag_dimension/grid_size)
    grid = np.zeros((8,8))
    x = 0
    y = 0
    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            cell = Resized_AR_tag[y:y+block, x:x+block]
            if cell.mean() > 255//2:
                grid[i][j] = 255
            x = x + block
        x = 0
        y = y + block
    inner_grid = grid[2:6, 2:6]
    return inner_grid


##### Fucntion to decode the AR tag in the frames #####
#
#   Now we will find the orientation and the id of the tag in the frame
#
def Tag_decoder(inner_grid):
    count = 0
    while not inner_grid[3,3] and count<4 :
        inner_grid = np.rot90(inner_grid,1)
        count+=1

    
    info_grid = inner_grid[1:3,1:3]
    info_grid_array = np.array((info_grid[0,0],info_grid[0,1],info_grid[1,1],info_grid[1,0]))
    tag_id = 0
    tag_id_bin = []
    for i in range(0,4):
        if(info_grid_array[i]) :
            tag_id = tag_id + 2**(i)
            tag_id_bin.append(1)
        else:
            tag_id_bin.append(0)

    return tag_id, tag_id_bin,count


##### Function to compute homography matrix to superimpose an image on tag #####
#
#   A transformation fucntion that will project (x,y) of an image as (u,v) in an other perspective
#
def Homography_matrix(corners1, corners2):

    if (len(corners1) < 4) or (len(corners2) < 4):
        print("Homography needs atleast 4 points to work")
        return 0

    x = corners1[:, 0]
    y = corners1[:, 1]
    xp = corners2[:, 0]
    yp = corners2[:,1]

    nrows = 8
    ncols = 9
    
    A = []
    for i in range(int(nrows/2)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)

    A = np.array(A)
    U, E, VT = np.linalg.svd(A)
    V = VT.transpose()
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]
    
    return H

##### Function for Bilinear interpolation of the coordinates of corners #####
#
#  This function will perform a two dimensional interpolation on the coordiniates to stabilize
#  the corner detection in everyframe

def Bilinear_Interpolation(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Fa = im[ y0, x0 ]
    Fb = im[ y1, x0 ]
    Fc = im[ y0, x1 ]
    Fd = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Fa + wb*Fb + wc*Fc + wd*Fd

#### Camera Intrinsic parameters ####
K =np.array([[1346.1005953,0,932.163397529403],
   [ 0, 1355.93313621175,654.898679624155],
   [ 0, 0,1]])

##### Projection matrix function #####
#
#   This function calculates lambda using Homography and Camera intrinsic parameters.
#   then it projects the image onto the tag 
def Projection_Matrix(h, K):  
    #taking column vectors h1,h2 and h3
    h1 = h[:,0]          
    h2 = h[:,1]
    h3 = h[:,2]

    #calculating lamda using H and K
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = lamda * np.matmul(np.linalg.inv(K),h)

    #check if determinant > . It has positive determinant when object is in front of camera
    det = np.linalg.det(b_t)

    if det > 0:
        b = b_t
    else:                    #If it is not positive, manually make it positive
        b = -1 * b_t  
        
    row1 = b[:, 0]
    row2 = b[:, 1]                      
    row3 = np.cross(row1, row2)
    
    t = b[:, 2]
    Rt = np.column_stack((row1, row2, row3, t))

    P = np.matmul(K,Rt)  
    return(P,Rt,t)

##### Inverse Warping fucntion #####
#
#   Inverse warping will give us the upright tag from a perspective position
#
def Inverse_warping(H,img,maxHeight,maxWidth):
    H_inv=np.linalg.inv(H)
    warped_image_image=np.zeros((maxHeight,maxWidth,3),np.uint8)
    for i in range(maxHeight):
        for j in range(maxWidth):
            f = [i,j,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            xb = np.clip(x/z,0,1919)
            yb = np.clip(y/z,0,1079)
            # x, y, z = np.matmul(H,f)
            warped_image_image[i][j] = img[int(yb)][int(xb)]
    return(warped_image_image)

testudo_img = cv2.imread("testudo.png")
testudo_img = cv2.resize(testudo_img, (160,160))


##### Function to rotate points according to the orientation of the tag #####
def Point_rotater(points):
    point_list = list(points.copy())
    top = point_list.pop(-1)
    point_list.insert(0, top)
    return np.array(point_list)


##### Function to scale the testudo image #####
#
#   Scaling the testudo image is necessary to superimpose it on the AR Tag
# 
def Scale_func(frame,scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)



##### Function for Superimposing Testudo on the AR Tag ##### 
def Target_Superimpose(image,testudo_img,corner_points,desirable_coordinates,min_y,max_y,min_x,max_x):
    
    rows,cols,ch = image.shape
    H = Homography_matrix( np.float32(corner_points),np.float32(desirable_coordinates))
    
    h_inv = np.linalg.inv(H)

    for a in range(0,tag.shape[1]):
        for b in range(0,tag.shape[0]):
            x, y, z = np.matmul(h_inv,[a,b,1])
            xb = np.clip(x/z,0,1919)
            yb = np.clip(y/z,0,1079)
            image[int(yb)][int(xb)] = Bilinear_Interpolation(testudo_img, b, a)
            
    return image

cap = cv2.VideoCapture("1tagvideo.mp4")
tag_dimension = 160


##### A while statement that executes all our fucntions #####
#
# Detection, Superimposition and Tracking
#
while(True):
    ret, frame = cap.read()
    if not ret:
        print("The Video has ended...")
        break
    image = frame.copy()

#Edge Detection
    grey_image,image_blur, thresh,image_edge = Process_image(image)
    image_edge = np.uint8(image_edge)

#Corner Detection
    frame,corner_points,desirable_coordinates,min_y,max_y,min_x,max_x = features(grey_image,image)
 
    if corner_points is not None:

        H = Homography_matrix( np.float32(corner_points),np.float32(desirable_coordinates))
        tag = Inverse_warping( H, image,tag_dimension, tag_dimension)
        tag = cv2.cvtColor(np.uint8(tag), cv2.COLOR_BGR2GRAY)
        ret,tag = cv2.threshold(np.uint8(tag), 230 ,255,cv2.THRESH_BINARY)
        tag = cv2.cvtColor(tag,cv2.COLOR_GRAY2RGB)
        inner_grid = Inner_grid(tag)
        tag_id, tag_id_bin,count = Tag_decoder(inner_grid)

        for i in range(count):
            desirable_coordinates = Point_rotater(desirable_coordinates)
            
#Superimposing and tracking the Testudo
        image = Target_Superimpose(image,testudo_img,corner_points,desirable_coordinates,min_y,max_y,min_x,max_x)

        H = Homography_matrix( np.float32(desirable_coordinates),np.float32(corner_points))

        P,Rt,t = Projection_Matrix(H,K)
        x1,y1,z1 = np.matmul(P,[0,0,0,1])
        x2,y2,z2 = np.matmul(P,[0,159,0,1])
        x3,y3,z3 = np.matmul(P,[159,0,0,1])
        x4,y4,z4 = np.matmul(P,[159,159,0,1])
        x5,y5,z5 = np.matmul(P,[0,0,-159,1])
        x6,y6,z6 = np.matmul(P,[0,159,-159,1])
        x7,y7,z7 = np.matmul(P,[159,0,-159,1])
        x8,y8,z8 = np.matmul(P,[159,159,-159,1])

#Superimposing and tracking a Cube on the AR Tag      
 
        cv2.line(image,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (255, 0,0), 2)
        cv2.line(image,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (0,126,0), 2)
        cv2.line(image,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (255,0,0), 2)
        cv2.line(image,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (0,0,185), 2)

        cv2.line(image,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (112,255,0), 2)
        cv2.line(image,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,136), 2)
        cv2.line(image,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,192), 2)
        cv2.line(image,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (69,254,0), 2)

        cv2.line(image,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (255,72,0), 2)
        cv2.line(image,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (255,0,184), 2)
        cv2.line(image,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (255,0,123), 2)
        cv2.line(image,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (255,45,0), 2)
    

    try:
        cv2.imshow('frame',image)
        result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
