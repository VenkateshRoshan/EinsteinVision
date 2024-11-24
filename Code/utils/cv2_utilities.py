import cv2 
import numpy as np


# Traditional Edge Detector 
def edge_detector(img):
    # Apply GaussianBlur to the image
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate and erode the image
    kernel = np.ones((3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    return edges

def dilate_image_with_mask(img, mask):
    # Dilate the image using the mask
    print("Mask shape", mask.shape)
    print("Image shape", img.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(img, kernel, iterations=1)
    mask = mask.astype('uint8')
    # Apply the mask on the dilated image
    masked = cv2.bitwise_and(dilated, dilated, mask=mask)
    return masked

def find_contours(img):
    # Find the contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def single_contour_processing(contour):
    contour_edges = cv2.approxPolyDP(contour, 0.03*cv2.arcLength(contour, True), True)
    
    hull = cv2.convexHull(contour)
    hull_edges = cv2.approxPolyDP(hull, 0.03*cv2.arcLength(hull, True), True)
    
    return contour_edges, hull_edges, hull

# Function to filter contours and hulls based on area threshold
def filter_contours_hulls(contours, hulls, threshold=20):
    filtered_contours = []
    filtered_hulls = []
    for contour, hull in zip(contours, hulls):
        if cv2.contourArea(contour) < threshold:
            filtered_contours.append(contour)
            filtered_hulls.append(hull)
    return filtered_contours, filtered_hulls

def multiple_contours_processing(contours, threshold=2):
    contour_edges_list = []
    hull_edges_list = []
    hull_list = []
    contour_list = []
    for contour in contours:
        contour_edges, hull_edges, hull = single_contour_processing(contour)
        
        edge_differece = abs(len(contour_edges) - len(hull_edges))
        if edge_differece < threshold:
            contour_edges_list.append(contour_edges)
            hull_edges_list.append(hull_edges)
            hull_list.append(hull)
            contour_list.append(contour)
            
    return contour_edges_list, hull_edges_list, hull_list, contour_list

def draw_contours(img, contours, hulls,thickness=2):
    #cv2.drawContours(img, contours, -1, (0,255,0), thickness)
    cv2.drawContours(img, hulls, -1, (0,0,255), thickness)
    return img

def check_point_inside_bbox(point, bbox):
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x+w, y+h
    if point[0] > x1 and point[1] > y1 and point[0] < x2 and point[1] < y2:
        return True
    return False

def filter_contours_by_area(contours, img, bbox=None):
    filtered_contours = []
    # Bounding box of lower 40 percent of the image
    bbox = (0, int(img.shape[0]*0.6), img.shape[1], int(img.shape[0]*0.4))
    for contour in contours:
        if cv2.contourArea(contour) > 40 and check_point_inside_bbox(contour[0][0], bbox):
            filtered_contours.append(contour)
            
    return filtered_contours
    
def find_arrow_direction(hull_list):
    directions = []
    # Find the direction of sharpest angle of convex hull
    for hull in hull_list:
        if len(hull) >=4:
            direction = find_arrow_orientation(hull)
            directions.append(direction)
        else:
            directions.append("None")
    return directions
                
                
def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def find_arrow_orientation(contour):
    # Compute the centroid of the contour
    M = cv2.moments(contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])
    centroid = (centroid_x, centroid_y)

    # Find the endpoints of the longest line segment from the centroid to the contour points
    max_distance = 0
    start_point = end_point = None
    for point in contour:
        pt = (point[0][0], point[0][1])
        distance = np.linalg.norm(np.array(pt) - np.array(centroid))
        if distance > max_distance:
            max_distance = distance
            start_point = pt

    # Calculate the angle of the line segment relative to the y-axis
    angle = np.arctan2(start_point[0] - centroid[0], start_point[1] - centroid[1]) * 180 / np.pi

    print(angle)
    # Determine the orientation of the arrow based on the angle
    if -30 <= angle <= 30:
        return "Up"
    elif 30 < angle <= 120:
        return "Up"
    elif -120 <= angle < -30:
        return "Down"
    else:
        return "Left"

# Save the image
def save_image(img, path):
    cv2.imwrite(path, img)
    print(f" The image is saved in: {path}")
    

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])
        
def sampson_distance(F, p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    F = np.array(F)
    return np.abs(p2.T @ F @ p1) / (np.sqrt(F @ p1 @ p1.T + F.T @ p2 @ p2.T))


def compute_fundamental_matrix(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    # Match keypoints between the two images
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    
    # Extract corresponding keypoints
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute fundamental matrix using RANSAC
    fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    
    return fundamental_matrix

def compute_fundamental_matrix_sift(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # Match keypoints between the two images
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Extract corresponding keypoints
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute fundamental matrix using RANSAC
    fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    
    return fundamental_matrix

def compute_intersection_point(epipolar_line, width, height):
    """
    Compute intersection point of epipolar line with image boundaries.
    """
    x0, y0, z0 = epipolar_line
    if z0 != 0:
        x_intersection = 0
        y_intersection = int((-x0 * x_intersection - z0) / y0)
        if 0 <= y_intersection < height:
            return x_intersection, y_intersection
        else:
            y_intersection = height - 1
            x_intersection = int((-y0 * y_intersection - z0) / x0)
            if 0 <= x_intersection < width:
                return x_intersection, y_intersection
            else:
                return None, None
    else:
        return None, None
    
# 
def calculate_movement(image1, image2, bbox, flow, F):
    """
    This function classifies car movement in image2 within the bounding box (bbox) based on flow between image1 and image2 using Sampson distance and fundamental matrix.

    Args:
        image1 (np.ndarray): First image (grayscale).
        image2 (np.ndarray): Second image (grayscale).
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        flow (np.ndarray): Flow image representing displacement vectors.
        F (np.ndarray): Fundamental matrix.

    Returns:
        bool: True if moving, False if static.
    """

    # Extract flow vectors and image dimensions within bounding box
    flow_subset = flow[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    height, width = flow_subset.shape[:2]

    # Select salient points within the bounding box using corner detection
    corners = cv2.goodFeaturesToTrack(image1[bbox[1]:bbox[3], bbox[0]:bbox[2]], maxCorners=100, qualityLevel=0.01, minDistance=10)
    points1 = np.int0(corners).reshape(-1, 2)

    # Initialize list to store Sampson distances
    sampson_distances = []

    # Iterate through each point in the first frame within the bounding box
    for point1 in points1:
        x1, y1 = point1

        # Extract flow vector for this point
        flow_vec = flow_subset[y1, x1]

        # Expected displacement based on flow
        expected_displacement = flow_vec

        # Reprojection using fundamental matrix
        x1_homog = np.array([x1, y1, 1])
        epipolar_line = np.dot(F, x1_homog)
        x2_expected, y2_expected = compute_intersection_point(epipolar_line, width, height)

        # Check if expected reprojected point is within image bounds
        if x2_expected is not None and 0 <= x2_expected < width and 0 <= y2_expected < height:
            actual_displacement = [image2[y2_expected, x2_expected] - image1[y1, x1]]

            # Sampson distance calculation
            sampson_distance = np.linalg.norm(expected_displacement - actual_displacement) ** 2 / (expected_displacement[0] ** 2)
            sampson_distances.append(sampson_distance)

    # Classification based on average Sampson distance and threshold
    threshold = 2 # Adjust this based on your application and expected flow values
    print(len(sampson_distances))
    print("====================================")
    print("Min Sampson Distance: ", np.min(sampson_distances))
    print("Max Sampson Distance: ", np.max(sampson_distances))
    print("Average Sampson Distance: ", np.mean(sampson_distances))
    print("====================================")
    if len(sampson_distances) == 0:
        return True
	
    avg_sampson_distance = np.mean(sampson_distances)
    print("Average Sampson Distance: ", avg_sampson_distance)
    if avg_sampson_distance < threshold :
        return True  # Moving
    else:
        return False  # Static