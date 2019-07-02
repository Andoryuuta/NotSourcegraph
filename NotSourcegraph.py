import cv2
import math

def minmax(a, b):
    if a > b:
        return (b, a)
    return (a, b)

def circle_intersect(x0, y0, x1, y1, r0, r1):
    return math.hypot(x0-x1, y0-y1) <= (r0+r1)

def contour_info(c):
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return x, y, radius, center

def check_contours_intersect(contours, test_contour):
    tx, ty, trad, rcenter = contour_info(test_contour)

    for c in contours:
        x, y, rad, center = contour_info(c)
        if circle_intersect(center[0], center[1], rcenter[0], rcenter[1], trad, rad):

            # Get the percent different in contour area.
            a0, a1 = minmax(cv2.contourArea(c), cv2.contourArea(test_contour))
            area_diff = ((a1-a0)/a1)*100

            if area_diff >= cv2.getTrackbarPos("c_area_min", "Not Sourcegraph") and area_diff <= cv2.getTrackbarPos("c_area_max", "Not Sourcegraph"):
                return True

    return False

def main():
    cam = cv2.VideoCapture(0)

    # Setup the window w/ setting trackbars.
    cv2.namedWindow("Not Sourcegraph")
    cv2.createTrackbar("c_area_min", "Not Sourcegraph", 0, 1000, lambda x: None)
    cv2.createTrackbar("c_area_max", "Not Sourcegraph", 93, 1000, lambda x: None)

    # Load the bad "Sourcegraph" and "Not Sourcegraph" overlay images.
    sg_img = cv2.imread('data/sg.png')
    not_sg_img = cv2.imread('data/not_sg.png')
    sg_alpha_img = cv2.imread('data/alpha.png')

    while True:
        ok, image = cam.read()
        if not ok:
            break

        # Make copies of the image for different displays.
        blue_contour_image = image.copy()
        orange_contour_image = image.copy()
        purple_contour_image = image.copy()
        out_image = image.copy()

        # Convert the image from BGR to HSV for masking.
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Mask out the images by color.
        blue_mask = cv2.inRange(image_hsv, (98, 93, 100), (111, 172, 255))
        orange_mask = cv2.inRange(image_hsv, (0, 110, 114), (22, 255, 255))
        purple_mask = cv2.inRange(image_hsv, (114, 65, 67), (164, 157, 255))

        # Blur to remove noise.
        gaussian_blur_blue = cv2.GaussianBlur(blue_mask, (0,0), 2)
        gaussian_blur_orange = cv2.GaussianBlur(orange_mask, (0,0), 2)
        gaussian_blur_purple = cv2.GaussianBlur(purple_mask, (0,0), 2)
    
        # Find the contours.
        blue_contours, blue_hierarchy = cv2.findContours(gaussian_blur_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        orange_contours, orange_hierarchy = cv2.findContours(gaussian_blur_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        purple_contours, purple_hierarchy = cv2.findContours(gaussian_blur_purple, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        

        # Draw the contours on the images.
        cv2.drawContours(blue_contour_image, blue_contours, -1, (0, 255, 0), 3)
        cv2.drawContours(orange_contour_image, orange_contours, -1, (0, 255, 0), 3)
        cv2.drawContours(purple_contour_image, purple_contours, -1, (0, 255, 0), 3)

        # Draw the circles on the images.
        for c in blue_contours:
            x, y, radius, center = contour_info(c)
            cv2.circle(blue_contour_image, (int(x), int(y)), int(radius), (0, 0, 255), 1)

        for c in orange_contours:
            x, y, radius, center = contour_info(c)
            cv2.circle(orange_contour_image, (int(x), int(y)), int(radius), (0, 0, 255), 1)

        for c in purple_contours:
            x, y, radius, center = contour_info(c)
            cv2.circle(purple_contour_image, (int(x), int(y)), int(radius), (0, 0, 255), 1)


        # Find the sourcegraph logo by searching for grouped contours of blue,
        # then checking if contours of orange and purple intersecting it,
        # limited by the contour area difference between the blue and (orange, purple) areas.
        is_sg = False
        for bc in blue_contours:
            bx, by, bradius, bcenter = contour_info(bc)

            found_orange = check_contours_intersect(orange_contours, bc)
            found_purple = check_contours_intersect(purple_contours, bc)

            if found_purple and found_orange:
                is_sg = True
                cv2.circle(out_image, (int(bx), int(by)), int(bradius), (0, 255, 255), 2)


        # Add the overlay.
        # (Overlay code c+p'ed from https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/)
        if is_sg:
            foreground = sg_img
        else:
            foreground = not_sg_img

        background = out_image
        alpha = sg_alpha_img
        foreground = foreground.astype(float)
        background = background.astype(float)
        alpha = alpha.astype(float)/255
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        outImage = cv2.add(foreground, background)
        out_image = outImage/255


        # Show the images.
        cv2.imshow("blue_contour_image", blue_contour_image)
        cv2.imshow("orange_contour_image", orange_contour_image)
        cv2.imshow("purple_contour_image", purple_contour_image)
        cv2.imshow("Not Sourcegraph", out_image)


        if cv2.waitKey(1) & 0xFF is ord('q'):
            break


if __name__ == '__main__':
    main()