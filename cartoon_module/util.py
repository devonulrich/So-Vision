import cv2

# rects are (x, y, width, height)
def IoU(rect1, rect2):
    lo1_x = rect1[0]
    lo1_y = rect1[1]
    hi1_x = rect1[0] + rect1[2]
    hi1_y = rect1[1] + rect1[3]

    lo2_x = rect2[0]
    lo2_y = rect2[1]
    hi2_x = rect2[0] + rect2[2]
    hi2_y = rect2[1] + rect2[3]

    intlo_x = max(lo1_x, lo2_x)
    intlo_y = max(lo1_y, lo2_y)
    inthi_x = min(hi1_x, hi2_x)
    inthi_y = min(hi1_y, hi2_y)

    intArea = (inthi_x - intlo_x) * (inthi_y - intlo_y)
    if intArea <= 0:
        return 0
    
    area1 = (hi1_x - lo1_x) * (hi1_y - lo1_y)
    area2 = (hi2_x - lo2_x) * (hi2_y - lo2_y)
    unionArea = area1 + area2 - intArea

    if unionArea == 0:
        return 0
    else:
        return intArea / unionArea

def load_img(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
