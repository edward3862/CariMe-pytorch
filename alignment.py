import os
import math
import linecache
from PIL import Image
import matplotlib.pyplot as plt
import copy
import threading

num_of_points = 17
time_enlarge = 1.3
resize_w = 256
resize_h = 256
path_data = 'data/WebCaricature'
path_output = 'data/WebCaricature_aligned'

path_images = os.path.join(path_data, 'OriginalImages')
path_points = os.path.join(path_data, 'FacialPoints')
path_output_images = os.path.join(path_output, 'image')
path_output_points = os.path.join(path_output, 'landmark')


def main():
    root = path_points
    image_list = list_all_images(root)
    start_threads(image_list)


def get_point_from_line(path, x):
    line = linecache.getline(path, x)
    x, y = line.strip().split(' ')
    return float(x), float(y)


def get_points_from_txt(path):
    points = [[0] * 2 for i in range(num_of_points)]
    for i in range(num_of_points):
        x, y = get_point_from_line(path, i+1)
        points[i][0] = x
        points[i][1] = y
    return points


def load_landmark(path):
    result = [[0] * 2 for i in range(num_of_points)]
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    f.close()
    for i in range(17):
        result[i][0] = int(lines[i].split('\t')[0])
        result[i][1] = int(lines[i].split('\t')[1])
    return result


def get_rotate_angle(points):
    xl = (points[8][0] + points[9][0]) / 2
    yl = (points[8][1] + points[9][1]) / 2
    xr = (points[10][0] + points[11][0]) / 2
    yr = (points[10][1] + points[11][1]) / 2
    if xl == xr:
        if yr > yl:
            return 90
        elif yr < yl:
            return -90
        else:
            print(points)
            raise RuntimeError('x=x,y=y')
    tan_x = (yr - yl) / (xr - xl)
    x_rad = math.atan(tan_x)
    x_angle = (180 * x_rad) / math.pi
    return x_angle


def rotate_image(name, filename, angle):
    path_image = os.path.join(path_images, name, filename)
    image = Image.open(path_image)
    image_rotated = image.rotate(angle, Image.BILINEAR)
    w, h = image.size
    return image_rotated, w, h


def calculate_new_point(x0, y0, angle, w, h):
    if angle == 0:
        return x0, y0
    angle *= -(math.pi / 180)
    x1 = x0 - w / 2
    y1 = y0 - h / 2
    r_square = x1 * x1 + y1 * y1
    if x1 == 0:
        tanx = -(1 / math.tan(angle))
    else:
        if y1 * math.tan(angle) == x1:
            x2 = 0
            y2 = math.sqrt(r_square)
            return x2, y2
        elif 1 - math.tan(angle) * (y1 / x1) == 0:
            x2 = 0
            y2 = math.sqrt(r_square)
            return x2, y2
        else:
            tanx = (y1 / x1 + math.tan(angle)) / (1 - math.tan(angle) * (y1 / x1))
    x2_square = r_square / (1 + tanx * tanx)
    x2_1 = math.sqrt(x2_square)
    y2_1 = x2_1 * tanx
    x2_2 = -x2_1
    y2_2 = -y2_1
    d1_square = (x1 - x2_1) * (x1 - x2_1) + (y1 - y2_1) * (y1 - y2_1)
    d2_square = (x1 - x2_2) * (x1 - x2_2) + (y1 - y2_2) * (y1 - y2_2)
    if d1_square < d2_square:
        x2 = x2_1
        y2 = y2_1
    else:
        x2 = x2_2
        y2 = y2_2
    x2 += w / 2
    y2 += h / 2
    return x2, y2


def calculate_new_points(points, angle, w, h):
    if angle == 0:
        return points
    else:
        result = [[0] * 2 for i in range(num_of_points)]
        for i in range(num_of_points):
            x0 = points[i][0]
            y0 = points[i][1]
            x2, y2 = calculate_new_point(x0, y0, angle, w, h)
            result[i][0] = x2
            result[i][1] = y2
        return result


def calculate_boundingbox(points):
    max_list = []
    min_list = []
    for j in range(len(points[0])):
        list = []
        for i in range(len(points)):
            list.append(points[i][j])
        max_list.append(max(list))
        min_list.append(min(list))
    x_max = max_list[0]
    y_max = max_list[1]
    x_min = min_list[0]
    y_min = min_list[1]

    delta_x = x_max - x_min
    delta_y = y_max - y_min
    length = abs(delta_x - delta_y) / 2
    if delta_x > delta_y:
        y_min -= length
        y_max += length
    else:
        x_min -= length
        x_max += length

    return x_max, x_min, y_max, y_min


def enlarge(x_max, x_min, y_max, y_min, time_enlarge, w, h):
    nx_max = x_max + (time_enlarge - 1) * (x_max - x_min) / 2
    ny_max = y_max + (time_enlarge - 1) * (y_max - y_min) / 2
    nx_min = x_min - (time_enlarge - 1) * (x_max - x_min) / 2
    ny_min = y_min - (time_enlarge - 1) * (y_max - y_min) / 2
    return nx_max, nx_min, ny_max, ny_min


def look(img, points, savepath):
    plt.clf()
    xs = [points[i][0] for i in range(17)]
    ys = [points[i][1] for i in range(17)]
    plt.imshow(img)
    plt.scatter(xs, ys, s=16)
    plt.savefig(savepath)
    plt.close()


def update_landmark_cropped(landmark, x_min, y_min):
    result = copy.deepcopy(landmark)
    for i in range(17):
        result[i][0] = landmark[i][0] - x_min
        result[i][1] = landmark[i][1] - y_min
    return result


def update_landmark_enlarged(landmark, w, h, resize_w, resize_h):
    time_w = resize_w / w
    time_h = resize_h / h
    for i in range(17):
        landmark[i][0] = landmark[i][0] * time_w
        landmark[i][1] = landmark[i][1] * time_h
    return landmark


def save_landmark(landmark, dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(17):
            f.write(str(int(landmark[i][0])) + '\t' + str(int(landmark[i][1])) + '\n')
    f.close()


def list_all_images(root):
    result = []
    for name in os.listdir(root):
        for file in os.listdir(os.path.join(root, name)):
            result.append(os.path.join(root, name, file))
    return result


def start_threads(image_list, n_threads=16):
    if n_threads > len(image_list):
        n_threads = len(image_list)
    n = int(math.ceil(len(image_list) / float(n_threads)))
    print('the thread num is {}'.format(n_threads))
    print('each thread images num is {}'.format(n))
    image_lists = [image_list[index:index + n] for index in range(0, len(image_list), n)]
    thread_list = {}
    for thread_id in range(n_threads):
        thread_list[thread_id] = MyThread(image_lists[thread_id], thread_id)
        thread_list[thread_id].start()

    for thread_id in range(n_threads):
        thread_list[thread_id].join()


class MyThread(threading.Thread):
    def __init__(self, image_list, thread_id):
        threading.Thread.__init__(self)
        self.image_list = image_list
        self.thread_id = thread_id

    def run(self):
        print('thread {} begin'.format(self.thread_id))

        image_len = len(self.image_list)
        print_interval = image_len // 100
        print_interval = print_interval if print_interval > 0 else 1

        for index, image_path in enumerate(self.image_list):
            name = image_path.split('/')[-2]
            filename = image_path.split('/')[-1][:-4]
            try:
                path_point_txt = image_path
                points = get_points_from_txt(path_point_txt)
                angle = get_rotate_angle(points)
                image_rotated, w, h = rotate_image(name, filename + '.jpg', angle)
                points_rotated = calculate_new_points(points, angle, w, h)
                x_max, x_min, y_max, y_min = calculate_boundingbox(points_rotated)
                x_max, x_min, y_max, y_min = enlarge(x_max, x_min, y_max, y_min, time_enlarge, w, h)
                image_cropped = image_rotated.crop((x_min, y_min, x_max, y_max))
                points_cropped = update_landmark_cropped(points_rotated, x_min, y_min)
                image_result = image_cropped.resize((resize_w, resize_h), Image.BILINEAR)
                w_cropped, h_cropped = image_cropped.size
                points_result = update_landmark_enlarged(points_cropped, w_cropped, h_cropped, resize_w, resize_h)
                dir = os.path.join(path_output_points, name)
                save_landmark(points_result, dir, filename + '.txt')
                path_output_image = os.path.join(path_output_images, name)
                if not os.path.exists(path_output_image):
                    os.makedirs(path_output_image)
                image_result.save(os.path.join(path_output_image, filename + '.jpg'), 'jpeg')
            except ZeroDivisionError:
                print(name)
                print(filename)
            if index % print_interval == 0 and index > 0:
                print('{}/{} in thread {} has been sloven'
                      .format(index, image_len, self.thread_id))

        print('thread {} end.'.format(self.thread_id))


if __name__ == '__main__':
    main()
