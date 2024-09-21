import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping
    # p:source, q:target
    if len(source_pts) == 0:
        return warped_image

    def warp(v):
        w = []
        sum_wi = 0
        sum_wi_pi = 0
        sum_wi_qi = 0
        for k in range(0, len(target_pts)):
            wi = (np.linalg.norm(v - target_pts[k]) + eps) ** (-2.0 * alpha)
            sum_wi += wi
            sum_wi_pi += wi * target_pts[k]
            sum_wi_qi += wi * source_pts[k]
            w.append(wi)
        w = np.array(w)
        p_star = sum_wi_pi / sum_wi
        q_star = sum_wi_qi / sum_wi
        pi__ = target_pts - p_star
        qi__ = source_pts - q_star
        v_p_star = v - p_star
        A = np.zeros((2, 2))
        A[:, 0] = v_p_star
        A[0, 1] = v_p_star[1]
        A[1, 1] = -v_p_star[0]
        tmp = np.zeros_like(pi__)
        tmp[:, 0] = pi__[:, 1]
        tmp[:, 1] = -pi__[:, 0]
        Ai = np.concatenate((pi__, tmp), axis=-1).reshape((-1, 2, 2))
        Aii = w[:, None, None] * (Ai @ A)
        f_vector_v = np.sum(np.einsum('ij,ijk->ik', qi__, Aii), axis=0)
        f_v = np.linalg.norm(v - p_star) * f_vector_v / (np.linalg.norm(f_vector_v) + eps) + q_star
        return f_v

    # ss x ss coarse grids and s x s fine grids with linear interpolate
    ss = 20  # resolution of coarse grids
    s = 5  # resolution of fine grids
    H = image.shape[0]
    W = image.shape[1]
    X = np.array([[range(W)] for _ in range(H)], dtype=np.float32).squeeze(1)
    Y = np.array([[range(H)] for _ in range(W)], dtype=np.float32).T.squeeze(1)
    minp = np.array([min(np.min(source_pts[:, 0]), np.min(target_pts[:, 0])),
                     min(np.min(source_pts[:, 1]), np.min(target_pts[:, 1]))])
    maxp = np.array([max(np.max(source_pts[:, 0]), np.max(target_pts[:, 0])),
                     max(np.max(source_pts[:, 1]), np.max(target_pts[:, 1]))])

    hx = np.array([])
    hy = np.array([])
    k = 0
    while k < W:
        hx = np.append(hx, k)
        if k <= minp[0] < k + ss:
            hx = np.concatenate([hx, np.arange(minp[0], maxp[0], s)])
            k = min(W - 1, ss * np.ceil(maxp[0] / ss))
            if k != maxp[0]:
                hx = np.append(hx, maxp[0])
            hx = np.append(hx, k)
        k += ss
    if hx[-1] != W - 1:
        hx = np.append(hx, W - 1)

    k = 0
    while k < H:
        hy = np.append(hy, k)
        if k <= minp[1] < k + ss:
            hy = np.concatenate([hy, np.arange(minp[1], maxp[1], s)])
            k = min(H - 1, ss * np.ceil(maxp[1] / ss))
            if k != maxp[1]:
                hy = np.append(hy, maxp[1])
            hy = np.append(hy, k)
        k += ss
    if hy[-1] != H - 1:
        hy = np.append(hy, H - 1)

    for i in range(hx.shape[0]):
        for j in range(hy.shape[0]):
            X[int(hy[j]), int(hx[i])], Y[int(hy[j]), int(hx[i])] = warp(np.array([hx[i], hy[j]]))
        if i != 0:
            for j in range(1, hy.shape[0]):
                x0 = int(hx[i - 1])
                x1 = int(hx[i])
                y0 = int(hy[j - 1])
                y1 = int(hy[j])
                xx = np.tile(np.linspace(0, 1, x1 - x0 + 1), y1 - y0 + 1)
                yy = np.repeat(np.linspace(0, 1, y1 - y0 + 1), x1 - x0 + 1)
                X[y0:y1 + 1, x0:x1 + 1] = (((xx - 1) * (yy - 1) * X[y0, x0] + xx * (1 - yy) * X[y0, x1] + \
                                            (1 - xx) * yy * X[y1, x0] + xx * yy * X[y1, x1]))\
                                            .reshape(y1 - y0 + 1, x1 - x0 + 1)
                Y[y0:y1 + 1, x0:x1 + 1] = (((1 - xx) * (1 - yy) * Y[y0, x0] + xx * (1 - yy) * Y[y0, x1] + \
                                            (1 - xx) * yy * Y[y1, x0] + xx * yy * Y[y1, x1]))\
                                            .reshape(y1 - y0 + 1, x1 - x0 + 1)
    X = np.clip(X, 0, W - 1)
    Y = np.clip(Y, 0, H - 1)
    warped_image = cv2.remap(image, X, Y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
