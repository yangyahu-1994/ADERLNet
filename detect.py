import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images  # 判断图片是否要保存
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断是否是摄像头

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir  生成路径

    # Initialize
    set_logging()  # 设置日志
    device = select_device(opt.device)  # 选择设备
    half = device.type != 'cpu'  # half precision only supported on CUDA  浮点数精度

    # Load model 加载模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride 最大下采样
    imgsz = check_img_size(imgsz, s=stride)  # check img_size 判断输入图片的尺寸是否是32的倍数

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16 浮点数精度：半精度

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize 模型最终输出的每个框再进行一遍分类，确定分类的准确性，提高分类精度
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:  # 不是摄像头
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)  # 数据加载

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names  # 把模型里面的分类的名称拿出来
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]  # 产生0-255的随机数，每一组产生3个，即给每个类产生一个颜色

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once预跑
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()  # 计算时间
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)  # 从numpy弄成tensor，放到设备上
        img = img.half() if half else img.float()  # uint8 to fp16/32  把每个数字变成浮点数
        img /= 255.0  # 0 - 255 to 0.0 - 1.0  # 因为训练的时候就除了255
        if img.ndimension() == 3:  # 判断tensor有几个维度
            img = img.unsqueeze(0)  # 在这个tensor外面再加一维

        # Warmup热身
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()  # 计算时间
        pred = model(img, augment=opt.augment)[0]  # 预测结果
        t2 = time_synchronized()

        # Apply NMS 将小于置信度的全部删除掉
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections预测值进行输出
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path 输入路径
            save_path = str(save_dir / p.name)  # img.jpg  输出路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results  将检测结果画在图上
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'  # 画到图上的类别
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)  # 画图
                        # plot_one_box(xyxy, im0, label=None, color=colors[int(cls)], line_thickness=1)  # 画图时不要label

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:  # 是否要展示图片
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)  # 将图片保存在输出路径那里
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 视频每一秒的帧数
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频的宽
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频的高
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数：权重文件
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    # 要推理的图片/视频路径
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    # 推理图片的尺寸
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # 置信度阈值
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # IoU阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # 设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 展示检测结果
    parser.add_argument('--view-img', action='store_true', help='display results')
    # 保存结果
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 保存置信度
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 不保存图片/视频
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 不做输出的类别
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # 两个框都大于0.45的IoU阈值，取较大的那个，小的去除
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    # 实例化
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():  # 推理不需要梯度
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
