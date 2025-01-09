import os
import tempfile

import h5py
import numpy as np
import torch
from PIL import Image
# Create your views here.
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from scipy.ndimage import zoom

from .utils import load_net, SegmentationModel
from .utils import predict_mask

# 加载模型
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'unet_best_model.pth')
net = SegmentationModel(in_chns=1, class_num=4).cuda()
model = load_net(net, MODEL_PATH)
# print(model)


@csrf_exempt
def segment_image(request):
    if request.method == 'POST':
        # 获取上传的文件
        image_file = request.FILES.get('file')
        if not image_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        try:
            # 将文件保存到临时目录
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                for chunk in image_file.chunks():
                    temp_file.write(chunk)
                temp_file.seek(0)

                # 加载图像并进行处理
                image = Image.open(temp_file.name).convert('RGB')
                mask = predict_mask(model, image)

                # 将 mask 转为列表返回
                mask_list = mask.tolist()
                return JsonResponse({'mask': mask_list})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def medical(request):
    return HttpResponse('success!!!')


@csrf_exempt
def upload_file(request):
    if request.method == 'GET':
        return render(request, 'upload.html')

    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file provided."})

        # Save the uploaded file
        uploaded_file = request.FILES['file']
        print(uploaded_file.name)
        file_name = uploaded_file.name
        # file_path = default_storage.save(f'medical/data/{file_name}', uploaded_file)
        file_path = os.path.join('medical', 'data', file_name)
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the uploaded file to the file system
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        # Perform segmentation
        try:
            # Read .h5 file content
            with h5py.File(file_path, 'r') as h5_file:
                if 'image' not in h5_file:
                    return JsonResponse({"error": "Key 'image' not found in .h5 file."})
                image_data = np.array(h5_file['image'])
                print(image_data.shape)

            # Save the original image for display (input image is already a single slice)
            original_image_path = os.path.join('static', 'medical', 'data',
                                               f'{os.path.splitext(file_name)[0]}_original.png')
            print(original_image_path)

            # 确保目录存在
            os.makedirs(os.path.dirname(original_image_path), exist_ok=True)

            # 保存图片
            original_image = Image.fromarray((image_data / np.max(image_data) * 255).astype(np.uint8))
            original_image.save(original_image_path)

            # segmentation
            pred = predict_single_image(image_data, model)
            # print('pred ----------------------- ',pred)
            # print(pred.shape)


            # Post-process segmentation result
            normalized_prediction = (pred - pred.min()) / (pred.max() - pred.min()) * 255
            normalized_prediction = normalized_prediction.astype(np.uint8)  # 转换为 uint8 类型

            # 将 NumPy 数组转换为图片
            result_image = Image.fromarray(normalized_prediction)

            # Save the segmentation result
            result_file_name = f'{os.path.splitext(file_name)[0]}_segmentation.png'
            result_path = f'medical/segment/{result_file_name}'

            # Ensure directory exists for saving the file
            segment_dir = os.path.join('static', 'medical', 'segment')
            os.makedirs(segment_dir, exist_ok=True)

            # Save the image directly to the filesystem
            full_result_path = os.path.join(segment_dir, result_file_name)
            result_image.save(full_result_path)

            return JsonResponse({
                "message": "File uploaded and segmented successfully.",
                "result_path": f'{result_file_name}',
                "original_image_path": original_image_path,
            })
        except Exception as e:
            return JsonResponse({"error": f"Segmentation failed: {str(e)}"})
    else:
        return JsonResponse({"error": "Invalid request method. Use POST."})



def predict_single_image(image, net, patch_size=[256, 256]):
    """
    对单个2D图像使用神经网络进行预测。
    参数:
    - image: 输入的2D图像，形状为 (Height, Width)。
    - net: 用于预测的神经网络。
    - patch_size: 用于调整图像大小的目标尺寸（默认 [256, 256]）。

    返回:
    - prediction: 网络预测结果，大小与输入图像一致。
    """
    # 获取输入图像的原始大小
    original_size = image.shape  # (256, 212)

    # 调整图像大小到目标 patch_size
    x, y = original_size
    resized_image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    # resized_image 的形状为 [256, 256]

    # 转换为 PyTorch Tensor，并增加批次和通道维度
    input_tensor = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()
    # input_tensor 的形状为 [1, 1, 256, 256]

    # 将模型设置为评估模式
    net.eval()

    # 禁用梯度计算，进行推理
    with torch.no_grad():
        output = net(input_tensor)
        # output 的形状通常为 [1, num_classes, 256, 256]，取决于网络的输出设计

        # 获取每个像素的预测类别（概率最大的类别索引）
        output = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
        # output 的形状变为 [256, 256]

        # 转换为 NumPy 数组
        output_np = output.cpu().detach().numpy()

    # 将预测结果缩放回原始大小
    prediction = zoom(output_np, (x / patch_size[0], y / patch_size[1]), order=0)
    # prediction 的形状为 [256, 212]，与输入图像大小一致

    return prediction


# def result_view(request, result_file_name):
#     """
#     Display the segmented image result.
#     """
#     print('-------------',result_file_name)
#     result_path = os.path.join('static', 'medical', 'segment', result_file_name)
#     print(result_path)
#     normalized_path = result_path.replace("\\", "/")
#     return render(request, 'result.html', {"result_path": normalized_path})
def result_view(request, result_file_name):
    """
    Display the original uploaded image and the segmentation result.
    """
    # Assuming the original image is stored with '_original' suffix
    result_path = os.path.join('static', 'medical', 'segment', result_file_name)
    original_path = os.path.join('static', 'medical', 'data', result_file_name)
    original_image_path = original_path.replace('_segmentation.png', '_original.png')

    # Check if the original image exists; if not, return an error
    if not os.path.exists(original_image_path):
        return render(request, 'error.html', {"message": "Original image not found."})

    result_path = result_path.replace("\\", "/")
    original_image_path = original_image_path.replace("\\", "/")
    # Render the result page with both images
    return render(request, 'result.html', {
        "result_path": result_path,
        "original_image_path": original_image_path,
    })
