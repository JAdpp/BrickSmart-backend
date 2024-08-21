import json
import os
import time
from .models import ModelTask, ComponentList
from .forms import PromptForm, ImageUploadForm
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import requests
import re
import django.http.request
from BackEnd.settings import BASE_DIR, MEDIA_ROOT
import threading
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import qiniu
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pulp
from openai import OpenAI
import trimesh
from scipy.ndimage import binary_fill_holes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from django.conf import settings
import base64

#OpenAI密钥(不要外传)
openai_api_key = 'your openai api key'
# 初始化OpenAI客户端
client = OpenAI(api_key=openai_api_key)

#七牛云密钥（不要外传）
access_key = 'your qiniu access key'
secret_key = 'your qiniu secret key'

# 构建七牛云鉴权对象
q = qiniu.Auth(access_key, secret_key)

# 要上传的七牛云空间名称
bucket_name = 'stay33'

#七牛云上传函数
def upload_file_to_qiniu(local_file_path, key):
    # 生成上传Token，可以指定过期时间等
    token = q.upload_token(bucket_name, key, 3600)

    # 构建一个上传对象
    ret, info = qiniu.put_file(token, key, local_file_path)

    if info.status_code == 200:
        print(f"Upload success! URL: http://your_domain/{key}")
        return f"https://qiniu.staykoi.asia/{key}"
    else:
        print("Upload failed!")
        print(info)
        return None

#下载并保存文件函数
def download_and_save_file(url, folder, task_id, extension):
    response = requests.get(url)
    if response.status_code == 200:
        file_path = f"{BASE_DIR.parent}/{folder}/{task_id}.{extension}"
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File saved to {file_path}")

        # 上传到七牛云
        key = f"{folder}/{task_id}.{extension}"
        qiniu_url = upload_file_to_qiniu(file_path, key)
        return file_path, qiniu_url
    else:
        print(f"Failed to download file from {url}")
        return None

# 判断是否结束对话的函数
def check_end_conversation(user_input):
    end_phrases = ["结束", "可以了", "停止", "结束对话", "可以结束", "我觉得可以了","OK","就这样"]
    for phrase in end_phrases:
        if phrase in user_input:
            return True
    return False

## 文生3D视图函数，通过调用Tripo AI的API实现
@csrf_exempt
def generate_model(request):
    # 处理GET请求，加载页面并显示组件选择部分
    if request.method == 'GET':
        # 获取所有元件列表，并传递给模板渲染
        # components = ComponentList.objects.all()
        return render(request, 'prompt.html')
    
    # 处理POST请求，分为生成模型和处理对话两种情况
    if request.method == 'POST':
        # components = ComponentList.objects.all()
        
        # 检查是否是生成模型的请求
        if 'generate_model' in request.POST:
            # 从会话中获取对话总结后的提示词（prompt）
            prompt = request.session.get('lego_prompt')
            
            # 如果找不到提示词，返回错误响应
            if not prompt:
                return JsonResponse({'error': 'Prompt not found'}, status=400)
            
            # 从请求中获取用户选择的组件
            # selected_components = json.loads(request.POST.get('components', '[]'))

            # 如果用户没有选择任何组件，返回错误响应
            #if not selected_components:
            #    return JsonResponse({'error': 'No components selected'}, status=400)

            # 调用GPT-4o-mini优化提示词，使其更适合生成3D模型
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a master at designing text prompts for generating 3D models, \
                                                    specializing in transforming children's whimsical words and narrated diaries into exquisite cartoon-style 3D models. \
                                                    Your prompts should focus on describing a single object rather than a scene, \
                                                    ensuring the description is suitable for conversion into LEGO models. \
                                                    The output should be a detailed sentence or a list of descriptive words separated by commas.\
                                                    Writing in English. " },
                    {"role": "user", "content": prompt}
                ],
            )
            
            # 获取优化后的提示词
            transformer_answer = completion.choices[0].message.content
            print(transformer_answer)
            
            # 将用户选择的组件转换为JSON格式
            # bricks_json = json.dumps(selected_components)
            
            # 调用Tripo3D API，将优化后的提示词转换为3D模型
            api_url = 'https://api.tripo3d.ai/v2/openapi/task'
            api_key = "Your Tripo API key"
            header = {
                'Authorization': f'Bearer {api_key}'
            }
            
            # 向Tripo3D API发送请求，生成3D模型
            response = requests.post(api_url, json={'type': 'text_to_model', 'prompt': transformer_answer}, headers=header)
            if response.status_code != 200:
                return JsonResponse({'error': 'API request failed', 'details': response.text}, status=response.status_code)
            
            # 处理API的响应，检查是否成功生成任务
            create_response_data = response.json()
            if 'data' not in create_response_data:
                return JsonResponse({'error': 'API response error', 'details': create_response_data}, status=500)
            
            # 获取生成任务的ID
            task_id = create_response_data['data']['task_id']
            print(f"Task ID: {task_id}")
            
            # 构建用于获取任务结果的URL和请求头
            get_task_url = f"https://api.tripo3d.ai/v2/openapi/task/{task_id}"
            get_headers = {
                'Authorization': f'Bearer {api_key}'
            }
            
            # 等待一段时间确保任务完成（25秒）
            time.sleep(25)
            
            # 获取任务结果，检查是否成功生成3D模型
            get_response = requests.get(get_task_url, headers=get_headers)
            if get_response.status_code != 200:
                return JsonResponse({'error': 'API request failed', 'details': get_response.text}, status=get_response.status_code)

            get_response_data = get_response.json()
            if 'data' not in get_response_data or get_response_data['data'].get('status') != 'success':
                return JsonResponse({'error': 'API response error', 'details': get_response_data}, status=500)
            
            # 下载保存并上传生成的3D模型和渲染图像到云存储
            model_url = get_response_data['data']['output']['model']
            image_url = get_response_data['data']['output']['rendered_image']

            model_file_path, model_qiniu_url = download_and_save_file(model_url, '3D-model', task_id, 'glb')
            image_file_path, image_qiniu_url = download_and_save_file(image_url, 'rendered-image', task_id, 'webp')


            # 获取模型名称并创建输出乐高教程输出目录
            model_name = task_id  # 使用task_id作为文件夹名称
            output_dir = os.path.join(settings.MEDIA_ROOT, 'tutorial', model_name)

            # 创建乐高教程输出输出目录（如果不存在）
            os.makedirs(output_dir, exist_ok=True)

            # 加载3D模型并转换为体素矩阵
            mesh = load_model(model_file_path)
            resolution = 20
            voxel_matrix = convert_to_voxel(mesh, resolution)

            # 将体素矩阵保存到会话中
            request.session['voxel_matrix'] = voxel_matrix.tolist()  # 将numpy数组转换为列表，才能存入会话中

            # 保存体素矩阵的可视化结果
            voxel_image_path = os.path.join(output_dir, 'voxel_model.png')
            visualize_and_save_voxel_matrix(voxel_matrix, voxel_image_path)
            
            # 将相对路径转换为基于 MEDIA_URL 的路径
            voxel_image_url = os.path.join(settings.MEDIA_URL, 'tutorial', model_name, 'voxel_model.png')            

            # 获取当前用户（如果用户已登录）
            user = request.user if request.user.is_authenticated else None
            
            # 将任务数据保存到数据库
            ModelTask.objects.create(
                task_id=task_id,
                prompt=transformer_answer,
                # bricks=bricks_json,
                model_download_url=model_qiniu_url,
                image_download_url=image_qiniu_url,
                #lego_url="",
                user=user
            )
            
            # 返回生成的3D模型、体素矩阵和渲染图像的URL
            return JsonResponse({'model_filename': model_qiniu_url, 'image_filename': image_qiniu_url, 'local_model_path': model_file_path,'voxel_image_path': voxel_image_url, 'task_id': task_id})

        
        # 如果是处理用户与智能助理对话的请求
        user_input = request.POST.get('prompt', '').strip()
        
        # 检查用户输入是否为空
        if not user_input:
            return JsonResponse({'error': 'Invalid input data'}, status=400)
        
        # 从会话中获取当前的对话历史，如果没有则初始化一个新的对话历史
        conversation_history = request.session.get('conversation_history', [
            {"role": "system", "content": "你是一个儿童的空间语言训练师，\
                                        你的职责是帮助家长引导孩子并提升空间语言表达能力。你当前的任务是根据孩子描述的场景或物体，进行对话。\
                                        引导家长让孩子描述得的更清楚，包含形状（边、曲线、圆、线等）、位置和方向（描述相关位置）、大小关系、指示词（这里、那里、哪里等）。"}
        ])

        # 如果用户输入表示结束对话
        if check_end_conversation(user_input):
            # 将用户输入添加到对话历史中
            conversation_history.append({"role": "user", "content": user_input})
            
            # 调用GPT-4o-mini生成对话总结
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_history + [{"role": "user", "content": "请用一个长句子总结孩子的描述。格式为：‘好的，孩子的描述如下：……’"}],
            )
            prompt = completion.choices[0].message.content
            
            # 将生成的总结保存到会话中，以便稍后生成3D模型
            request.session['lego_prompt'] = prompt

            return JsonResponse({'message': prompt, 'done': True})
        
        else:
            # 如果对话未结束，将用户输入添加到对话历史中
            conversation_history.append({"role": "user", "content": user_input})

            # 调用GPT-4o-mini继续对话
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_history,
            )

            # 获取助手的回复并添加到对话历史中
            model_reply = completion.choices[0].message.content

            conversation_history.append({"role": "assistant", "content": model_reply})
            
            # 更新会话中的对话历史
            request.session['conversation_history'] = conversation_history

            return JsonResponse({'message': model_reply, 'done': False})
    
    # 如果请求方法不是GET或POST，返回错误响应
    return JsonResponse({'error': 'Invalid request method'}, status=405)



# 图生3D视图函数，通过调用Tripo AI的api实现
@csrf_exempt
def generate_model_image(request):
    # 获取所有元件列表
    components = ComponentList.objects.all() 
    if request.method == 'POST':  # 判断请求方法是否为POST
        form = ImageUploadForm(request.POST, request.FILES)
        print(request.FILES) # 打印上传的文件信息

        if form.is_valid(): # 判断表单是否有效
            image = form.cleaned_data['image'] # 获取上传的图片
            image_path = default_storage.save('uploads/' + image.name, ContentFile(image.read())) # 将图片保存到指定路径

            # 构建元件信息列表
            selected_components = []
            for component in components:
                try:
                    component_qty = int(request.POST.get(f'component_qty_{component.pid}', 0))
                    if int(component_qty) > 0:
                        selected_components.append({'sn': component.sn, 'qty': component_qty})
                except ValueError:
                    continue  # 如果转换失败，跳过这个元件
            # JSON化元件信息
            bricks_json = json.dumps(selected_components)

            # 设置API的URL和密钥
            api_url = "https://api.tripo3d.ai/v2/openapi/upload"
            api_key = "tsk_S9zAZ08NFPuKt9le3qqr6rDbHUQO38dVoqm7zOf1U49"
            headers = {
                "Authorization": f"Bearer {api_key}"
            }

            # 构建文件路径和文件类型
            file_path = os.path.join(MEDIA_ROOT, image_path)
            files = {'file': (file_path, open(file_path, 'rb'), 'image/jpeg')}
            print(files)

            # 发送POST请求以上传图像并获取响应
            response = requests.post(api_url, headers=headers, files=files).json()
            print(response)

            if response['code'] == 0:
                file_token = response['data']['image_token']
                print(f"image token: {file_token}")
                create_task_url = "https://api.tripo3d.ai/v2/openapi/task"
                create_headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }
                create_data = {
                    'type': 'image_to_model',
                    "file": {
                        "type": "jpg",
                        "file_token": file_token
                    }
                }

                # 等待一段时间，确保图像上传任务处理完成
                # 实际等待时间可能需要根据任务的复杂度调整
                time.sleep(5)

                # 创建任务用于将图像转换为模型
                create_response = requests.post(create_task_url, json=create_data, headers=create_headers)
                create_response_data = create_response.json()

                if create_response_data['code'] == 0:
                    task_id = create_response_data['data']['task_id']
                    print(f"Task ID: {task_id}")

                    # 构建用于获取结果的 URL 和请求头
                    get_task_url = f"https://api.tripo3d.ai/v2/openapi/task/{task_id}"
                    get_headers = {
                        'Authorization': f'Bearer {api_key}'
                    }

                    # 等待一段时间，确保任务处理完成
                    time.sleep(25)  # 实际等待时间可能需要根据任务的复杂度调整

                    # 获取任务结果
                    get_response = requests.get(get_task_url, headers=get_headers)
                    get_response_data = get_response.json()
                    if get_response_data['data']['status'] == 'success':
                        model_url = get_response_data['data']['output']['model']
                        image_url = get_response_data['data']['output']['rendered_image']

                    # 使用线程异步下载并保存模型和渲染图像文件
                    threading.Thread(target=download_and_save_file,
                                        args=(model_url, '3D-model', task_id, 'glb')).start()
                    threading.Thread(target=download_and_save_file,
                                        args=(image_url, 'rendered-image', task_id, 'webp')).start()

                    # 保存任务数据到数据库
                    ModelTask.objects.create(
                        task_id=task_id,
                        prompt=image_path,
                        bricks=bricks_json,  
                        model_download_url= f"3D-model/{task_id}.glb",
                        image_download_url= f"rendered-image/{task_id}.webp",
                        lego_url="",  # Assuming this gets set elsewhere
                        user=request.user
                    )

                    # 返回模型页面并展示模型URL
                    return render(request, 'model.html', {'model_url': model_url})
    else:
        form = ImageUploadForm()

    # 返回上传页面或者表单错误信息
    return render(request, 'upload.html', {'form': form, 'components': components})



## 3D模型转乐高积木相关变量、函数&视图
# 定义算法中的参数
alpha_1 = 4.0
alpha_2 = 0.8
e_max = 1.0
rho = 25

# 乐高积木的列表，定义了不同的尺寸
brick_list = [[1, 2, 3, 4],
                [1, 2, 3, 4]]
# bricks you have, namely 1x1, 1x2, 1x3, 1x4, 2x1, 2x2, 2x3, 2x4

HEIGHT = 20

# 加载3D模型
def load_model(file_path):
    mesh = trimesh.load(file_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh

# 将3D模型转换为体素矩阵
def convert_to_voxel(mesh, resolution):
    # Scale and translate the mesh to fit in the voxel grid
    scale_factor = (resolution - 1) / np.max(mesh.extents)
    mesh.apply_scale(scale_factor)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_translation([resolution / 2] * 3)
    
    # Create an empty voxel matrix
    voxel_matrix = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # Fill voxel matrix based on mesh vertices
    indices = np.floor(mesh.vertices).astype(int)
    voxel_matrix[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    
    # Fill holes and rotate for correct orientation
    voxel_matrix = binary_fill_holes(voxel_matrix).astype(int)
    
    # Adjust orientation: Transpose and rotate the voxel matrix
    voxel_matrix = np.transpose(voxel_matrix, (0, 2, 1))
    voxel_matrix = np.rot90(voxel_matrix, k=1, axes=(1, 0))
    
    return voxel_matrix

# 可视化并保存体素矩阵
def visualize_and_save_voxel_matrix(matrix, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(matrix, edgecolor='k')
    plt.savefig(save_path)
    plt.close()

# 定义乐高积木的类
class Brick:
    def __init__(self, x, y, l, w):  # default height is 1, width is chosen from 1 or 2
        self.x0 = x
        self.y0 = y
        self.length = l
        self.width = w

# 创建乐高积木
def create_brick(x, y, z, l, w):
    brick_height = 1.2 # 砖块高度
    stud_radius = 0 # 小圆柱半径
    stud_height = 0 # 小圆柱高度
    verts = []
    faces = []
    base_verts = [
        [0, 0, 0], [w, 0, 0], [w, l, 0], [0, l, 0],
        [0, 0, brick_height], [w, 0, brick_height], [w, l, brick_height], [0, l, brick_height]
    ]
    faces.append([0, 1, 2, 3])
    faces.append([4, 5, 6, 7])
    faces.extend([
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7]
    ])
    
    # 将z乘以 brick_height 以确保每一层砖块不会重叠
    verts.extend([[x + vx, y + vy, z * brick_height + vz] for vx, vy, vz in base_verts])
    
    for i in range(w):
        for j in range(l):
            circle_verts = []
            n = 20
            for k in range(n):
                theta = 2 * np.pi * k / n
                circle_verts.append([x + i + 0.5 + stud_radius * np.cos(theta),
                                     y + j + 0.5 + stud_radius * np.sin(theta),
                                     z + brick_height])
                circle_verts.append([x + i + 0.5 + stud_radius * np.cos(theta),
                                     y + j + 0.5 + stud_radius * np.sin(theta),
                                     z + brick_height + stud_height])
            circle_faces = [[m, m + 1, (m + 3) % (2 * n), (m + 2) % (2 * n)] for m in range(0, 2 * n, 2)]
            verts.extend(circle_verts)
            faces.extend([[len(verts) - 2 * n + f[0], len(verts) - 2 * n + f[1], len(verts) - 2 * n + f[2], len(verts) - 2 * n + f[3]] for f in circle_faces])
    return verts, faces

# 定义乐高积木的条状结构
class Strip:
    def __init__(self, x, y):
        self.x0 = x
        self.y0 = y
        self.length = 1
        self.width = 1

    def set_length(self, l):
        self.length = l

    def merge_width(self):
        self.width = 2

    # 使用整数规划算法计算最优高度
    def LpIntH(self, l_r, l_list):
        prob = pulp.LpProblem("LEGO", pulp.LpMinimize)
        n = len(l_list)
        var_list = [pulp.LpVariable("x" + str(i), lowBound=0, cat=pulp.LpInteger) for i in range(n)]
        prob += pulp.lpSum(var_list[i] for i in range(n))
        prob += (pulp.lpSum(l_list[i] * var_list[i] for i in range(n)) == l_r)
        prob.solve()
        return pulp.value(prob.objective)
    
    # 计算条状结构与间隙的距离
    def BorderGapD(self, brick, gap_list):
        x_right = brick.x0 + brick.length
        d_xr = float('inf')
        for gap in gap_list:
            d = np.abs(x_right - gap)
            if d < d_xr:
                d_xr = d
        return d_xr
    
    # 计算条状结构在平面内的构建成本
    def cost_in_strip(self, l_r, brick, gap_list):
        w = brick.width
        l_list = brick_list[w - 1]
        L_max = max(l_list)
        c = 1
        if l_r > rho:
            r_1 = np.ceil((l_r - rho) / L_max) * L_max
            r_2 = l_r - r_1
            h_1 = r_1 / L_max
            h_2 = self.LpIntH(r_2, l_list)
        else:
            h_1 = 0
            h_2 = self.LpIntH(l_r, l_list)
        d_xr = alpha_1 * np.exp(- alpha_2 * self.BorderGapD(brick, gap_list))
        e = np.random.uniform(high=e_max)
        return (c + h_1 + h_2 + d_xr + e)

    # 构建条状结构
    def build_strip(self, gap_list):
        bricks = brick_list[self.width - 1]
        build_list = []
        next_gap_list = []
        l_r = self.length
        w = self.width
        x = self.x0
        y = self.y0
        while l_r > 0:
            c_xr = float('inf')
            for l in bricks:
                if l <= l_r:
                    X = Brick(x, y, l, w)
                    c = self.cost_in_strip(l_r - l, X, gap_list)
                    if c < c_xr:
                        c_xr = c
                        build_x = X
                        next_gap_x = X.x0 + X.length
            build_list.append(build_x)
            next_gap_list.append(next_gap_x)
            y = y + build_x.length
            l_r = l_r - build_x.length
        return build_list, next_gap_list

# 定义平面类，用于生成乐高积木构造平面
class Plane:
    def __init__(self, dir, z, matrix):
        self.dir = dir
        self.z0 = z
        self.matrix = matrix
        self.strip_list = []
        self.build_matrix = []

    # 对平面进行分割，生成条状结构
    def segment(self):
        if self.dir == 0:
            plane_matrix = self.matrix
        else:
            plane_matrix = self.matrix.T
        for i in range(plane_matrix.shape[0]):
            starter = 0
            for j in range(plane_matrix.shape[1]):
                if starter == 0:
                    if plane_matrix[i][j] == 1:
                        strip = Strip(i, j)
                        l = 1
                        starter = 1
                else:
                    if plane_matrix[i][j] == 1:
                        l = l + 1
                    else:
                        if l > 0:
                            strip.set_length(l)
                            self.strip_list.append(strip)
                        starter = 0
                        l = 0
    # 合并条状结构
    def merge(self):
        i = 0
        while i < len(self.strip_list):
            if self.strip_list[i].width == 1:
                x = self.strip_list[i].x0
                y = self.strip_list[i].y0
                l = self.strip_list[i].length
                for j in range(i + 1, len(self.strip_list)):
                    if (self.strip_list[j].x0 == x + 1 
                        and self.strip_list[j].y0 == y 
                        and self.strip_list[j].width == 1
                        and self.strip_list[j].length == l):
                        self.strip_list[i].merge_width()
                        self.strip_list.pop(j)
                        break
            i = i + 1

    # 构建平面，将条状结构组合成平面结构
    def build_plane(self):
        gap_list = [0, self.matrix.shape[1]]
        next_gap_list = [0, self.matrix.shape[1]]
        line_index = 0
        for strip in self.strip_list:
            if strip.x0 > line_index:
                gap_list = next_gap_list
                line_index = strip.x0
            build_list, next_gap_list = strip.build_strip(gap_list)
            self.build_matrix.append(build_list)

    # 绘制平面，将平面结构可视化为3D图像
    def draw_plane(self, step, ax, accumulated_bricks):
        for strip in self.build_matrix:
            for brick in strip:
                verts, faces = create_brick(brick.x0, brick.y0, self.z0, brick.length, brick.width)
                poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
                accumulated_bricks.append((verts, faces))

        for verts, faces in accumulated_bricks:
            poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
            poly = Poly3DCollection(poly3d, facecolors='white', linewidths=1, edgecolors='k', alpha=1)
            poly.set_sort_zpos(verts[0][2])  # 根据 Z 轴位置排序
            poly.set_alpha(None)  # 禁用透明度处理
            ax.add_collection3d(poly)

        ax.set_xlim([0, self.matrix.shape[0]])
        ax.set_ylim([0, self.matrix.shape[1]])
        ax.set_zlim([0, HEIGHT])
        ax.view_init(elev=45, azim=45)
        ax.text2D(0.05, 0.95, f'Step {step}', transform=ax.transAxes, fontsize=12, fontweight='bold')
        plt.axis('off')

# 生成乐高积木教程视图
@csrf_exempt
def generate_lego_tutorial(request):
    if request.method == 'POST':
        model_path = request.POST.get('model_path')

        if not model_path:
            return JsonResponse({'error': 'Model path is required'}, status=400)

        try:
            # 获取模型名称并创建输出目录
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            output_dir = os.path.join(settings.MEDIA_ROOT, 'tutorial', model_name)

            # 从会话中获取 voxel_matrix
            voxel_matrix = np.array(request.session.get('voxel_matrix', None))
            if voxel_matrix is None:
                return JsonResponse({'error': 'Voxel matrix not found in session'}, status=400)

            lego_matrix = voxel_matrix            
            
            # HEIGHT = lego_matrix.shape[0]

            step = 1
            accumulated_bricks = []
            instructions = []
            for z in range(lego_matrix.shape[0]):
                matrix = lego_matrix[:, :, z]
                if np.any(matrix):
                    # Rotate the matrix 180 degrees around the Z-axis
                    matrix = np.flip(matrix, axis=(0, 1))  # Flip along both X and Y axes
                    plane = Plane(0, z, matrix)
                    plane.segment()
                    plane.merge()
                    plane.build_plane()
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    plane.draw_plane(step, ax, accumulated_bricks)
                    step_image_path = os.path.join(output_dir, f'lego_step_{step}.png')
                    plt.savefig(step_image_path)
                    plt.close()
                    
                    # Generate instruction for the current step
                    instruction = generate_instruction(step_image_path, openai_api_key)
                    instructions.append({
                        'step': step,
                        'image_url': settings.MEDIA_URL + f'tutorial/{model_name}/lego_step_{step}.png',
                        'instruction': instruction
                    })
                    step += 1

            return JsonResponse({
                'message': 'LEGO tutorial generated successfully',
                'instructions': instructions
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)


# 生成每一步骤的积木拼装引导提示
def generate_instruction(image_path, openai_api_key):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # 将图片编码为Base64格式
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    # 构建API请求的载荷
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "你是一个家庭引导师，\
                            你的职责是帮助家长引导孩子并提升语言表达能力。你当前的任务是根据图片中乐高搭建教程的步骤，\
                            在界面中实时显示引导提示。例如‘问问孩子最上方的积木是什么？’或‘现在拼的是模型的哪一部分？’。\
                            步骤图会由底向上搭建，每次输出一句提示语。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    # 调用OpenAI API生成引导提示
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    assistant_message = response_data['choices'][0]['message']['content']
    return assistant_message

## 空间语言提问智能引导助手
@csrf_exempt
def lego_storytelling(request):
    # 处理GET请求，加载页面
    if request.method == 'GET':
        return render(request, 'prompt.html')
    
    # 处理POST请求，处理对话
    if request.method == 'POST':
       # 如果是处理用户与智能助理对话的请求
        user_input = request.POST.get('prompt', '').strip()
        
        # 检查用户输入是否为空
        if not user_input:
            return JsonResponse({'error': 'Invalid input data'}, status=400)
        
        # 从会话中获取当前的对话历史，如果没有则初始化一个新的对话历史
        conversation_history2 = request.session.get('conversation_history2', [
            {"role": "system", "content": "你是一个儿童的空间语言训练师，\
                                        你的职责是帮助家长引导孩子并提升空间语言表达能力。你当前的任务是根据孩子目前已搭建的乐高积木，\
                                        引导家长让孩子练习空间语言词汇，这些词汇包含形状（边、曲线、圆、线等）、位置和方向（描述相关位置）、大小关系、指示词（这里、那里、哪里等）。\
                                        请询问孩子有关该乐高积木涉及以上词汇的问题。"}
        ])

        # 如果用户输入表示结束对话
        if check_end_conversation(user_input):
            # 将用户输入添加到对话历史中
            conversation_history2.append({"role": "user", "content": user_input})
            
            # 调用GPT-4o-mini生成对话总结
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_history2 + [{"role": "user", "content": "请根据回答，请评估一下孩子的空间语言水平。"}],
            )
            story = completion.choices[0].message.content

            return JsonResponse({'message': story, 'done': True})
        
        else:
            # 如果对话未结束，将用户输入添加到对话历史中
            conversation_history2.append({"role": "user", "content": user_input})

            # 调用GPT-4o-mini继续对话
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_history2,
            )

            # 获取助手的回复并添加到对话历史中
            model_reply = completion.choices[0].message.content

            conversation_history2.append({"role": "assistant", "content": model_reply})
            
            # 更新会话中的对话历史
            request.session['conversation_history2'] = conversation_history2

            return JsonResponse({'message': model_reply, 'done': False})
    
    # 如果请求方法不是GET或POST，返回错误响应
    return JsonResponse({'error': 'Invalid request method'}, status=405)
