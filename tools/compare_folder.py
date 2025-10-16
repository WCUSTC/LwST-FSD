import os
import shutil

def compare_and_copy_different_files(query_folder, target_folder, output_query_folder, output_target_folder):
    # 遍历query_folder中的所有文件和文件夹
    for root, dirs, files in os.walk(query_folder):
        for file in files:
            if file.endswith('.py'):
                # 构建query_folder中文件的完整路径
                query_file_path = os.path.join(root, file)
                # 构建target_folder中对应文件的完整路径
                relative_path = os.path.relpath(query_file_path, query_folder)
                target_file_path = os.path.join(target_folder, relative_path)

                # 检查target_folder中是否存在对应文件
                if os.path.exists(target_file_path):
                    # 读取两个文件的内容
                    with open(query_file_path, 'r', encoding='utf-8') as f1, open(target_file_path, 'r', encoding='utf-8') as f2:
                        query_content = f1.read()
                        target_content = f2.read()

                    # 比较文件内容
                    if query_content != target_content:
                        # 构建输出文件夹中对应文件的完整路径
                        output_query_file_path = os.path.join(output_query_folder, relative_path)
                        output_target_file_path = os.path.join(output_target_folder, relative_path)

                        # 创建输出文件夹的目录结构
                        os.makedirs(os.path.dirname(output_query_file_path), exist_ok=True)
                        os.makedirs(os.path.dirname(output_target_file_path), exist_ok=True)

                        # 复制不同内容的文件到输出文件夹
                        shutil.copy2(query_file_path, output_query_file_path)
                        shutil.copy2(target_file_path, output_target_file_path)
                else:
                    # 如果target_folder中不存在对应文件，将query_folder中的文件复制到输出文件夹
                    output_query_file_path = os.path.join(output_query_folder, relative_path)
                    os.makedirs(os.path.dirname(output_query_file_path), exist_ok=True)
                    shutil.copy2(query_file_path, output_query_file_path)

    # 遍历target_folder中的所有文件和文件夹，找出query_folder中不存在的文件
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if file.endswith('.py'):
                target_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(target_file_path, target_folder)
                query_file_path = os.path.join(query_folder, relative_path)

                if not os.path.exists(query_file_path):
                    output_target_file_path = os.path.join(output_target_folder, relative_path)
                    os.makedirs(os.path.dirname(output_target_file_path), exist_ok=True)
                    shutil.copy2(target_file_path, output_target_file_path)

if __name__ == "__main__":
    query_folder = r"E:\mmdetection320\models"
    target_folder = r"E:\mmdetection320\mmdet\models"
    output_query_folder = r"E:\mmdetection320\diff_files_models\query"
    output_target_folder = r"E:\mmdetection320\diff_files_models\target"

    compare_and_copy_different_files(query_folder, target_folder, output_query_folder, output_target_folder)