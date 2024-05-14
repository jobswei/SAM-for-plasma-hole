

filename="new"
# 输入视频文件路径和文件名
input_video = f"outputs/videos/{filename}.avi"

# 输出视频文件路径和文件名
output_video = f'outputs/videos/{filename}.mp4'

from moviepy.editor import VideoFileClip



# 使用moviepy进行转换
video = VideoFileClip(input_video)
video.write_videofile(output_video, codec='libx264')
