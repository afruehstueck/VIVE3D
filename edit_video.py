import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import click
from datetime import datetime
import pickle
    
from vive3D.visualizer import *
from vive3D.eg3d_generator import *
from vive3D.landmark_detector import *
from vive3D.video_tool import *
from vive3D.segmenter import *
from vive3D.inset_pipeline import *
from vive3D.aligner import *
from vive3D.interfaceGAN_editor import *
from vive3D.config import *

@click.command()
@click.option('-s', '--savepoint_path', type=str, help='Savepoint directory', required=True)
@click.option('-v', '--source_video', type=str, help='Path to source video', required=True)
@click.option('--start_sec', type=int, default=0)
@click.option('--end_sec', type=int, default=0)
@click.option('--resize_video', type=int, default=1)
@click.option('--focal_length', type=float, help='Generator Focal Length', default=3.6)
@click.option('--camera_position', type=(float, float, float), nargs=3, help='Generator Camera Position', default=(0, 0.05, 0.2))
@click.option('--edit_type', type=str, help='Editing type', default='o')
@click.option('--edit_strength', type=float, help='Editing direction alpha', default=1.0)
@click.option('--border', type=int, help='Border size for inset boundary', default=50)
@click.option('--yaw', type=float, help='Target yaw angle', default=None)
@click.option('--pitch', type=float, help='Target pitch angle', default=None)
@click.option('-d', '--device', type=str, help='GPU device that should be used.', default='cuda')
@click.option('--loss_threshold', type=float, default=0.5, help='Early stopping threshold for inversion. Empirically selected per video.')


# @click.option('--resize_video', type=int, default=1)
# @click.option('--focal_length', type=float, help='Generator Focal Length', default=3.6)
# @click.option('--camera_position', type=(float, float, float), nargs=3, help='Generator Camera Position', default=(0, 0.05, 0.2))
# 

def main(**config):
    _main(**config)


def _main(savepoint_path,
          source_video, 
          start_sec, 
          end_sec, 
          resize_video,
          focal_length,
          camera_position,
          edit_type,
          edit_strength,
          border,
          yaw,
          pitch,
          loss_threshold,
          device):
    
    device = torch.device(device)
    assert os.path.exists(savepoint_path), f'Savepoint folder does not exist.'
    
    print(f'*******************************************************************************')
    print(f'Loading personalized generator from {savepoint_path}/G_tune.pkl')
    tuned_generator_path = f'{savepoint_path}/G_tune.pkl'
    assert os.path.exists(tuned_generator_path), f'Generator is not available at {tuned_generator_path}, please check savepoint_path'
    generator = EG3D_Generator(tuned_generator_path, device, load_tuned=True)
    generator.set_camera_parameters(focal_length=focal_length, cam_pivot=camera_position)
    
    print(f'*******************************************************************************')
    print(f'Loading video and inversion....')
    video_output_path = os.getcwd()+f'/video/{savepoint_path.split("/")[-1]}'
    os.makedirs(video_output_path, exist_ok=True)
    
    # create video tool instance for target video
    frames_path = f'{config.PROJECT}/frames'
    vid = VideoTool(source_video, frames_path)
    
    
    offsets_path = f'{savepoint_path}/inversion_{vid.get_video_title()}_{start_sec}-{end_sec}_w_offsets.pt'
    angles_path = f'{savepoint_path}/inversion_{vid.get_video_title()}_{start_sec}-{end_sec}_angles.pt'
    assert os.path.exists(offsets_path) and os.path.exists(angles_path), f'Offsets and angles do not exist in savepoint folder. Run invert_video.py first.'
    w_offsets_video = torch.load(offsets_path).to(device)
    angles = torch.load(angles_path)
    yaws_video = angles[:, 0].to(device)
    pitches_video = angles[:, 1].to(device)    
    
    
    print(f'*******************************************************************************')
    print(f'Applying edit {edit_type} with strength {"+" if edit_strength>0 else "-"}{abs(edit_strength)} to face')
    editor = Editor(device=device)
    
    edit = [(edit_strength, edit_type)]
    edit_description = ''
    for e in edit:
        if e[0] != 0:
            edit_description += f'_{e[1]}{"+" if e[0]>0 else "-"}{abs(e[0])}'

    w_person = torch.load(f'{savepoint_path}/inversion_w_person.pt').to(device)
    
    w_default = editor.edit(w_person, 'default', 0.0, w_offsets_video)
    w_modify = editor.multi_edit(w_person, edit, w_offsets_video)
    face_edited = generator.generate(w_modify, yaws_video, pitches_video)
    #vid.write_frames_to_video(tensor_to_image(face_edited), f'{video_output_path}/{vid.get_video_title()}_{start_sec}-{end_sec}{edit_description}_face')

    use_video_flow = (yaw is not None) or (pitch is not None)        
    
    if yaw == None:
        ys = yaws_video
        yaw = 'o'
    else:
        ys = yaw*torch.ones_like(yaws_video).to(device)
            
    if pitch == None:
        ps = pitches_video
        pitch = 'o'
    else:
        ps = pitch*torch.ones_like(pitches_video).to(device)
    
    print(f'*******************************************************************************')
    print(f'Optimizing inset...')
    
    params = dict(edge_size_y_top = 8,
                  border_size = border,
                  edge_size = (90, 40),
                  face_dilate = 0,
                  original_yaws_video = yaws_video,
                  original_pitches_video = pitches_video,
                  use_flow = use_video_flow,
                  use_w_dist = True,
                  include_neck = (yaw!='o' and abs(yaw)>0.15),
                  border_loss_threshold = loss_threshold, #pick an appropriate threshold empirically based on video
                  return_flow_directions = False,
                  plot_progress = False)
        
    # additionally required tools
    segmenter = Segmenter(device=device)
    landmark_detector = LandmarkDetector(device=device)
    align = Aligner(landmark_detector=landmark_detector, segmenter=segmenter, device=device)
    pipeline = Pipeline(generator, segmenter, align, device=device)
    
    frames_video = vid.extract_frames_from_video(start_sec, end_sec, resize=resize_video)
    w_offsets = torch.load(f'{savepoint_path}/inversion_w_offsets.pt').to(device)
    reference_neutral_face = generator.generate(w_person, 0.0, -0.1)
    reference_face_landmarks = landmark_detector.get_landmarks(tensor_to_image(reference_neutral_face), get_all=False)

    face_tensors_video, segmentation_tensors_video, landmarks_video = align.get_face_tensors_from_frames(frames_video, reference_face=reference_neutral_face, smooth_landmarks=True)

    output = pipeline.inset_video(frames_video, face_tensors_video, landmarks_video, reference_face_landmarks, w_modify, w_default, ys, ps, **params)
    output_inset, output_faces, _, _, _ = output
    
    angle_prefix = ''
    angle_prefix += f'_y={yaw}' if yaw != 'o' else ''
    angle_prefix += f'_p={pitch}' if pitch != 'o' else ''
    fname = f'{vid.get_video_title()}_inset{angle_prefix}{boundary_str}'
    vid.write_frames_to_video(output_inset, f'{video_output_path}/{fname}')

#     tuned_generator_path = f'{savepoint_path}/G_tune.pkl'
#     assert os.path.exists(tuned_generator_path), f'Generator is not available at {tuned_generator_path}, please check savepoint_path'
#     generator = EG3D_Generator(tuned_generator_path, device, load_tuned=True)
#     generator.set_camera_parameters(focal_length=focal_length, cam_pivot=camera_position)
    
#     print(f'*******************************************************************************')
#     print(f'Loading video {source_video.split("/")[-1]} from secs {start_sec}-{end_sec} and cropping faces')
    
#     # additionally required tools
#     segmenter = Segmenter(device=device)
#     landmark_detector = LandmarkDetector(device=device)
#     align = Aligner(landmark_detector=landmark_detector, segmenter=segmenter, device=device)
    
#     frames_video = vid.extract_frames_from_video(start_sec, end_sec, resize=resize_video)
    
#     w_person = torch.load(f'{savepoint_path}/inversion_w_person.pt').to(device)
#     w_offsets = torch.load(f'{savepoint_path}/inversion_w_offsets.pt').to(device)
#     reference_neutral_face = generator.generate(w_person, 0.0, -0.1)

#     face_tensors_video, segmentation_tensors_video, landmarks_video = align.get_face_tensors_from_frames(frames_video, reference_face=reference_neutral_face, smooth_landmarks=True)
        
#     vid.write_frames_to_video(frames_video, f'{video_output_path}/{vid.get_video_title()}_{start_sec}-{end_sec}_source')
#     vid.write_frames_to_video(tensor_to_image(face_tensors_video), f'{video_output_path}/{vid.get_video_title()}_{start_sec}-{end_sec}_source_face')

    
#     print(f'*******************************************************************************')
#     print(f'Invert video sequence...')
    
    
#     # create pipeline instance
#     pipeline = Pipeline(generator, segmenter, align, device=device)
    
#     selected_face_tensors = torch.load(f'{savepoint_path}/selected_face_tensors.pt').to(device)
#     faces_accum_segmentation = segmenter.get_eyes_mouth_BiSeNet(selected_face_tensors.to(device), dilate=8).any(dim=0)

#     w_offsets_video, yaws_video, pitches_video = pipeline.inversion_video(w_person, w_offsets, face_tensors_video, face_segmentation=faces_accum_segmentation, loss_threshold=loss_threshold, plot_progress=False)
    
#     torch.save(w_offsets_video.cpu(), f'{savepoint_path}/inversion_{vid.get_video_title()}_w_offsets.pt')
#     torch.save(torch.tensor(list(zip(yaws_video, pitches_video))).cpu(), f'{savepoint_path}/inversion_{vid.get_video_title()}_angles.pt')        
    

if __name__ == '__main__':
    main()