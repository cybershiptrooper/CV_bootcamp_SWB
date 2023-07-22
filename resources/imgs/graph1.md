graph LR

subgraph CV Domains

    video
    multi_modal_t_t_im_v
    image_rl_policy
    3D
    synthesis
    image_stitching_homography_ops
    knowledge_representation
    generation
    _3d_scenes_synthesis_reconstruction
    video_generation
    pix_to_pix_stuff
    faces
    super_resolution
    medical_image_proc
    brain_scans_using_stable_diffusion
    style_transfer
    satellite_drone_imaging
    cross_view_synthesis
    railway_lines
    camoflauge
    inference_on_images
    pose_estimation
    semantic_segment
    texture_segment_supervised
    object_detection
    robotics
    _3d_objects
    SLAM
    everything_with_less_data_no_labelling

end

video --> multi_modal_t_t_im_v
multi_modal_t_t_im_v --> image_rl_policy
image_rl_policy --> 3D
3D --> synthesis
3D --> image_stitching_homography_ops
3D --> knowledge_representation
generation --> _3d_scenes_synthesis_reconstruction
generation --> video_generation
generation --> pix_to_pix_stuff
pix_to_pix_stuff --> faces
pix_to_pix_stuff --> super_resolution
pix_to_pix_stuff --> medical_image_proc
medical_image_proc --> brain_scans_using_stable_diffusion
medical_image_proc --> semantic_segment
medical_image_proc --> texture_segment_supervised
pix_to_pix_stuff --> style_transfer
pix_to_pix_stuff --> satellite_drone_imaging
satellite_drone_imaging --> cross_view_synthesis
satellite_drone_imaging --> railway_lines
satellite_drone_imaging --> camoflauge
inference_on_images --> pose_estimation
inference_on_images --> semantic_segment
inference_on_images --> texture_segment_supervised
inference_on_images --> object_detection
robotics --> _3d_objects
robotics --> SLAM

knowledge_representation -.->|works in image too| image_rl_policy
brain_scans_using_stable_diffusion -.->|More recently| medical_image_proc
_3d_scenes_synthesis_reconstruction -.->|MRI, CT, PET, radon transforms| medical_image_proc
everything_with_less_data_no_labelling -->|Segment:| semantic_segment
everything_with_less_data_no_labelling -->|Texture, supervised| texture_segment_supervised
