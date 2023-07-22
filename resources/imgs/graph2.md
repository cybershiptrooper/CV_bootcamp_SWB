graph LR

subgraph CV_Domains
    A1(video)
    A2(multi modal/t<->im/v)
    A3(image -> rl policy)
    A4(3D Computer Vision)
    A5(Generation)
    A6(Inference on Images)
    A7(Robotics)
    A8(Everything with Very Less Data/No Labeling)
end

subgraph 3D
    A4a(3D Scene Synthesis)
    A4b(Image Stitching and Homography Operations)
    A4c(Knowledge Representation in Images)
    A7a(3D Object pose Estimation)
    A7b(SLAM - Simultaneous Localization and Mapping)
end

subgraph Generative Methods
    A5a(3D Scenes Synthesis/Reconstruction)
    A5b(Video Generation)
    A5c(Image-to-Image Translation)
    A5c1(Face Generation)
    A5c2(Super-Resolution)
    A5c3(Medical Image Processing)
    A5c3_1(Reconstruction: MRI, CT, PET, Radon Transforms)
    A5c3_2(Segmentation is Huge Here)
    A5c3_3(More Recently: Brain Scans using Stable Diffusion)
    A5c4(Style Transfer)
    A5c5(Satellite/Drone Imaging)
    A5c5_1(Cross View Synthesis)
    A5c5_2(Railway Lines Analysis)
    A5c5_3(Camouflage Detection)
end

subgraph Inference
    A6a(Pose Estimation)
    A6b(Semantic Segmentation)
    A6b1(Texture Segmentation - Supervised)
    A6c(Object Detection)
    A6d(Action Recognition)
end

A1 --> CV_Domains
A2 --> CV_Domains
A3 --> CV_Domains
A4 --> CV_Domains
A5 --> CV_Domains
A6 --> CV_Domains
A7 --> CV_Domains
A8 --> CV_Domains

A4a --> A4
A4b --> A4
A4c --> A4

A5a --> A5
A5b --> A5
A5c --> A5
A5c1 --> A5c
A5c2 --> A5c
A5c3 --> A5c
A5c3_1 --> A5c3
A5c3_2 --> A5c3
A5c3_3 --> A5c3
A5c4 --> A5c
A5c5 --> A5c
A5c5_1 --> A5c5
A5c5_2 --> A5c5
A5c5_3 --> A5c5

A6a --> A6
A6b --> A6
A6b1 --> A6b
A6c --> A6

A7a --> A7
A7b --> A7

CV_Domains --> A4
CV_Domains --> A5
CV_Domains --> A6
CV_Domains --> A7

A1 --> A5b
A1 --> A6d
%% A1 --> A6b
%% A1 --> A6c