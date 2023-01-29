# DEEP_SORT 

> `DeepSORT (Deep Appearance-Based Single Object Tracking)` is an algorithm for tracking objects in video sequences.

## Applications :
DeepSORT is particularly useful in scenarios where objects are partially occluded or have different orientations, as it is able to maintain the count of objects even in these challenging conditions. It has been widely used in various applications such as surveillance, traffic monitoring, and sports analysis.

## So what is Deep Sort ?

DeepSORT (Deep Appearance-Based Single Object Tracking) is an algorithm that allows for tracking objects in video sequences. It combines object detection and tracking in one algorithm, making it efficient and accurate.

The algorithm uses a deep neural network for object detection, which generates bounding boxes around objects of interest. Then, a tracking algorithm is applied to match the objects across frames and keep count of unique objects in the video.

DeepSORT also uses an online metric learning technique, called the "Hungarian algorithm", to match detections across frames and maintain the count of unique objects even in the presence of partial occlusions or different orientations.

This algorithm is widely used in various applications such as surveillance, traffic monitoring, and sports analysis. Its ability to maintain count of objects in challenging conditions make it a powerful tool for these scenarios.
