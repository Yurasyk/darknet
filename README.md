# Oriented Yolo-v3 for Windows and Linux
### (neural network for object detection and 1D orientation estimation)

1. [How to use](#how-to-use)
2. [How to compile on Linux](#how-to-compile-on-linux)
3. [How to compile on Windows](#how-to-compile-on-windows)
4. [How to train (Pascal VOC Data)](#how-to-train-pascal-voc-data)
5. [How to train (to detect your custom objects)](#how-to-train-to-detect-your-custom-objects)
6. [When should I stop training](#when-should-i-stop-training)
7. [How to calculate mAP on PascalVOC 2007](#how-to-calculate-map-on-pascalvoc-2007)
8. [How to improve object detection](#how-to-improve-object-detection)
9. [How to mark bounded boxes of objects and create annotation files](#how-to-mark-bounded-boxes-of-objects-and-create-annotation-files)
10. [Using Yolo9000](#using-yolo9000)
11. [How to use Yolo as DLL](#how-to-use-yolo-as-dll)



|  ![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png) | &nbsp; ![map_fps](https://hsto.org/webt/pw/zd/0j/pwzd0jb9g7znt_dbsyw9qzbnvti.jpeg) mAP (AP50) https://pjreddie.com/media/files/papers/YOLOv3.pdf |
|---|---|

* Yolo v3 source chart for the RetinaNet on MS COCO got from Table 1 (e): https://arxiv.org/pdf/1708.02002.pdf
* Yolo v2 on Pascal VOC 2007: https://hsto.org/files/a24/21e/068/a2421e0689fb43f08584de9d44c2215f.jpg
* Yolo v2 on Pascal VOC 2012 (comp4): https://hsto.org/files/3a6/fdf/b53/3a6fdfb533f34cee9b52bdd9bb0b19d9.jpg


# "You Only Look Once: Unified, Real-Time Object Detection (versions 2 & 3)"
A Yolo cross-platform Windows and Linux version (for object detection). Contributtors: https://github.com/pjreddie/darknet/graphs/contributors

This repository is forked from Linux-version: https://github.com/pjreddie/darknet

More details: http://pjreddie.com/darknet/yolo/

This repository supports:

* both Windows and Linux
* both OpenCV 2.x.x and OpenCV <= 3.4.0 (3.4.1 and higher isn't supported)
* both cuDNN v5-v7
* CUDA >= 7.5
* also create SO-library on Linux and DLL-library on Windows

##### Requires: 
* **Linux GCC>=4.9 or Windows MS Visual Studio 2015 (v140)**: https://go.microsoft.com/fwlink/?LinkId=532606&clcid=0x409  (or offline [ISO image](https://go.microsoft.com/fwlink/?LinkId=615448&clcid=0x409))
* **CUDA 9.1**: https://developer.nvidia.com/cuda-downloads
* **OpenCV 3.4.0**: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.0/opencv-3.4.0-vc14_vc15.exe/download
* **or OpenCV 2.4.13**: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.13/opencv-2.4.13.2-vc14.exe/download
  - OpenCV allows to show image or video detection in the window and store result to file that specified in command line `-out_filename res.avi`
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported

### How to use:

##### How to use on the command line:

On Linux use `./darknet` instead of `darknet.exe`, like this:`./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights`

* **Yolo v3** on your custom objets: `darknet.exe detector test data/obj.data cfg/yolo-obj.cfg yolo-obj.weights -i 0 -thresh 0.25` 

(only after modification of yolo-obj.cfg as stated below and obtaining of the trained model with weights yolo-obj.weights)

### How to compile on Linux:

Just do `make` in the darknet directory.
Before make, you can set such options in the `Makefile`: [link](https://github.com/AlexeyAB/darknet/blob/9c1b9a2cf6363546c152251be578a21f3c3caec6/Makefile#L1)
* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
* `OPENCV=1` to build with OpenCV 3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* `DEBUG=1` to bould debug version of Yolo
* `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
* `LIBSO=1` to build a library `darknet.so` and binary runable file `uselib` that uses this library. Or you can try to run so `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4` How to use this SO-library from your own code - you can look at C++ example: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp


### How to compile on Windows:

1. If you have **MSVS 2015, CUDA 9.1, cuDNN 7.0 and OpenCV 3.x** (with paths: `C:\opencv_3.0\opencv\build\include` & `C:\opencv_3.0\opencv\build\x64\vc14\lib`), then start MSVS, open `build\darknet\darknet.sln`, set **x64** and **Release**, and do the: Build -> Build darknet. **NOTE:** If installing OpenCV, use OpenCV 3.4.0 or earlier. This is a bug in OpenCV 3.4.1 in the C API (see [#500](https://github.com/AlexeyAB/darknet/issues/500)).

    1.1. Find files `opencv_world320.dll` and `opencv_ffmpeg320_64.dll` (or `opencv_world340.dll` and `opencv_ffmpeg340_64.dll`) in `C:\opencv_3.0\opencv\build\x64\vc14\bin` and put it near with `darknet.exe`
    
    1.2 Check that there are `bin` and `include` folders in the `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1` if aren't, then copy them to this folder from the path where is CUDA installed
    
    1.3. To install CUDNN (speedup neural network), do the following:
      
    * download and install **cuDNN 7.0 for CUDA 9.1**: https://developer.nvidia.com/cudnn
      
    * add Windows system variable `cudnn` with path to CUDNN: https://hsto.org/files/a49/3dc/fc4/a493dcfc4bd34a1295fd15e0e2e01f26.jpg
    
    1.4. If you want to build **without CUDNN** then: open `\darknet.sln` -> (right click on project) -> properties  -> C/C++ -> Preprocessor -> Preprocessor Definitions, and remove this: `CUDNN;`

2. If you have other version of **CUDA (not 9.1)** then open `build\darknet\darknet.vcxproj` by using Notepad, find 2 places with "CUDA 9.1" and change it to your CUDA-version, then do step 1

3. If you **don't have GPU**, but have **MSVS 2015 and OpenCV 3.0** (with paths: `C:\opencv_3.0\opencv\build\include` & `C:\opencv_3.0\opencv\build\x64\vc14\lib`), then start MSVS, open `build\darknet\darknet_no_gpu.sln`, set **x64** and **Release**, and do the: Build -> Build darknet_no_gpu

4. If you have **OpenCV 2.4.13** instead of 3.0 then you should change pathes after `\darknet.sln` is opened

    4.1 (right click on project) -> properties  -> C/C++ -> General -> Additional Include Directories:  `C:\opencv_2.4.13\opencv\build\include`
  
    4.2 (right click on project) -> properties  -> Linker -> General -> Additional Library Directories: `C:\opencv_2.4.13\opencv\build\x64\vc14\lib`
    
5. If you have GPU with Tensor Cores (nVidia Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x:
    `\darknet.sln` -> (right click on project) -> properties -> C/C++ -> Preprocessor -> Preprocessor Definitions, and add here: `CUDNN_HALF;`

### How to compile (custom):

Also, you can to create your own `darknet.sln` & `darknet.vcxproj`, this example for CUDA 9.1 and OpenCV 3.0

Then add to your created project:
- (right click on project) -> properties  -> C/C++ -> General -> Additional Include Directories, put here: 

`C:\opencv_3.0\opencv\build\include;..\..\3rdparty\include;%(AdditionalIncludeDirectories);$(CudaToolkitIncludeDir);$(cudnn)\include`
- (right click on project) -> Build dependecies -> Build Customizations -> set check on CUDA 9.1 or what version you have - for example as here: http://devblogs.nvidia.com/parallelforall/wp-content/uploads/2015/01/VS2013-R-5.jpg
- add to project all `.c` & `.cu` files and file `http_stream.cpp` from `\src`
- (right click on project) -> properties  -> Linker -> General -> Additional Library Directories, put here: 

`C:\opencv_3.0\opencv\build\x64\vc14\lib;$(CUDA_PATH)lib\$(PlatformName);$(cudnn)\lib\x64;%(AdditionalLibraryDirectories)`
-  (right click on project) -> properties  -> Linker -> Input -> Additional dependecies, put here: 

`..\..\3rdparty\lib\x64\pthreadVC2.lib;cublas.lib;curand.lib;cudart.lib;cudnn.lib;%(AdditionalDependencies)`
- (right click on project) -> properties -> C/C++ -> Preprocessor -> Preprocessor Definitions

`OPENCV;_TIMESPEC_DEFINED;_CRT_SECURE_NO_WARNINGS;_CRT_RAND_S;WIN32;NDEBUG;_CONSOLE;_LIB;%(PreprocessorDefinitions)`

- compile to .exe (X64 & Release) and put .dll-s near with .exe:

    * `pthreadVC2.dll, pthreadGC2.dll` from \3rdparty\dll\x64

    * `cusolver64_91.dll, curand64_91.dll, cudart64_91.dll, cublas64_91.dll` - 91 for CUDA 9.1 or your version, from C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\bin

    * For OpenCV 3.2: `opencv_world320.dll` and `opencv_ffmpeg320_64.dll` from `C:\opencv_3.0\opencv\build\x64\vc14\bin` 
    * For OpenCV 2.4.13: `opencv_core2413.dll`, `opencv_highgui2413.dll` and `opencv_ffmpeg2413_64.dll` from  `C:\opencv_2.4.13\opencv\build\x64\vc14\bin`



## How to train (to detect your custom objects):


Training Oriented Yolo v3:

1. Create file `yolo-obj.cfg` with the same content as in `yolov3.cfg` (or copy `yolov3.cfg` to `yolo-obj.cfg)` and:

  * change line batch to [`batch=64`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L3)
  * change line subdivisions to [`subdivisions=8`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)
  * change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers:
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L610
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L696
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L783
  * change line `random=1` to `random=0`:
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L615
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L701
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L788
  * change [`filters=255`] to filters=(classes + 7)x3 in the 3 `[convolutional]` before each `[yolo]` layer
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L603
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L689
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L776

  So if `classes=1` then should be `filters=24`. If `classes=2` then write `filters=27`.
  
  **(Do not write in the cfg-file: filters=(classes + 5)x3)**
  
  (Generally `filters` depends on the `classes`, `coords` and number of `mask`s, i.e. filters=`(classes + coords + 1)*<number of mask>`, where `mask` is indices of anchors. If `mask` is absence, then filters=`(classes + coords + 1)*num`)

2. Create file `obj.names` in the directory `build\darknet\x64\data\`, with objects names - each in new line

3. Create file `obj.data` in the directory `build\darknet\x64\data\`, containing (where **classes = number of objects**):

  ```
  classes= 2
  train  = data/train.txt
  valid  = data/test.txt
  names = data/obj.names
  backup = backup/
  ```

4. Put image-files (.jpg) of your objects in the directory `build\darknet\x64\data\obj\`

5. You should label each object on images from your dataset. Use this visual GUI-software for marking bounded boxes of objects and generating annotation files for Yolo v2 & v3: https://github.com/AlexeyAB/Yolo_mark

It will create `.txt`-file for each `.jpg`-image-file - in the same directory and with the same name, but with `.txt`-extension, and put to file: object number and object coordinates on this image, for each object in new line: `<object-class> <x> <y> <width> <height>`

  Where: 
  * `<object-class>` - integer number of object from `0` to `(classes-1)`
  * `<x> <y> <width> <height>` - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
  * for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
  * atention: `<x> <y>` - are center of rectangle (are not top-left corner)

  For example for `img1.jpg` you will be created `img1.txt` containing:

  ```
  1 0.716797 0.395833 0.216406 0.147222
  0 0.687109 0.379167 0.255469 0.158333
  1 0.420312 0.395833 0.140625 0.166667
  ```
Then use this modified GUI https://github.com/Yurasyk/Yolo_mark for marking orientations of each object. It will modify previously created files with orientational information.

  For example if will modify `img1.txt` to the next view:

  ```
  1 0.716797 0.395833 0.216406 0.147222 -0.356167 -0.246199
  0 0.687109 0.379167 0.255469 0.158333 0.760006 0.584722
  1 0.420312 0.395833 0.140625 0.166667 0.812397 0.745797 
  ```

6. Create file `train.txt` in directory `build\darknet\x64\data\`, with filenames of your images, each filename in new line, with path relative to `darknet.exe`, for example containing:

  ```
  data/obj/img1.jpg
  data/obj/img2.jpg
  data/obj/img3.jpg
  ```

7. Download pre-trained weights for the convolutional layers (154 MB): https://pjreddie.com/media/files/darknet53.conv.74 and put to the directory `build\darknet\x64`

8. Start training by using the command line: `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74`

    (file `yolo-obj_xxx.weights` will be saved to the `build\darknet\x64\backup\` for each 100 iterations)
    (To disable Loss-Window use `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show`, if you train on computer without monitor like a cloud Amazaon EC2)

9. After training is complete - get result `yolo-obj_final.weights` from path `build\darknet\x64\backup\`

 * After each 100 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just copy `yolo-obj_2000.weights` from `build\darknet\x64\backup\` to `build\darknet\x64\` and start training using: `darknet.exe detector train data/obj.data yolo-obj.cfg yolo-obj_2000.weights`

    (in the original repository https://github.com/pjreddie/darknet the weights-file is saved only once every 10 000 iterations `if(iterations > 1000)`)

 * Also you can get result earlier than all 45000 iterations.
 
 **Note:** If during training you see `nan` values for `avg` (loss) field - then training goes wrong, but if `nan` is in some other lines - then training goes well.
 
### How to train tiny-yolo (to detect your custom objects):

Do all the same steps as for the full yolo model as described above. With the exception of:
* Download default weights file for yolov3-tiny: https://pjreddie.com/media/files/yolov3-tiny.weights
* Get pre-trained weights `yolov3-tiny.conv.15` using command: `darknet.exe partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15`
* Make your custom model `yolov3-tiny-obj.cfg` based on `cfg/yolov3-tiny_obj.cfg` instead of `yolov3.cfg`
* Start training: `darknet.exe detector train data/obj.data yolov3-tiny-obj.cfg yolov3-tiny.conv.15`
 
## When should I stop training:

Usually sufficient 2000 iterations for each class(object). But for a more precise definition when you should stop training, use the following manual:

1. During training, you will see varying indicators of error, and you should stop when no longer decreases **0.XXXXXXX avg**:

  > Region Avg IOU: 0.798363, Class: 0.893232, Obj: 0.700808, No Obj: 0.004567, Avg Recall: 1.000000,  count: 8
  > Region Avg IOU: 0.800677, Class: 0.892181, Obj: 0.701590, No Obj: 0.004574, Avg Recall: 1.000000,  count: 8
  >
  > **9002**: 0.211667, **0.060730 avg**, 0.001000 rate, 3.868000 seconds, 576128 images
  > Loaded: 0.000000 seconds

  * **9002** - iteration number (number of batch)
  * **0.060730 avg** - average loss (error) - **the lower, the better**

  When you see that average loss **0.xxxxxx avg** no longer decreases at many iterations then you should stop training.

2. Once training is stopped, you should take some of last `.weights`-files from `darknet\build\darknet\x64\backup` and choose the best of them:

For example, you stopped training after 9000 iterations, but the best result can give one of previous weights (7000, 8000, 9000). It can happen due to overfitting. **Overfitting** - is case when you can detect objects on images from training-dataset, but can't detect objects on any others images. You should get weights from **Early Stopping Point**:

![Overfitting](https://hsto.org/files/5dc/7ae/7fa/5dc7ae7fad9d4e3eb3a484c58bfc1ff5.png) 

To get weights from Early Stopping Point:

  2.1. At first, in your file `obj.data` you must specify the path to the validation dataset `valid = valid.txt` (format of `valid.txt` as in `train.txt`), and if you haven't validation images, just copy `data\train.txt` to `data\valid.txt`.

  2.2 If training is stopped after 9000 iterations, to validate some of previous weights use this commands:

(If you use another GitHub repository, then use `darknet.exe detector recall`... instead of `darknet.exe detector map`...)

* `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
* `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_8000.weights`
* `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_9000.weights`

And comapre last output lines for each weights (7000, 8000, 9000):

Choose weights-file **with the highest IoU** (intersect of union) and mAP (mean average precision)

For example, **bigger IOU** gives weights `yolo-obj_8000.weights` - then **use this weights for detection**.

Example of custom object detection: `darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`

* **IoU** (intersect of union) - average instersect of union of objects and detections for a certain threshold = 0.24

* **mAP** (mean average precision) - mean value of `average precisions` for each class, where `average precision` is average value of 11 points on PR-curve for each possible threshold (each probability of detection) for the same class (Precision-Recall in terms of PascalVOC, where Precision=TP/(TP+FP) and Recall=TP/(TP+FN) ), page-11: http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

**mAP** is default metric of precision in the PascalVOC competition, **this is the same as AP50** metric in the MS COCO competition.
In terms of Wiki, indicators Precision and Recall have a slightly different meaning than in the PascalVOC competition, but **IoU always has the same meaning**.

![precision_recall_iou](https://hsto.org/files/ca8/866/d76/ca8866d76fb840228940dbf442a7f06a.jpg)


### Custom object detection:

Example of custom object detection: `darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`
