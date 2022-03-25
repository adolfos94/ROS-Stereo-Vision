# Stereo Rectification

C++ codes for stereo camera calibration and rectification. The data contained in this example is based on the **IMX219-83 Stereo Camera**.

![IMX219-83 Stereo Images](/res/stereo_example_rect.png "IMX219-83 Stereo Camera")

## Build

1. Create build directory.

```command
    $ adolfo in ROS-Stereo-Camera/Stereo Calibration
    mkdir build
```

2. Navigate to build directory.

```command
    $ adolfo in ROS-Stereo-Camera/Stereo Calibration
    cd build
```

3. Call cmake and make.

```command
    $ adolfo in ROS-Stereo-Camera/Stereo Calibration/build
    cmake .. && make
```

## Run

### Store Calibration Images

- Run GetCalibrationImages program. It saves automatically all the stereo images in live mode.

```command
    adolfo in ROS-Stereo-Camera/Stereo Calibration/build
    ./GetCalibrationImages
```

### Extract Calibration Parameters.

- Run GetCalibrationMatrices program. It saves automatically all the stereo images params. The file it is saved as **"stereocalib.yml"**.

```command
    adolfo in ROS-Stereo-Camera/Stereo Calibration/build
    ./GetCalibrationMatrices
```

-

### Display Rectified Images.

- Run GetRectifiedImages program. It displays the stereo rectified images using **"stereocalib.yml"**.\
  <span style="color:orange"> If an argument is passed it will use the default images saved.</span>

```command
    adolfo in ROS-Stereo-Camera/Stereo Calibration/build
    ./GetRectifiedImages
```
