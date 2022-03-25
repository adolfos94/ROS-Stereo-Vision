# Stereo Rectification

C++ codes for stereo camera calibration and rectification. The data contained in this example is based on the **IMX219-83 Stereo Camera**.

## Build

1. Create build directory.

```command
    mkdir build
```

2. Navigate to build directory.

```command
    cd build
```

3. Call cmake and make.

```command
    cmake .. && make
```

## Run

### Store Calibration Images

- Run GetCalibrationImages program. It saves automatically all the stereo images in live mode.

```command
    ./GetCalibrationImages
```

### Extract Calibration Parameters.

- Run GetCalibrationMatrices program. It saves automatically all the stereo images params.

```command
    ./GetCalibrationMatrices
```

- The file it is saved as **"stereocalib.yml"**.

### Display Rectified Images.

- Run GetRectifiedImages program. It displays the stereo rectified images using **"stereocalib.yml"**.\
  <span style="color:orange"> If an argument is passed it will use the default images saved.</span>

```command
    ./GetRectifiedImages
```
