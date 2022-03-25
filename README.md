# ROS Stereo Camera

Cameras provide image data to the robot that can be used for object identification, tracking and manipulation tasks. This is a ROS Package for Jetson CSI Stereo Camera for Computer Vision Tasks.

### This package contains:

- **Stereo Camera Publisher:** Obtains left and right images from **IMX219-83 Stereo Camera**.
- **Stereo Camera Suscriber:** Displays the images using OpenCV.

# Hardware Stereo Module

![IMX219-83 Stereo Camera](/res/IMX219-83-Stereo-Camera-1.jpg "IMX219-83 Stereo Camera")

Waveshare Documentation: [IMX219-83 Stereo Camera](https://www.waveshare.com/wiki/IMX219-83_Stereo_Camera).

## Test with Jetson Nano

### Hardware Connection

- Connect the camera to the CSI interfaces of Jetson Nano.
- Connect an HDMI LCD to Jetson Nano.
- Connect the I2C interface (only the **SDA** and **SCL** pins are required) of the Camera to I2C interface of the **Jetson Nano Developer Kit** (the **Pin3**, and **Pin5**).

### Sofware Testing

- Open a Terminal.
- Check the video devices with command:

  ```console
  ls /dev/video*
  ```

- Check if both video0 and video1 are detected

  - video0

  ```console
  DISPLAY=:0.0 gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)20/1' ! nvoverlaysink -e
  ```

  - video1

  ```console
  DISPLAY=:0.0 gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)20/1' ! nvoverlaysink -e
  ```

- Test the **ICM20948**

  ```console
  wget https://www.waveshare.com/w/upload/a/a4/D219-9dof.tar.gz
  tar zxvf D219-9dof.tar.gz]
  cd D219-9dof/03-double-camera-display
  mkdir build
  cd build
  cmake ..
  make
  ./double-camera-display
  ```

- <span style="color:red"> If you find that the image captured is red. You can try to download .isp file and install it:</span>

  ```console
  wget https://www.waveshare.com/w/upload/e/eb/Camera_overrides.tar.gz
  tar zxvf Camera_overrides.tar.gz
  sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/
  sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
  sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
  ```

# Installation

## Requirements

- JetPack 4.6.1
- OpenCV (Installed with JetPack)
  - **Optional**
    1. Change the file
    ```command
    nano /opt/ros/melodic/share/cv_bridge/cmake/cv_bridgeConfig.cmake
    ```
    2. Change these lines:
    ```command
    set(_include_dirs "include;/usr/include;/usr/include/opencv")
    ```
    to:
    ```command
    set(_include_dirs "include;/usr/include;/usr/include/opencv4")
    ```
- ZSH (If you are using bash adapt your command in Step 4)

## Install ROS

- Follow steps [Ubuntu install of ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu).

  _I recommend to install_

  ```command
  sudo apt install ros-melodic-desktop-full
  ```

- Initialize ROS Core.

  ```console
  roscore
  ```

## Integrate this repo into your ROS Workspace.

**1. Clone this repo as a ROS Workspace.**

```command
cd ~/
git clone https://github.com/adolfos94/ROS-Stereo-Camera.git
```

**2. Navigate to the new ROS Workspace or you can integrate the source files to another ROS Workspace.**

```command
cd ROS-Stereo-Camera
```

**3. Build the package in the catking workspace.**

```console
$ adolfo in ROS-Stereo-Camera at jetson-nano
catkin_make
```

**4. Add the workspace to your ROS environment.**

```console
$ adolfo in ROS-Stereo-Camera at jetson-nano
source devel/setup.zsh
```

**5. Start the Publisher.**

```console
$ adolfo in ROS-Stereo-Camera at jetson-nano
rosrun stereo_camera_pub stereo_camera_pub_node
```

**6. Start the Suscriber.**

```console
$ adolfo in ROS-Stereo-Camera at jetson-nano
rosrun stereo_camera_sub stereo_camera_sub_node
```

![Example Stereo Camera](/res/stereo_example.png "Example Stereo Camera")

# Modules

## [Stereo Rectification](/Stereo%20Calibration/)
