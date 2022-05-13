import pyzed.sl as sl
import cv2
import numpy as np
import cv_viewer.tracking_viewer as cv_viewer
import ogl_viewer.viewer as gl
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D


def main():
    zed = sl.Camera()

    # Creating an InitParameters object and setting configuration parameters for ZED
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Using HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = True

    # Opening the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    obj_param = sl.ObjectDetectionParameters()
    # Enabling the human tracking and object detection module with positional tracking
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
    obj_param.enable_tracking = True
    obj_param.image_sync = True
    obj_param.enable_mask_output = False
    #Choosing the 34 landmarks pose model
    obj_param.body_format = sl.BODY_FORMAT.POSE_34 
    # Optimizing the person joints position
    obj_param.enable_body_fitting = True

    camera_infos = zed.get_camera_information()
    if obj_param.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Object Detection: Loading Module...")

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    
    # Optimal confidence threshold for indoor setting since it is in close range with objects 
    obj_runtime_param.detection_confidence_threshold = 40

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_infos.camera_resolution.width, 1280), min(camera_infos.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_infos.camera_resolution.width
                 , display_resolution.height / camera_infos.camera_resolution.height]

    print(image_scale)

    # Starting video stream
    viewer = gl.GLViewer()
    

    #Creating objects of the ZED API class to capture objects and their positions
    bodies = sl.Objects()
    image = sl.Mat()
    all_positions = []
    
    k = 0
    runtime_parameters = sl.RuntimeParameters()
    frames = []
    # Starting frame capturing of the scene, frames set to 150, looping over each
    while k < 150:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            

            #Retrieving image with the camera
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.retrieve_objects(bodies, obj_runtime_param)
            viewer.update_view(image, bodies) 

            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale,bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            print("showing image and keypoints and frames: ",k)
            cv2.waitKey(10)
            

            #Retrieving objects from the image from one frame
            err = zed.retrieve_objects(objects, obj_runtime_param)
            
            obj_array = bodies.object_list
            print(str(len(obj_array)) + " Person(s) detected\n")
            if len(obj_array) > 0:
                first_object = obj_array[0]
                print("First Person attributes:")
                print(" Confidence (" + str(int(first_object.confidence)) + "/100)")
                if obj_param.enable_tracking:
                    print(" Tracking ID: " + str(int(first_object.id)) + " tracking state: " + repr(
                        first_object.tracking_state) + " / " + repr(first_object.action_state))
                #Getting the position, velocity and dimensions of the position of the pbject         
                position = first_object.position
                velocity = first_object.velocity
                dimensions = first_object.dimensions


                #Getting the 3D x,y,z position coordinates into a list, "all_positions" and saving them in a CSV file "Position"
                print(type(position))
                position_list = []
                for i in position:
                    position_list.append(i)

                all_positions.append(position_list)
                print("===========new_list_pos",position_list,type(position_list))
                fields = ["x coord","y_coord", "z_coord"]
                with open('Position', 'w') as f:
                    write = csv.writer(f)
        
                    write.writerow(fields)
                    write.writerows(all_positions)




                #Printing the above information
                print(" 3D position of object in space: [x:{0},y:{1},z:{2}]\n Velocity of object: [x:{3},y:{4},z:{5}]\n 3D dimentions of space: [x:{6},y:{7},z:{8}]".format(
                    position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], dimensions[0],
                    dimensions[1], dimensions[2]))
                if first_object.mask.is_init():
                    print(" 2D mask available")
                print(" Keypoint 2D ")
                keypoint_2d = first_object.keypoint_2d
                for it in keypoint_2d:
                    print("    " + str(it))
                print("\n Keypoint 3D ")
                keypoint = list(first_object.keypoint)
                print(type(keypoint), type(keypoint[0]))
                for it in keypoint:
                    print("    " + str(it))
            

            viewer.exit()
            k +=1
            print("total frames: =============================================================", k)
            frames.append(k)

    # Closing the camera once the number of frames have been looped over
    zed.close()

    #Printing object position and length of all the captured frames for movement detection and computation, which detected a person
    print(all_positions[len(all_positions)-1][0],all_positions[len(all_positions)-1][1],all_positions[len(all_positions)-1][2])
    print(len(all_positions))

    #Getting net distance (displacement) of the person in the scene by using the Euclidean Distance Formula (EDF) across the first and last coordinate of the person
    space = np.sqrt(np.square(all_positions[0][0]-all_positions[len(all_positions)-1][0])+np.square(all_positions[0][1]-all_positions[len(all_positions)-1][1])+np.square(all_positions[0][2]-all_positions[len(all_positions)-1][2]))
    print("space covered: ",space)


    #Calculating actual distance covered, storing each individual distance between the consecutive coordinates in a list and 
    #taking the absolute sum of all of them to compute the distance
    tot_distance = []
    plot_distance_x = 0
    plot_x = []

    #Looping over all frames with people detected
    for i in range(0,len(all_positions)):    #total distance
        j = 0
        # Distance in 3D space (x,y,z) of between two consecutive frames being computed through EDF: i and i-1, and storing the result into list: tot_distance
        one_distance = np.sqrt(np.square(all_positions[i][j]-all_positions[i-1][j])+np.square(all_positions[i][j+1]-all_positions[i-1][j+1])+np.square(all_positions[i][j+2]-all_positions[i-1][j+2]))
        tot_distance.append(one_distance)
        
        
    # Taking the sum of the total in the list: tot_distance to compute total distance in meters
    total_distance = sum(np.absolute(tot_distance))
    print("total xyz dist: ", total_distance)  

    #Initializing the system of measuring movements across single dimensions: side-to-side movements measured along x-axis, 
    # front-back movements measured along z axis, up-down movements measured along y-axis
    no_movement = 0
    #List for storing the x-distances in meters
    dist_x = []  
    total_dist_x = 0     #total distance covered in meters
    x_max = 0            #maximum coordinate reached by the person in space
    x_min = 0            #minimum coordinate reached by the person in space
    x_start = all_positions[0][0]           #starting position of person                  #distance along x
    for x in range(0,len(all_positions)):
        # print(x)
        # print(all_positions[x][0])

        #measuring the distance using EDF in just one axis (x) 
        one_x_distance = all_positions[x][0]-all_positions[x-1][0]    
        total_dist_x += np.absolute(one_x_distance)

        #Getting updated sums of distances after each coordinate to plot graphs later 
        plot_distance_x += np.absolute(one_x_distance)
        plot_x.append(plot_distance_x)

        #Distances covered in each frame stored in this list
        dist_x.append(one_x_distance)

        # Finding min and max positions reached by the person in the scene
        if all_positions[x][0] > x_max:
            x_max = all_positions[x][0]
        if all_positions[x][0] < x_min:
            x_min = all_positions[x][0]

        x_last = all_positions[x][0]

    total_x_distance = sum(np.absolute(dist_x))
    movement_x = 0
    print(total_x_distance)
    print(dist_x)
    print("total x dist: ", total_dist_x)
    print("x_max: ",x_max)
    print("x_min: ",x_min)
    print("x_last: ",x_last)
    print("x_start: ",x_start)

    #Setting threshold to detect movement in x direction considering the scope of error based on observations
    x_threshold = 0.5

    if total_x_distance > x_threshold:

        # Detecting various combinations of side to side movements, based on four parameters: start_position, end_position, minimum_coordinate, maximum_coordinate 
        # and printing the result
        if x_max > x_last and x_last > x_min and x_max == x_start:
            print("you moved left to right and then little right to left by ", total_dist_x, " meters")
        elif x_max > x_last and x_last > x_min and x_min == x_start:
            print("you moved right to left and then little left to right by ", total_dist_x, " meters")
        elif x_start > x_min and x_start < x_max and x_max == x_last:
            print("first you moved left to right and then you moved right to further left by ", total_dist_x, " meters") 
        elif x_start > x_min and x_start < x_max and x_min == x_last:
            print("first you moved right to left and then you moved left to further right by ",total_dist_x, " meters")
        elif x_min < (x_last-0.1):
            print("you moved right to left by ", total_dist_x, " meters")
        elif x_max > x_last:
            print("you moved left to right by ", total_dist_x, " meters")
        else: 
            print("detected sideways movement", total_dist_x, " meters")

        movement_x +=1
    else:        
        print("no movement along x")
        no_movement +=1
        

    
    # Starting the procedure of detecting front and back movements (measured along z-axis)in the same way
    plot_distance_z = 0
    plot_z = []
    dist_z = []
    z_max = 0    
    z_min = 10000                           #distance along z
    for z in range(0,len(all_positions)):
        
        one_z_distance = all_positions[z][2]-all_positions[z-1][2]
        dist_z.append(one_z_distance)
        plot_distance_z += np.absolute(one_z_distance)
        plot_z.append(plot_distance_z)

        if all_positions[z][2] > z_max:
            z_max = all_positions[z][2]
        if all_positions[z][2] < z_min:
            z_min = all_positions[z][2]
        
        z_last = all_positions[z][2]

    total_z_distance = sum(np.absolute(dist_z))
    movement_z = 0
    print("total z distance: ", total_z_distance)
    print("z_max: ", z_max)
    print("z_min: ", z_min)
    print("z_last: ", z_last)
    z_threshold = 0.6
    if total_z_distance > z_threshold:

        # Detecting combinations of front and back movements based on 3 parameters: maximum_Z_coord, minimum_z_coord, last_z_coord
        if z_last == z_min:
            print("front movement by ", total_z_distance, " meters")
        
        elif z_last == z_max:
            print("back movement by ", total_z_distance, " meters")

        elif z_last + 0.5 < z_max:
            print("detected back and front movement by ", total_z_distance, " meters")
        else: print("detected slight z movement by ", total_z_distance, " meters")
        movement_z +=1
    else:    
        print("no movement along z")
        no_movement += 1
        

    #Starting the procedure to detect up and down movements (measured along y-axis) in the same way
    dist_y = []
    y_max = 0    
    y_min = 10000 
    plot_distance_y = 0
    plot_y = []                          
    for y in range(0,len(all_positions)):
        
        one_y_distance = all_positions[y][1]-all_positions[y-1][1]
        dist_y.append(one_y_distance)
        plot_distance_y += np.absolute(one_y_distance)
        plot_y.append(plot_distance_y)
        if all_positions[y][1] > y_max:
            y_max = all_positions[y][1]
        if all_positions[y][1] < y_min:
            y_min = all_positions[y][1]
        
        y_last = all_positions[y][1]

    total_y_distance = sum(np.absolute(dist_y))
    print("total y distance: ", total_y_distance)
    print("y_max: ", y_max)
    print("y_min: ", y_min)
    print("y_last: ", y_last)
    movement_y = 0
    y_threshold = 0.7



    if total_y_distance > y_threshold:

        #Detecting combinations of upward-downward movements
        if y_last == y_min:
            print("up movement by ", total_y_distance, " meters")
        
        elif y_last == y_max:
            print("down movement by ", total_y_distance, " meters")

        elif y_last > y_min:
            print("up-down movement", total_y_distance, " meters")

        movement_y +=1
    else:
         
        print("no movement along y")
        no_movement += 1


    #Detecting movements in more than one dimension, for example: if the person moves sideways as well as backwards (diagonally)
    if movement_x == 1 and movement_z == 1 and movement_y == 0:
        print("you moved sideways and front/back")           #  <-- diagonal  
    elif movement_x == 1 and movement_z == 0 and movement_y == 1:
        print("you moved sideways and up/down") 
    elif movement_x == 0 and movement_z == 1 and movement_y == 1:
        print("you moved fornt/back and up/down")
    elif movement_x == 1 and movement_z == 1 and movement_y == 1:
        print("you moved in all directions")
    
    #if no threshold value of legit movement is observed, the person is said to be stationary
    if no_movement == 3:
        print("you are stationary")    

    # print(len(dist_x))
    # print(len(dist_y))

    #Plotting graphs of variation of different kinds of movements and distances covered throughout the length of the video, output shown and explained in the report, graphs are in Graphs folder
    frames = []
    for i in range(0,len(dist_x)):
        frames.append(i)
    plt.plot(frames,plot_x, label = "x distance")
    plt.plot(frames,plot_y, label = "y distance")
    plt.plot(frames,plot_z, label = "z distance")

    

    

    plt.xlabel('frames')
    plt.ylabel('distance')
    plt.legend()
    plt.show()
    
    #Plotting 3D graphs, showing variation of movements wrt each axes
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(plot_x,plot_z,plot_y, zdir='z')
    

    ax.set_xlabel('X distance')
    ax.set_ylabel('Z distance')
    ax.set_zlabel('Y distance')
    
    plt.show()

    




 




    

    
        

if __name__ == "__main__":
    main()
