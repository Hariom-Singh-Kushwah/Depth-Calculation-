

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path
import random as rng
import xlsxwriter
import math
import os
import time
import redis
import snap7
from snap7.snap7exceptions import Snap7Exception
# from fpdf import FPDF
from xlsxwriter.utility import xl_range
from prettytable import PrettyTable
from termcolor import colored

plc = snap7.client.Client()
r = redis.Redis("localhost", 6379)


rng.seed(123)


# *****************VARIABLES*********************

# Camera capture settings
res_x = 1280
res_y = 720
fps = 30

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    # print("No input parameter have been given.")
    # print("For help type --help")
    print(colored("Input mode: Device", 'green', 'on_grey', attrs=['bold']))
    print(colored("Note:Pass arguments if you want to read from Bag file", 'red', 'on_yellow', attrs=['dark']))
    read_from_device = 1
    # exit()
else:
    print(colored("Input mode: Recorded File", 'green', 'on_grey', attrs=['bold']))
    print(
        colored("Note: Do not pass arguments if you want to read from Device", 'red', 'on_yellow', attrs=['dark']))
    read_from_device = 0

# Check if the given file have bag extension
if read_from_device == 0:
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    if read_from_device == 0:
        bag_path = '08/08_nopp_ae.bag'
        rs.config.enable_device_from_file(config, bag_path)

    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, res_x, res_y, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, res_x, res_y, rs.format.rgb8, fps)

    # queue = rs.frame_queue(1)
    # Start streaming from file
    profile = pipeline.start(config)
    # profile = pipeline.start(config, queue) # 3 lines should be commented to disable frame_queue. #1

       
    if read_from_device == 1:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.depth_units, 0.0001)
        depth_sensor.set_option(rs.option. enable_auto_exposure, False)
        depth_sensor.set_option(rs.option.exposure, 105000.000)
        depth_sensor.set_option(rs.option.gain, 16.000)
        depth_sensor.set_option(rs.option.laser_power, 240.000)
        print("Camara settings has been configured successfully!")

    # SET PRESETS
    preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
    # print('preset range:'+str(preset_range))
    for i in range(int(preset_range.max)):
        visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
        #print('%02d: %s'%(i,visulpreset))
        if visulpreset == "High Accuracy":
            depth_sensor.set_option(rs.option.visual_preset, i)
            print("Camera operating in High Accuracy Mode")

    # Enable or Disable emitter
    if depth_sensor.supports(rs.option.emitter_enabled):
        #print('Enabling emitter...')
        depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
        print("Emitter Enabled")
    #is_emitted_enabled = depth_sensor.get_option(rs.option.emitter_enabled)
    #print('Emitter Enabled? {}'.format(is_emitted_enabled))
except Exception as e:
    print(e)

def depth():

    # Camera capture settings
    #res_x = 1280
    #res_y = 720
    #fps = 30

    #The following list can be modified if pixel points need to be changed
    center_pixels = [
        (413, 406),
        (1011, 406),
        (485, 386),
        (902, 386),
        (658, 399),
    ]

    # Post Processing
    # decimation_filter = 0
    disparity_filters = 0
    spatial_filter = 1  # Enable/Disable settings from above
    temporal_filter = 1
    # hole_filling_filter = 0

    # Set number of pixels to consider per point
    pixel_range = 1  # Layers of pixel enclosures, set 0 for one pixel
    pdc = 1  # Distance between pixel layers

    # Number of output samples to be collected
    output_sample_size = 10

    def get_depth(pixelxy, depth_frame, depth_intrin):
        '''Finds depth  and distance of pixel'''
        x, y = pixelxy
        depth = depth_frame.get_distance(x, y)
        dx, dy, dz = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
        distance = (math.sqrt((dx ** 2) + (dy ** 2) + (dz ** 2))) * 1000
        depth = depth * 1000
        return depth, distance

    def show_pixels(pixelxy, color_image, index):
        '''Highlight pixels on image'''
        x, y = pixelxy
        pixel_name = 'P' + str(index)
        # color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        yellow = (0, 255, 255)
        black = (0, 0, 0)
        red = (0, 0, 255)
        cv2.circle(color_image, (x, y), 3, red, -1)  # Center Pixel Point
        if index < len(Pixel_colors):
            cv2.circle(color_image, (x - 10, y - 25), 12, Pixel_colors[index], -1)
        else:
            cv2.circle(color_image, (x - 10, y - 25), 12, yellow, -1)
        cv2.circle(color_image, (x - 10, y - 25), 13, black, 1)
        cv2.putText(color_image, pixel_name, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.rectangle(color_image, (x-50,y+20), (x+50, y+40), yellow, -1)

    def resizer(image, scale_percent):
        '''Resizes the image based on given percentage value'''
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # dim = (500, 500)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return image

    def write_results_excel(write_col):
        # global write_col
        cell_range = xl_range(start_row, write_col, end_row, write_col)
        formula1 = '=AVERAGE(%s)' % cell_range
        formula2 = '=STDEV(%s)' % cell_range
        formula3 = '=MODE(%s)' % cell_range
        worksheet.write_formula(write_row, write_col, formula1, total_format1)
        worksheet.write_formula(write_row + 1, write_col, formula2, total_format2)
        worksheet.write_formula(write_row + 2, write_col, formula3, total_format3)
        write_col += 1
        return write_col

    def print_data_linebyline_onImage(tx, ty, color_image, text, i):
        black = (0, 0, 0)
        yellow = (0, 255, 255)
        if i < len(Pixel_colors):
            cv2.putText(color_image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Pixel_colors[i], 1)
        else:
            cv2.putText(color_image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
        return color_image

    def show_data_on_image(color_image):
        yellow = (0, 255, 255)
        white = (242, 242, 242)
        # cv2.rectangle(color_image, (10,10), (160, 150), yellow, -1)
        # cv2.putText(color_image, "Point Depth", (20, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # for Pixels

        # Left
        if res_y == 480:
            ly = 300  # for 480p
        else:
            ly = 600  # for 720p
        lx = 10

        cv2.rectangle(color_image, (lx, ly), (lx + 210, ly + 80), white, -1)
        cv2.putText(color_image, "Left", (lx + 85, ly + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        L_Dia_txt = "Diameter     : " + str(left_diameter)
        L_Pinion_txt = "Pinion Depth : " + str(pinion_depth_wrt_left)
        cv2.putText(color_image, L_Dia_txt, (lx + 10, ly + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(color_image, L_Pinion_txt, (lx + 10, ly + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Right
        if res_y == 480:
            rx = 430  # for 480p
            ry = 300
        else:
            rx = 1060  # for 720p
            ry = 600

        cv2.rectangle(color_image, (rx, ry), (rx + 210, ry + 80), white, -1)
        cv2.putText(color_image, "Right", (rx + 85, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        R_Dia_txt = "Diameter     : " + str(right_diameter)
        R_Pinion_txt = "Pinion Depth : " + str(pinion_depth_wrt_right)
        cv2.putText(color_image, R_Dia_txt, (rx + 10, ry + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(color_image, R_Pinion_txt, (rx + 10, ry + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Point Depth
        tx = 10  # 35
        ty = 10  # 60
        cv2.rectangle(color_image, (tx, ty), (tx + 150, ty + 140), white, -1)
        cv2.putText(color_image, "Point Depth", (tx + 10, ty + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0),
                    1)  # for Pixels

        for i, val in enumerate(p):
            pixel_name = 'P' + str(i) + ": "
            text = pixel_name + str(val)
            # print("tx, ty ", tx, ty)
            # print(type(ty))
            # cv2.putText(color_image, text, (tx, ty),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if i == 0:
                ty += 50
                color_image = print_data_linebyline_onImage(tx + 25, ty, color_image, text, i)
            else:
                color_image = print_data_linebyline_onImage(tx + 25, ty, color_image, text, i)
            ty += 20

        return color_image

        # POST PROCESSING SETTINGS
        # Decimation - reduces depth frame density while preserving z-accuracy and performing rudimentary hole-filling
    try:
        decimation = rs.decimation_filter()
        # decimation.set_option(rs.option.filter_magnitude, 4) #control the amount of decimation (linear scale factor)

        # Spatial    - edge-preserving spatial smoothing
        spatial = rs.spatial_filter() # emphasize the effect of the filter by cranking-up smooth_alpha and smooth_delta options


        # Hole Filling filter offers additional layer of depth extrapolation
        hole_filling = rs.hole_filling_filter()

        # Temporal   - reduces temporal noise
        temporal = rs.temporal_filter()

        # At longer range, it also helps using disparity_transform to switch from depth representation to disparity form
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        # Create opencv window to render image in
        # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

        # Set alignment color/depth
        align_to = rs.stream.depth
        align = rs.align(align_to)

        # Create colorizer object
        colorizer = rs.colorizer()
        table = PrettyTable(['Pixel No', 'Depth', 'Distance'])
        table2 = PrettyTable(['Measure', 'Left', 'Right'])

        # Green, Blue, Red, Purple, Orange
        Pixel_colors = [
            (51, 153, 0),
            (255, 102, 0),
            (0, 0, 204),
            (255, 0, 102),
            (0, 102, 255),
        ]

        # ********************************Write data to Excel Initialization START***********************
        exl_index_count = 0
        timestr = time.strftime("%Y%m%d-%H%M%S")

        if read_from_device == 0:
            # In case the file is in directory
            if '/' in args.input:
                bag_filename = args.input.split('/')
                excel_file = "/home/ssapl/cds2/rs2/code/Results/" + timestr + "-" + bag_filename[len(bag_filename) - 1] + ".xlsx"
            # In case the file lies in same directory as code
            else:
                bag_filename = args.input
                excel_file = "/home/ssapl/cds2/rs2/code/Results/" + timestr + "-" + bag_filename + ".xlsx"
        else:
            # Reading from Device
            bag_filename = 'Device'
            excel_file = "/home/ssapl/cds2/rs2/code/Results/" + timestr + "-" + bag_filename + ".xlsx"  # read bag from same dir

        workbook = xlsxwriter.Workbook(excel_file)
        worksheet = workbook.add_worksheet("Depth Data")

        bold = workbook.add_format({'bold': True, 'align': 'center', 'border': 1, 'valign': 'vcenter'})
        filter_format = workbook.add_format(
            {'bold': False, 'align': 'center', 'border': 1, 'valign': 'vcenter', 'fg_color': '#5cf773'})

        total_format1 = workbook.add_format({'bold': True, 'align': 'center', 'border': 1,
                                             'valign': 'vcenter', 'fg_color': '#88fcc6'})

        total_format2 = workbook.add_format({'bold': True, 'align': 'center', 'border': 1,
                                             'valign': 'vcenter', 'fg_color': '#fcf988'})

        total_format3 = workbook.add_format({'bold': True, 'align': 'center', 'border': 1,
                                             'valign': 'vcenter', 'fg_color': '#fcca88'})

        merge_format = workbook.add_format({
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#4fdce3'})

        merge_format1 = workbook.add_format({
            'bold': 1,
            'border': 1,
            'align': 'right',
            'valign': 'vcenter',
            'fg_color': '#f5f5f5',
            'font_color': 'navy'})

        worksheet.merge_range('A1:O2', 'Project: Drive Head - Pinion Depth', merge_format1)
        worksheet.insert_image('A1', '/home/ssapl/cds2/rs2/code/ssa-logo1.png', {'x_scale': 0.6, 'y_scale': 0.6, 'x_offset': 10})

        worksheet.merge_range('A3:A4', 'SL No.', merge_format)
        # Write Row Names - Pixel Number
        row = 3
        col = 1
        for i in range(len(center_pixels)):
            item = "P " + str(i)
            worksheet.write(row, col, item, bold)
            col += 1

        worksheet.merge_range(2, 1, 2, col - 1, 'Pixels', merge_format)

        worksheet.merge_range('H3:I3', 'Diameter', merge_format)
        worksheet.write(row, col + 1, "Left", bold)
        worksheet.write(row, col + 2, "Right", bold)

        worksheet.merge_range('K3:L3', 'Pinion Depth', merge_format)
        worksheet.write(row, col + 4, "Left", bold)
        worksheet.write(row, col + 5, "Right", bold)

        worksheet.merge_range('N3:O3', 'Delta', merge_format)
        worksheet.write(row, col + 7, "P0,P1", bold)
        worksheet.write(row, col + 8, "P2,P3", bold)
        # workbook.close()
        # ********************************Write data to Excel Initialization END***********************

        # ********************************Set multiple pixels per point START***********************
        pixels = []
        points = 0  # pixel points for each pixel
        
        for x, y in center_pixels:
            x_list = []
            y_list = []
            temp_x = x - (pixel_range * pdc)
            temp_y = y - (pixel_range * pdc)
            for _ in range((2 * (pixel_range + 1)) - 1):
                x_list.append(temp_x)
                y_list.append(temp_y)
                temp_x += pdc
                temp_y += pdc

            for i in x_list:
                for j in y_list:
                    pixels.append((i, j))
                    points += 1

        points = points / len(center_pixels)

        point_text = "Reading " + str(int(points)) + " pixel per point"
        print()
        print(colored(point_text, 'green'))
        # print(colored('Hello, World!', 'green', 'on_red'))
        time.sleep(1)

        # ********************************Set multiple pixels per point END***********************

        for x in range(5):  # Give time for Autoexposure to adjust
            frames = pipeline.wait_for_frames()

        Data_count = 0
        pinion_depth_wrt_left_list = []
        # Streaming loop
        while True:

            # Capture several consecutive frames

            frames1 = []
            for x in range(10):
                frames = pipeline.wait_for_frames()
                # frames = queue.wait_for_frame()# 2
                frames1.append(frames.get_depth_frame())
                # frames1.append(frames.as_frameset().get_depth_frame())# 3

            # for x in range(len(frames1)): #This can be commented if the next for loop is enabled below
            #    frame = temporal.process(frames1[x])

            # Applying filters sequentially one after another
            for x in range(len(frames1)):
                frame = frames1[x]
                # if decimation_filter == 1:
                # frame = decimation.process(frame) # Use with caution - This will reduce frame size
                # worksheet.write('R1', "Decimation", filter_format)
                if disparity_filters == 1:
                    frame = depth_to_disparity.process(frame)
                    worksheet.write('S1', "Disparity", filter_format)
                if spatial_filter == 1:
                    frame = spatial.process(frame)
                    worksheet.write('Q1', "Spatial", filter_format)  # Enable/Disable settings from above
                if temporal_filter == 1:
                    frame = temporal.process(frame)
                    worksheet.write('Q2', "Temporal", filter_format)
                if disparity_filters == 1:
                    frame = disparity_to_depth.process(frame)
                # if hole_filling_filter == 1:
                # frame = hole_filling.process(frame)
                # worksheet.write('R2', "Disparity", filter_format)

            # Get frameset of depth
            # frames = pipeline.wait_for_frames()

            # Get depth frame
            # depth_frame = frames.get_depth_frame() # Original
            depth_frame = frame.as_depth_frame()
            aligned_frames = align.process(frames)
            aligned_color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not aligned_color_frame: continue

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics

            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            color_image = np.asanyarray(aligned_color_frame.get_data())

            count = 0
            p = []
            temp_depth = 0
            temp_dist = 0

            # Resetting to point to first column of next row
            row += 1
            col = 0

            exl_index_count += 1
            worksheet.write(row, col, exl_index_count, bold)

            for i in range(len(pixels)):
                count += 1
                temp = get_depth(pixels[i], depth_frame, depth_intrin)
                # print("pixel", len(p),"Depth:",temp[0])
                temp_depth += temp[0]
                temp_dist += temp[1]
                if count % points == 0:
                    # print()
                    col += 1
                    avg_depth = temp_depth / points
                    avg_dist = temp_dist / points

                    avg_depth = float("{:.3f}".format(avg_depth))
                    avg_dist = float("{:.3f}".format(avg_dist))
                    
                    #***************Mod 11 Feb 2021***********
                    if len(p) == 2:
                        avg_depth = float("{:.3f}".format(avg_depth-11.7)) #Delta adjustment for P2
                        p.append(avg_depth)
                    elif len(p) == 3:
                        avg_depth = float("{:.3f}".format(avg_depth-9.2)) #Delta adjustment for P3
                        p.append(avg_depth)
                    elif len(p) == 4:
                        avg_depth = float("{:.3f}".format(avg_depth-5.3)) #Delta adjustment for P4
                        p.append(avg_depth)
                    else:
                        p.append(avg_depth)
                    #*************Mod ends****************
                    #p.append(avg_depth)
                    table.add_row([len(p) - 1, avg_depth, avg_dist])
                    worksheet.write(row, col, avg_depth)
                    temp_depth, temp_dist = 0, 0
                # if points == 1:
                #     show_pixels(pixels[i], color_image, len(p)-1) #for 1 point len(p) will start from 1
                # else if count % points == 1:
                #     show_pixels(pixels[i], color_image, len(p))#for more points len(p) will start from 0

            for i in range(len(center_pixels)):
                if i>1:
                    show_pixels(center_pixels[i], color_image, i)

            #right_diameter = (p[3] - p[1] - 30.37)  # P2 P4
            #left_diameter = (p[2] - p[0] - 30.45)  # P1 P3
            right_diameter = 150
            left_diameter = 150
            right_radius = right_diameter / 2
            left_radius = left_diameter / 2
            #pinion_depth_wrt_right = (p[4] - p[3]) + right_radius
            #pinion_depth_wrt_left = (p[4] - p[2]) + left_radius
            pinion_depth_wrt_right = (p[4] - p[3]) + right_radius
            pinion_depth_wrt_left = (p[4] - p[2]) + left_radius

            right_diameter = float("{:.3f}".format(right_diameter))
            left_diameter = float("{:.3f}".format(left_diameter))
            right_radius = float("{:.3f}".format(right_radius))
            left_radius = float("{:.3f}".format(left_radius))
            pinion_depth_wrt_right = float("{:.3f}".format(pinion_depth_wrt_right))
            pinion_depth_wrt_left = float("{:.3f}".format(pinion_depth_wrt_left))
            pinion_depth_wrt_left_list.append(pinion_depth_wrt_left)

            #new_pinion_depth_wrt_right = p[4] - (p[1] + 30.37 + right_radius)
            #new_pinion_depth_wrt_left = p[4] - (p[0] + 30.45 + left_radius)

            table2.add_row(['Diameter', left_diameter, right_diameter])
            table2.add_row(['Pinion Depth', pinion_depth_wrt_left, pinion_depth_wrt_right])
            # table2.add_row(['Depth T2B', new_pinion_depth_wrt_left, new_pinion_depth_wrt_right])

            worksheet.write(row, col + 2, left_diameter)
            worksheet.write(row, col + 3, right_diameter)

            worksheet.write(row, col + 5, pinion_depth_wrt_left)
            worksheet.write(row, col + 6, pinion_depth_wrt_right)

            P0P1 = float("{:.3f}".format(abs(p[0] - p[1])))
            P2P3 = float("{:.3f}".format(abs(p[2] - p[3])))
            worksheet.write(row, col + 8, P0P1)
            worksheet.write(row, col + 9, P2P3)

            # cell_range = xl_range(4, 1, row, 1)#F1:J1
            # formula = '=SUM(%s)' % cell_range
            # print("formula: ", formula)
            # worksheet.write_formula(row+1, 0, '=AVERAGE()')# P0

            print()
            print(table)
            print(table2)
            table.clear_rows()
            table2.clear_rows()
            # print()
            Data_count += 1
            print("Data Collected:", Data_count)
            print("===========================================================")

            # Display Values on image
            color_image = show_data_on_image(color_image)

            # Create opencv window to render image
            # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("Depth Stream", depth_color_image)
            cv2.imshow("color Stream", color_image)



            # *********************************EXIT process*******************************
            key = cv2.waitKey(1)
            # if pressed escape or 'q' exit program
            if key == 27 or key & 0xFF == ord('q') or Data_count >= output_sample_size:  # make this a var
                # cell_range = xl_range(4, 1, row, 1)#B5:B end
                # formula = '=_xlfn.STDEV.S(%s)' % cell_range
                # #print("formula: ", formula)
                # worksheet.write_formula(row+1, 1, formula)# P0

                worksheet.write(row + 1, 0, "MEAN", total_format1)
                worksheet.write(row + 2, 0, "STDEV", total_format2)
                worksheet.write(row + 3, 0, "MODE", total_format3)
                # Try and put the following lines in a separate function
                start_row = 4  # constant
                end_row = row  # constant
                write_row = row + 1  # constant
                write_col = 1  # var

                for i in range(len(p)):
                    write_col = write_results_excel(write_col)

                # For Diameter and Pinion Depth
                for j in range(3):
                    write_col += 1
                    for i in range(2):
                        write_col = write_results_excel(write_col)

                workbook.close()
                cv2.destroyAllWindows()
                # pdf.output("output.pdf", "F")
                PWD = os.getcwd()
                #file_path = colored('file://' + PWD + '/' + excel_file, 'cyan')
                file_path = colored('file://'+ excel_file, 'cyan')
                # print(file_path)
                print("\nProgram successfully Completed! Output written to:\n", file_path)
                break
    except Exception as e:
        print(e)
    finally:
        return float("{:.3f}".format(np.mean(pinion_depth_wrt_left_list)))


def check_plc():
    try:
        if plc.get_connected():
            return 1
        else:
            print("PLC: Not Connected! Trying to reconnect..")
            plc.connect('192.168.3.1', 0, 0)
            time.sleep(0.1)
            if not plc.get_connected():
                print("PLC: Connection Failed!")
                return 0 
            else:
                print("PLC: Communication Established Successfully!")
                return 1
    except Snap7Exception as e:
        print("PLC: Connection Failed!")
        return 0

def plc_trigger():
    try:
            b1=bytearray([0x00]) #1 byte defined
            addr=plc.db_read(801,0,1) #read 1 byte db after offset 262
            b1=snap7.util.get_bool(addr,0,0) #reading bit by bit
            if b1 == True:
                return True
            else:
                return False
    except:
        pass

def output_to_plc(d):
    try:
        real=depth
        #real=float(real)
        buff=bytearray([0x00,0x00,0x00,0x00])
        snap7.util.set_real(buff,0,real) #0 is index from where to start data writing on db
        plc.db_write(801,2,buff) # 0 is the offset value
        
        #Send value to reset plc trigger
        b2=bytearray([0x00])
        snap7.util.set_bool(b2,0,1,True)
        plc.db_write(801,0,b2)

    except Snap7Exception as e:
        pass

def check_camera():
    if os.system("ping -c 1 192.168.0.100") == 0:
        print("Camera is Connected")
        return 1
    else:
        print("Camera is Disconnected")
        return 0


if __name__ == "__main__":
    check_plc()
    check_camera()
    r.set("get_depth", "0")
    print("CDS2 is operational")
    while True:
        try:
            if r.get("get_depth") == b'1': # plc_trigger():
                r.set("depth_check", "start")
                result = depth()
                #output_to_plc(result)
                r.set("depth", str(result))
                r.set("get_depth", "0")
                r.set("depth_check", "complete")
        except Exception as e:
            print(e)
            continue
