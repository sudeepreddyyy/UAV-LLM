import airsim
import math
import numpy as np
import logging
import os 
import bcrypt
from datetime import datetime
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import json
import time
from ultralytics import YOLO
from PIL import Image
import glob
import asyncio

CREDENTIALS_FILE = 'credentials.txt'
last_command_time = time.time()

OBJECT_LOCATIONS_FILE = 'object_locations.json'

if not os.path.isfile(OBJECT_LOCATIONS_FILE): 
    raise FileNotFoundError(f"The file {OBJECT_LOCATIONS_FILE} does not exist.")

# Configure logging
logging.basicConfig(filename='airsim_wrapper.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_object_name(key):
    if key.startswith("wheat_"):
        num = key.split("_")[1]
        if num.isdigit() and 2 <= int(num) <= 8052:
            return f"SM_Wheat_C{num}"
    return objects_dict.get(key, None)


def is_valid_email(email):
    # Simple regex for validating an email
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

class AirSimWrapper:

    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        logging.info('Connected to AirSim')
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        logging.info('Drone is armed and API control is enabled')

        initial_wind = airsim.Vector3r(0, 0, 0)
        self.client.simSetWind(initial_wind)

        # Initialize variables for logging
        self.logging_enabled = False
        self.log_duration = None
        self.log_start_time = None
        self.log_file = None
        self.log_data = []
        global original_position
        original_position = self.load_original_position()
        if original_position is None:
            original_position = self.get_current_position()
    
    
            self.save_original_position(original_position)
#set parameters for wind
    def set_wind(self, x, y, z):
        
        wind = airsim.Vector3r(x, y, z)
        self.client.simSetWind(wind)
        print(f"Wind set to: X={x} m/s, Y={y} m/s, Z={z} m/s")
#gets the original location cooridnates from json file.
    global ORIGINAL_POSITION_FILE 
    ORIGINAL_POSITION_FILE = 'original_position.json'
    wind = airsim.Vector3r(5, 0, 0)
    
    def z(self):
        return self.client
    def takeoff(self): #called when the drone needs to takeoff
        try:
            self.client.takeoffAsync().join()
            print("Takeoff successful.")
        except Exception as e:
            print(f"Error during takeoff: {str(e)}")

    def save_original_position(self):
        """Save the original position to a file."""
        with open(ORIGINAL_POSITION_FILE, 'w') as f:
            json.dump(get_drone_position(), f)

    def load_original_position(self):
        """Load the original position from a file."""
        if os.path.exists(ORIGINAL_POSITION_FILE):
            with open(ORIGINAL_POSITION_FILE, 'r') as f:
                return json.load(f)
        return None
    
    def land_at_original_position(self):
 
 
        print("go into it")
        print(original_position)
        print("now its at - ")
        print(self.get_current_position())
        self.client.moveToPositionAsync(int(original_position['x_val']), int(original_position['y_val']), int(original_position['z_val']), 2).join()
    
    
        time.sleep(2)
    
    # Lands the drone
        self.client.landAsync().join()
        return True
    

    def search_for_object_and_move_to_it(self, obj_name):
        
        global last_command_time
        
        print(f"Searching for {obj_name}...")
       
        start_time1 = time.time()
        # Set pitch to a fixed value once at the beginning
        self.adjust_camera_pitch(-15)  # Set pitch angle for better object detection

        object_found = False  # Track if the object is found
        rot = 0
        rotation_attempts = 0
        max_rotations = 8  

        while rotation_attempts < max_rotations:
            # Capture image and detect objects
            img = self.get_image()
            obj_list, obj_locs = self.detect_objects(img)

            # Normalize object name for comparison
            obj_name_normalized = obj_name.lower()
            detected_objects = [obj.lower() for obj in obj_list]
            print(f"Detected objects (normalized): {detected_objects}")

            # Check if the object is found
            if obj_name_normalized in detected_objects:
                obj_idx = detected_objects.index(obj_name_normalized)
                print(f"Object '{obj_name}' found!")
                end_time1 = time.time()  #Timer end
                elapsed_time1 = end_time1 - start_time1
                print(f"Command execution to find the object time: {elapsed_time1:.2f} seconds")

                # Get the distance, angle, and height to the object
                distance, angle, height = self.get_object_distance_and_angle_and_height(img, obj_locs[obj_idx])

                # Move towards the object if a valid distance is detected
                if distance is not None and distance > 15:  # Move if the object is more than 15 cm away
                    print(f"Distance to {obj_name}: {distance:.2f} cm. Moving closer...")
                    self.move_towards_object(distance, angle, height, obj_name, rot)
                    object_found = True  # Set object found flag to true
                    break  

                elif distance is not None and distance <= 15:
                    
                    print(f"Reached {obj_name}. Stopping search.")
                    
                    return  # Stop after reaching the object

                else:
                    print(f"Object '{obj_name}' detected, but could not determine distance. Trying again...")

            if not object_found:
                # Rotate the drone to the left if object is not found
                print(f"{obj_name} not found. Rotating to search...")
                self.turn_left()
                rot = rot + 1
                rotation_attempts += 1
                if (rot > 3):
                    rot = 0

    def turn_left(self):
        self.client.rotateByYawRateAsync(-45, 1).join()
    def turn_right(self):
        self.client.rotateByYawRateAsync(45, 1).join()

    def move_towards_object(self, distance, angle, height, obj_name, rot):
        """
        Moves the drone towards the object based on distance, angle, and height.
        Adjusts drone's movement until it gets close enough. Once reached, prompts the user.
        """
        step_distance = 0.5  
        step_height = 0.05  
        proximity_threshold = 30  

        print(f"Moving towards {obj_name}...")
        distance = distance/100

        current_pos = self.get_current_position()

        if rot == 0:
            dx = int(current_pos['x_val'])
            dy = int(current_pos['y_val'])-distance
            dz = int(current_pos['z_val'])
        elif rot == 1:
            dx = int(current_pos['x_val'])-distance
            dy = int(current_pos['y_val'])
            dz = int(current_pos['z_val'])     
        elif rot == 2:
            dx = int(current_pos['x_val'])
            dy = int(current_pos['y_val'])+distance
            dz = int(current_pos['z_val'])           
        else:
            dx = int(current_pos['x_val'])+distance
            dy = int(current_pos['y_val'])
            dz = int(current_pos['z_val'])
        self.fly_to([dx,dy,dz])

        print(f"Reached {obj_name}!")
        
        
        result_dirs = glob.glob('results/output*')
        if result_dirs:
            latest_result_dir = max(result_dirs, key=os.path.getctime)
            
            image_files = glob.glob(os.path.join(latest_result_dir, '*.jpg')) + glob.glob(os.path.join(latest_result_dir, '*.png'))
            if image_files:
                
                latest_image_file = image_files[0]
                img = Image.open(latest_image_file)
                img.show()  # Display the captured image
                print(f"Displaying image: {latest_image_file}")
            else:
                print("No images found in the latest results directory.")
                return  # Exit the method if no images are found
        else:
            print("No results directories found.")
            return  

        user_input = input("Does the image look okay? (yes/no): ").lower()

        if user_input == 'yes':
            print("Proceeding with the current image.")
                    
            current_position = self.get_current_position()
            x = current_position['x_val']
            y = current_position['y_val']
            z = current_position['z_val']


            command_text = f'**Command**: "Go to {obj_name}"\n   Response:\n  ```python\n  aw.takeoff()  # Command to take off\n  aw.fly_to([{x}, {y}, {z}])  # Command to fly to {obj_name}\n  ```\n'

                    
            airsim_basics_file = 'C:\\Users\\sudee\\anaconda3\\envs\\chatgpt\\PromptCraft-Robotics-main\\chatgpt_airsim\\prompts\\airsim_basic.txt'

            try:
                # Append the command and response to the file
                with open(airsim_basics_file, 'a') as file:
                    file.write('\n' + command_text)
                print(f"Appended new command to {airsim_basics_file}")
            except Exception as e:
                print(f"An error occurred while writing to the file: {e}")
                
        else:
            print("The image does not look okay. Retrying or taking necessary action.")    

    def get_depth_image(self):
        
        return np.zeros((480, 640))

    def get_depth_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)  # Request depth image in float format
        ])
    
        if len(responses) > 0:
            response = responses[0]
        
        # Check if the data is a valid float buffer (not empty)
            if response.image_data_float:
                # Convert depth data into a NumPy array (floats)
                depth_image = np.array(response.image_data_float, dtype=np.float32)
                depth_image = depth_image.reshape(response.height, response.width)
                return depth_image
            else:
                raise ValueError("Depth image data is empty or invalid.")
        else:
            raise ValueError("No image response received from AirSim.")

    def get_object_distance_and_angle_and_height(self, img, bbox):
        # Get the image dimensions
        img_width, img_height = img.size

        # Compute the center of the bounding box in image coordinates
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        # Retrieve depth image and depth at the bounding box center
        depth_image = self.get_depth_image()

        # Ensure depth image is valid
        if depth_image is None or center_x < 0 or center_x >= depth_image.shape[1] or center_y < 0 or center_y >= depth_image.shape[0]:
            logging.error("Invalid depth image or bounding box center out of bounds.")
            return None, None, None

        # Fetch the depth value at the bounding box center
        depth = depth_image[center_y, center_x]

        # Filter invalid depth values (adjust threshold based on AirSim's depth range)
        if depth <= 0 or depth > 1000:  # Adjusted threshold to ignore too-large depth values
            logging.error(f"Invalid depth value: {depth}.")
            return None, None, None

        # Convert depth to cm (assuming AirSim returns depth in meters)
        distance = depth * 100

        # Compute the angle to the object in radians
        angle = math.atan2(center_x - img_width / 2, img_height / 2 - center_y)

        # Compute the height of the object in the image (relative to the image height)
        height = (bbox[3] - bbox[1]) / img_height
    
        print(angle, depth)
        return distance, angle, height

    def adjust_camera_pitch(self, pitch_angle):
        """
        Adjusts the drone's pitch by tilting it to the desired pitch_angle.
        Since AirSim doesn't support direct camera pitch for multirotors, we adjust the drone's tilt instead.
        :param pitch_angle: The angle to tilt the drone.
        """
        # Fetch the current orientation of the drone
        current_orientation = self.client.simGetVehiclePose().orientation
        roll, pitch, yaw = airsim.to_eularian_angles(current_orientation)

        # Set the new pitch angle (in radians)
        new_pitch = math.radians(pitch_angle)

        # Adjust the drone's orientation to the new pitch
        new_orientation = airsim.to_quaternion(new_pitch, roll, yaw)
        self.client.simSetVehiclePose(airsim.Pose(self.client.simGetVehiclePose().position, new_orientation), True)

        print(f"Adjusted drone pitch to {pitch_angle} degrees.")


    def detect_objects(self, img):
        obj_list, obj_locs = self.process_image_with_yolo(img)
        if not obj_list:
            print("No objects detected in the current image.")
        else:
            print(f"Detected objects: {obj_list}")
        return obj_list, obj_locs

    def get_image(self):
        return self.capture_image()  
    
    def get_current_position(self):
        """Get the current position of the drone."""
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        return {
            "x_val": position.x_val,
            "y_val": position.y_val,
            "z_val": position.z_val
        }


    def capture_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        if not responses or len(responses) == 0:
            logging.error("No image received from AirSim")
            return None

        for idx, response in enumerate(responses):
            if response.width == 0 or response.height == 0:
                logging.error("Invalid image data received from AirSim")
                return None

        # Process image data into numpy array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

        # Ensure the size is valid before reshaping
            if img1d.size != response.width * response.height * 3:
                logging.error("Mismatch in image data size")
                return None

            img_rgb = img1d.reshape(response.height, response.width, 3)
        
        # Save the image
            if not os.path.exists('captures'):
                os.makedirs('captures')
        
            filename = f"capture_{idx}.png"
            file_path = os.path.join('captures', filename)
        
            airsim.write_png(file_path, img_rgb)
            logging.info(f"Image captured and saved as {filename}")

        # Return the loaded image object instead of file path
            return Image.fromarray(img_rgb)
    
        logging.error("Failed to capture image")
        return None

    def process_image_with_yolo(self, image_path):
        model =YOLO("yolov8x.pt")
        mypath = image_path
        results = model.predict(mypath, save=True, imgsz=640, conf=0.5, device='0', project="results", name="output",show_boxes=True)
        
        detected_objects = []
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                detected_objects.append(result.names[int(box.cls[0])])
                bbox = box.xyxy[0].cpu().numpy()
                bounding_boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])  
        logging.info(f"YOLO processed the image: {detected_objects}")
        return detected_objects, bounding_boxes

    def capture_and_process_image(self):
        # Capture image
        image_path = self.capture_image()
        # Process image with YOLO
        detected_objects = self.process_image_with_yolo(image_path)
        return detected_objects,  bounding_boxes        
    
    def land(self):
        self.client.landAsync().join()

    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def fly_to(self, point):
        if point[2] > 0:
            self.client.moveToPositionAsync(point[0], point[1], -point[2], 5).join()
        else:
            self.client.moveToPositionAsync(point[0], point[1], point[2], 5).join()
    
    def fly_to_object(self, object_name):
        """Move the drone to the specified object using its coordinates."""
        try:
            with open(OBJECT_LOCATIONS_FILE, 'r') as f:
                locations = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return
        
        if object_name in locations:
            location = locations[object_name]
            self.client.moveToPositionAsync(
                location['x'],
                location['y'],
                location['z'],
                5
            ).join()
            print(f"Moved to {object_name} at {location}")
        else:
            print(f"Object {object_name} not found in locations file.")
                    
    def fly_path(self, points):
        airsim_points = []
        for point in points:
            if point[2] > 0:
                airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
            else:
                airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()

    def set_yaw(self, yaw):
        self.client.rotateToYawAsync(yaw, 5).join()

    def get_yaw(self):
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return yaw

    def get_position(self, object_name):
        query_string = objects_dict[object_name] + ".*"
        object_names_ue = []
        while len(object_names_ue) == 0:
            object_names_ue = self.client.simListSceneObjects(query_string)
        pose = self.client.simGetObjectPose(object_names_ue[0])
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def set_weather(self, weather):
        if weather == "snowy_day":
            self.client.simEnableWeather(True)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
            logging.info('Weather set to snowy day')
        else:
            logging.warning(f'Invalid weather condition: {weather}')

    def prepare_log_file(self, log_type):
        # Ensure the 'logs' directory exists
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Prepare the log file name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"log_{log_type}_{timestamp}.csv"
        filepath = os.path.join(logs_dir, filename)
        return filepath

    def get_sensor_snapshot(self):
        # Get the state of the drone
        state = self.client.getMultirotorState()

        # Orientation (quaternion)
        orientation_q = state.kinematics_estimated.orientation
        # Convert quaternion to Euler angles for roll, pitch, yaw
        roll, pitch, yaw = airsim.utils.to_eularian_angles(orientation_q)

        # Position
        position = state.kinematics_estimated.position
        x, y, z = position.x_val, position.y_val, position.z_val

        # GPS data - assuming GPS sensor is available and set up in settings.json
        gps_data = self.client.getGpsData()
        lat, lon, alt = gps_data.gnss.geo_point.latitude, gps_data.gnss.geo_point.longitude, gps_data.gnss.geo_point.altitude

        # Print sensor data snapshot
        print(f"Orientation (Roll, Pitch, Yaw): {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
        print(f"Position (X, Y, Z): {x:.2f}, {y:.2f}, {z:.2f}")
        print(f"GPS (Latitude, Longitude, Altitude): {lat}, {lon}, {alt}")
    
    def send_otp(self, email):
        otp = random.randint(1000, 9999)
        from_email = "airsimpnw@gmail.com"  
        from_password = os.getenv("EMAIL_PASSWORD")  

        message = MIMEMultipart()
        message['From'] = from_email
        message['To'] = email
        message['Subject'] = 'Your OTP'
        message.attach(MIMEText(f'Your OTP is {otp}. Please enter this to log in.', 'plain'))

        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as session:
                session.starttls()
                session.login(from_email, from_password)
                session.sendmail(from_email, email, message.as_string())
            print("OTP sent successfully.")
        except smtplib.SMTPAuthenticationError:
            print("Failed to authenticate with the email server. Check your credentials.")
            return None
        except Exception as e:
            print(f"An error occurred during SMTP setup: {e}")
            return None

        return otp
    @staticmethod
    def send_otp(email):
        otp = random.randint(1000, 9999)
        from_email = "airsimpnw@gmail.com"  # Your Gmail address
        from_password = "jvbc gqqj ifln apbw"  # The App Password you generated

    # Setup the MIME
        message = MIMEMultipart()
        message['From'] = from_email
        message['To'] = email
        message['Subject'] = 'Your OTP'
        body = f"Your OTP is {otp}. Please enter this to log in."
        message.attach(MIMEText(body, 'plain'))

    # Create SMTP session
        session = smtplib.SMTP('smtp.gmail.com', 587)  # Gmail SMTP server
        session.starttls()  # Enable security
        try:
            session.login(from_email, from_password)  # Login with your email and App Password
            text = message.as_string()
            session.sendmail(from_email, email, text)
        except smtplib.SMTPAuthenticationError:
            print("Failed to login, check your email address and App Password.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            session.quit()

        return otp



    def register_user(self):
        name = input("Enter your name: ")
        email = input("Enter your email: ")
        password = input("Enter your password: ")
    
        with open(CREDENTIALS_FILE, "a") as file:
            file.write(f"{name},{password},{email}\n")
            print(f"Data written to file: {name},{password},{email}")  # Debugging output
        print("User registered successfully!")


    async def login_user(self):
        name = input("Enter your name: ")
        password = input("Enter your password: ")
    
        with open(CREDENTIALS_FILE, "r") as file:
            credentials_found = False
            for line in file:
                username, userpass, email = line.strip().split(',')
                print(f"Checking {username} with {userpass}")  # Debugging output
                if username == name and userpass == password:
                    credentials_found = True
                    print("Login successful!")
                    otp = self.send_otp(email)
                    user_otp = input("Enter the OTP sent to your email: ")
                    if int(user_otp) == otp:
                        print("OTP verified, you are logged in!")
                        return True
                    else:
                        print("Invalid OTP, please try again.")
                    break
            if not credentials_found:
                print("Invalid credentials, please try again.")

    
     
