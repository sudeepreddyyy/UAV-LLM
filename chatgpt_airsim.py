import openai
import re
import argparse
import sys
print(sys.path)
from airsim_wrapper import AirSimWrapper
import os
import time 
import json
import speech_recognition as sr
import threading
import time
from dotenv import load_dotenv
import airsim
import numpy as np
import nest_asyncio
import asyncio

# last_command_time = 0
command_processed = threading.Event()
inactivity_threshold = 49  # xx seconds for inactivity warning


# Setup argument parser
parser = argparse.ArgumentParser(description="ChatGPT AirSim Voice Command Interaction Script")
parser.add_argument("--prompt", type=str, default="prompts/airsim_basic.txt", help="Path to the chat prompt file.")
parser.add_argument("--sysprompt", type=str, default="system_prompts/airsim_basic.txt", help="Path to the system prompt file.")
args = parser.parse_args()

# Load configuration from file
with open("config.json", "r") as f:
    config = json.load(f)

# Initialize ChatGPT with the OpenAI API key
print("Initializing ChatGPT...")
openai.api_key = config["OPENAI_API_KEY"]

# Load system prompt
with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

# Initialize chat history
chat_history = [
    {"role": "system", "content": sysprompt},
    {"role": "user", "content": "Voice command initialized."}
]

with open(args.prompt, "r") as f:

    prompt = f.read()

chat_history.append({"role": "user", "content": prompt})

def ask(prompt):
    chat_history.append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=chat_history,
        temperature=0
    )
    chat_history.append({"role": "assistant", "content": completion.choices[0].message.content})
    return chat_history[-1]["content"]

# Regular expression to extract Python code blocks
code_block_regex = re.compile(r"```python\s+(.*?)```", re.DOTALL)

def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks).strip()
        return full_code
    return None

global so
so = False

global counts
counts = 0
def recognize_speech_to_text():
    recognizer = sr.Recognizer()
    
    global so
    global counts
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for voice commands...")
        audio = recognizer.listen(source)

        try:
            command_text = recognizer.recognize_google(audio).lower()
            print(f"Recognized command: {command_text}")
            if "hello" in command_text:
                so = False
                counts = 0
                return command_text
            elif "start" in command_text:
                if exit_event.is_set():
                    exit_event.clear()

                image_thread = threading.Thread(target=image, daemon=True)
                image_thread.start()
            elif "stop" in command_text:
                
                exit_event.set()

            else:
                counts +=1
                print("please say wake command")

                so = True
                
        except sr.UnknownValueError:
            counts +=1
            print("Sorry, I couldn't understand that.")
            
            so = True
            return None
        except sr.RequestError as e:
            counts+=1
            
            print(f"Could not request results; {e}")
            so = True
            return None



global statess
statess = True


def monitor_activity_and_timeout():
    print("Monitoring started...")
    global last_command_time, command_processed
    prompt_count = 0  # To keep track of the number of prompts given
    listening_for_commands = False
    global so
    last_command_time = time.time()
    

    while True:
        time.sleep(1)  # Check every second
        
        # Calculate elapsed time since the last command or user interaction
        current_time = time.time()
        # start_time = time.time()  # timer start
        elapsed_time = current_time - last_command_time

        if listening_for_commands:
            if elapsed_time > 60:  # Assuming 30 seconds of silence means command session ended
                listening_for_commands = False
                print("Listening session ended.")
                last_command_time = time.time()  # Reset the timer after the listening session
            continue
        
        if elapsed_time > 49 :
            if prompt_count < 3:
                print(f"You've been inactive for {int(elapsed_time)} seconds. Do you want to give a new command? Attempt {prompt_count + 1}/3 (yes/no)")
                response = input().strip().lower()
                if response == "yes":
                    command_processed.clear()  # Clear the flag to wait for a new command
                    listening_for_commands = True
                    print("Listening for voice commands...")
                    prompt_count = 0 
                    elapsed_time = 0 # Reset prompt count if a command is given
                else:
                    last_command_time = time.time()  # Ensure timer is reset even if no command given
                    prompt_count += 1
            else:
                print("No further commands received after 3 attempts. Returning the drone to initial position.")
                statess = False
                aw.land_at_original_position()
                break  # Exit the loop after action is taken
        elif elapsed_time % 10 == 0:  # Provide feedback every 10 seconds
            print(f"Waiting for command... ({int(elapsed_time)} seconds elapsed)")



def authenticate_user():
    choice = input("Do you want to [register] or [login]? ")
    if choice.lower() == 'register':
        username = input("Enter a username: ")
        email = input("Enter your email: ")
        password = input("Enter a password: ")
          # Assuming email is needed only for registration
        return aw.register_user(username, email, password)  # Assuming this returns True if registration is successful
    elif choice.lower() == 'login':
        username = input("Enter your username: ")
        password = input("Enter your password: ")
        return aw.verify_user(username, password)  # Assuming this returns True if login is successful
    return False  # Return False if neither register nor login is chosen

def voice_command_loop():
    global last_command_time
    global counts
    start_time = 0

    # Start the background thread for monitoring
    threading.Thread(target=monitor_activity_and_timeout, daemon=True).start()

    while statess:
        command_text = recognize_speech_to_text()
        if command_text:
            last_command_time = time.time()  # Update last command time
            command_processed.clear()
            if "capture picture" in command_text.lower():
                detected_objects = aw.capture_and_process_image()
                print(f"Detected objects: {', '.join(detected_objects)}")

                # Prompt user for confirmation
                user_response = input("Please check the image. Say Yes/No: ").strip().lower()
                if user_response != 'yes':
                    # Delete the captured image if the user says no
                    os.remove(aw.last_captured_image_path)  # Assuming you store the path in aw
                    print("Image deleted.")
                else:
                    # Continue with the code if the user says yes (you can add further processing here if needed)
                    print("Image kept. Proceeding...")
            
            elif "find and move to" in command_text.lower():
                object_name = command_text.split("find and move to")[1].strip()
                aw.search_for_object_and_move_to_it(object_name)
            else:    

                response = ask(command_text)
            
            # Check if the command_processed event is set to skip processing if timeout occurred
                start_time = time.time()
                if not command_processed.is_set():     
                    print(f"ChatGPT response: {response}")
                    code = extract_python_code(response)
                    if code:
                        print(f"Executing code:\n{code}")
                        exec(code)

             # Mark the command as processed to reset the monitoring logic
            command_processed.set()
            end_time = time.time()  #Timer end
            elapsed_time = end_time - start_time
            
            print(f"Command execution time: {elapsed_time:.2f} seconds") 
        if counts > 5:
            aw.land_at_original_position()
        
class colors:
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

print(f"{colors.GREEN}Initializing AirSim...{colors.ENDC}")




aw = AirSimWrapper()

print(dir(aw))  
aw.set_weather("snowy_day")
print(f"{colors.GREEN}Done.{colors.ENDC}")

async def authenticate_user(aw):
    while True:
        choice = input("Do you want to [register] or [login]? ")
        if choice.lower() == 'register':
            if aw.register_user():
                print("Registration successful, please log in.")
                continue  # Prompt for login after successful registration
            else:
                print("Registration failed, please try again.")
        elif choice.lower() == 'login':
            mm = await aw.login_user()
            if mm:
                print("Login successful, proceeding with the application.")
                return True
            else:
                print("Login failed, please try again.")
        else:
            print("Invalid option, please choose 'login' or 'register'.")
            continue

exit_event = threading.Event()

async def main():
    nest_asyncio.apply()
    aw = AirSimWrapper()
    result = await authenticate_user(aw)

    if result:
        print("Authentication successful, proceeding with the application.")
        
        # Start the voice command loop only after successful login
        #threading.Thread(target=monitor_activity_and_timeout, daemon=True).start()
        voice_thread = threading.Thread(target=voice_command_loop, daemon=True)
        voice_thread.start()
        # image_thread = threading.Thread(target=image, daemon=True)
        # image_thread.start()

        try:
            while True:
                time.sleep(1)  # Keep the main thread alive
        except KeyboardInterrupt:
            print("Stopping voice command interface. Quitting application...")
            voice_thread.join()  # Ensure proper cleanup by joining the thread
    else:
        print("Authentication failed.")

global Image_count
Image_count = 0
def image():
    global Image_count
    x  = aw.z()
    save_directory = 'C:/Users/sudee/anaconda3/envs/chatgpt/PromptCraft-Robotics-main/chatgpt_airsim/images'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    while True:
        if exit_event.is_set():
                break
    
        responses = x.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

# Process the captured image
        for idx, response in enumerate(responses):
            Image_count+=1
            if response.pixels_as_float:
                raise ValueError("Unsupported format. Pixels cannot be floats.")
    
    # Convert the image to a numpy array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

    # Save the image as a PNG file
            airsim.write_png(os.path.join(save_directory, f'image_{Image_count}.png'), img_rgb)
            
            #time.sleep(15)


if __name__ == "__main__":
    #main()
    asyncio.run(main())

    # Only start the voice command loop thread after successful authentication
#voice_thread = threading.Thread(target=voice_command_loop, daemon=True)
#voice_thread.start()


with open(args.prompt, "r") as f:
    prompt = f.read()

ask(prompt)
print("Welcome to the AirSim chatbot! Voice command interface is now active. Speak your commands.")

try:
    while True:
        time.sleep(1)  # Keep the main thread alive
except KeyboardInterrupt:
    print("Stopping voice command interface. Quitting application...")
