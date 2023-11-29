import argparse
import os
import time
from gtts import gTTS

required_items = ['hot_pad', 'pan', 'oatmeal', 'bowl', 'measuring_cup', 'measuring_spoons', 'small_spoon', 'salt',
                  'big_spoon', 'timer', 'measuring_cup_glass']


def save_items(file_names, detection):
    with open(file_names, 'w') as f:
        f.write(" ".join(detection))
    return


def load_items(file_name):
    with open(file_name, 'r') as f:
        results = f.readlines()[0].strip().split()
    num_dist = int(results.pop())
    return results, num_dist


def speak_results(detected_items, num_distractor, voice_path):
    missing_items = [item.replace("_", " ") for item in required_items if item not in detected_items]
    unnecessary_items = [item.replace("_", " ") for item in detected_items if item not in required_items]

    text = ""
    ready = False
    if missing_items:
        text += "The following required items: " + ", ".join(
            missing_items) + " are missing. Please prepare these items. "
    else:
        text += "All required items are present. "
    if unnecessary_items:
        distractor_text = "" if num_distractor == 0 else f", and {num_distractor} other items"
        text += "The following unnecessary items: " + ", ".join(
            unnecessary_items) + distractor_text + " were found. Please remove those items."
    else:
        text += "No unnecessary items were found."
    if not missing_items and not unnecessary_items:
        text += " You are all set!"
        ready = True

    # Specify the language (you can change 'en' to a different language code if needed)
    tts = gTTS(text=text, lang='en', slow=False)

    # Save the speech as an audio file
    tts.save(voice_path)

    # Play the audio file using a suitable command for your system
    play_command = None

    # Check the operating system and set the play command accordingly
    if os.name == 'posix':  # Unix-like systems
        # play_command = 'afplay'  # macOS
        # play_command = 'aplay'  # Linux
        play_command = 'mpg123'  # Raspbian
    elif os.name == 'nt':  # Windows
        play_command = 'start'
    else:
        print("Unsupported operating system")

    if play_command:
        os.system(f"{play_command} {voice_path}")

    return ready


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('item_path', help="Path to load all the detected items.")
    args.add_argument('voice_path', help="Path to save the generated voice file.")
    return args.parse_args()


if __name__ == '__main__':
    file_path = getargs().item_path
    voice_path = getargs().voice_path
    stop = False
    while not stop:
        if os.path.isfile(file_path):
            items, num = load_items(file_path)
            stop = speak_results(items, num, voice_path)
        time.sleep(5)
