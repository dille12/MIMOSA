import pyautogui
import time

def keep_awake(interval=30, movement=100):
    """
    Moves the mouse slightly every `interval` seconds to keep the computer awake.

    :param interval: Time in seconds between movements (default: 30s)
    :param movement: Pixels to move (default: 10px)
    """
    print("Keeping your computer awake. Press CTRL+C to stop.")

    try:
        while True:
            print("Moving...")
            x, y = pyautogui.position()  # Get current mouse position
            pyautogui.moveTo(x + movement, y, duration=1)  # Move slightly right
            pyautogui.moveTo(x, y, duration=1)  # Move back to original position
            time.sleep(interval)  # Wait before next move
    except KeyboardInterrupt:
        print("\nStopped. Your computer can now sleep.")

# Run the function
keep_awake(interval=30, movement=10)
