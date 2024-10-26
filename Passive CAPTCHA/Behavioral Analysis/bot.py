import pynput
from pynput.keyboard import Key, Listener

def on_press(key):
    try:
        print('Pressed:', key.char)
    except AttributeError:
        print('Pressed:', key)

def on_release(key):
    if key == Key.esc:
        # Stop listener
        return False

# Start listener
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

key_labels = {
    'a': 'letter',
    '1': 'number',
    'Enter': 'special'
}

# ... (rest of the code)

def on_press(key):
    try:
        print('Pressed:', key.char, 'Label:', key_labels[key.char])
    except AttributeError:
        print('Pressed:', key)
         
