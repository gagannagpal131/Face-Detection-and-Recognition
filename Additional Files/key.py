from pynput.keyboard import Key, Listener

def press(key):
    #print('pressed\n'.format(key))
    #print(key)
    
    if key == Key.enter:
        print('hello')
        return False
    elif key == Key.backspace:
        print('fuck off')

# Collect events until released
with Listener(on_press=press) as listener:
    listener.join()
    #print('hello')