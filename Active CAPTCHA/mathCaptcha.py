import random
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk

# Constants
STANDARD_SIZE = (500,500)
FONT_SIZE = 24
FONT_PATH = "arial.ttf"
LABEL_FONT_SIZE = 14
LABEL_COLOR = (255, 255, 255)
NOISE_LINES = 5
RECTANGLE_COLOR = (0, 0, 255)  # Blue color for rectangle
TRUE_RECTANGLE_COLOR = (0, 255, 0)  # Green color for true rectangle
FALSE_RECTANGLE_COLOR = (255, 0, 0)  # Red color for false rectangle
# Path to the directory containing the background images
BACKGROUND_IMAGE_DIR = r"C:\Users\kavin_1xozkcy\OneDrive\BTech-CSECS\Semesters\Sem-05\SIH\activeCaptcha\background_images\images\Pebbles"  
# Function to load and resize a random background image
def load_random_background_image():
    image_files = [f for f in os.listdir(BACKGROUND_IMAGE_DIR) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_path = os.path.join(BACKGROUND_IMAGE_DIR, random.choice(image_files))
    image = Image.open(image_path)
    return image.resize(STANDARD_SIZE)

# Function to generate a simple math equation
def generate_equation():
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
    operation = random.choice(['+', '-'])
    correct_result = num1 + num2 if operation == '+' else num1 - num2
    is_correct = random.choice([True, False])
    displayed_result = correct_result if is_correct else correct_result + random.choice([-1, 1])
    equation = f"{num1} {operation} {num2} = {displayed_result}"
    return equation, is_correct

# Function to draw a rectangle and text on an image
def draw_rectangle_and_text(draw, image, x, y, width, height, color, text, font_size):
    draw.rectangle((x, y, x + width, y + height), fill=color)
    draw.text((x + 10, y + 10), text, fill=(255, 255, 255), font=ImageFont.truetype(FONT_PATH, font_size))

# Function to overlay the equation, rectangles, and labels on a background image
def create_captcha_image(equation, is_correct):
    background_image = load_random_background_image()
    draw = ImageDraw.Draw(background_image)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    label_font = ImageFont.truetype(FONT_PATH, LABEL_FONT_SIZE)

    # Define rectangle sizes
    equation_rectangle_size = (150, 50)
    true_rectangle_size = (75, 25)
    false_rectangle_size = (75, 25)

    # Store placed rectangles to check for overlap
    placed_rectangles = []

    def get_random_position(size):
        """Generate random (x, y) coordinates for a given rectangle size within image boundaries."""
        max_x = background_image.size[0] - size[0]
        max_y = background_image.size[1] - size[1]
        return random.randint(0, max_x), random.randint(0, max_y)

    def check_overlap(rect1, rect2):
        """Check if two rectangles overlap."""
        (x1, y1, w1, h1) = rect1
        (x2, y2, w2, h2) = rect2
        return not (x1 + w1 <= x2 or x1 >= x2 + w2 or y1 + h1 <= y2 or y1 >= y2 + h2)

    def place_rectangle(size, max_attempts=100):
        """Try to place a rectangle randomly without overlapping other rectangles."""
        for _ in range(max_attempts):
            x, y = get_random_position(size)
            new_rect = (x, y, size[0], size[1])
            if all(not check_overlap(new_rect, rect) for rect in placed_rectangles):
                placed_rectangles.append(new_rect)
                return new_rect
        # If unable to place without overlap after max_attempts, return None
        return None

    # Place the equation rectangle
    equation_rectangle = place_rectangle(equation_rectangle_size)
    if equation_rectangle:
        draw_rectangle_and_text(draw, background_image, *equation_rectangle, RECTANGLE_COLOR, equation, FONT_SIZE)

    # Place the "True" rectangle
    true_rectangle = place_rectangle(true_rectangle_size)
    if true_rectangle:
        draw_rectangle_and_text(draw, background_image, *true_rectangle, TRUE_RECTANGLE_COLOR, "True", LABEL_FONT_SIZE)

    # Place the "False" rectangle
    false_rectangle = place_rectangle(false_rectangle_size)
    if false_rectangle:
        draw_rectangle_and_text(draw, background_image, *false_rectangle, FALSE_RECTANGLE_COLOR, "False", LABEL_FONT_SIZE)

    # Store rectangle positions and labels for tracking
    if true_rectangle and false_rectangle:
        background_image.info['true_rectangle'] = true_rectangle
        background_image.info['false_rectangle'] = false_rectangle
        background_image.info['correct_choice'] = is_correct

    return background_image

# Function to update the CAPTCHA display in the GUI
def show_captcha():
    equation, is_correct = generate_equation()
    image = create_captcha_image(equation,is_correct)
    captcha_image = ImageTk.PhotoImage(image)
    canvas.itemconfig(captcha_image_id, image=captcha_image)
    root.captcha_image_pil = image  # Store the PIL image
    root.captcha_image_tk = captcha_image  # Store the Tkinter PhotoImage
    root.true_rectangle = image.info['true_rectangle']
    root.false_rectangle = image.info['false_rectangle']
    root.correct_choice = image.info['correct_choice']

# Function to handle user clicks on the canvas
def on_canvas_click(event):
    true_rectangle_x, true_rectangle_y, true_rectangle_width, true_rectangle_height = root.true_rectangle
    false_rectangle_x, false_rectangle_y, false_rectangle_width, false_rectangle_height = root.false_rectangle

    if (true_rectangle_x <= event.x <= true_rectangle_x + true_rectangle_width and
        true_rectangle_y <= event.y <= true_rectangle_y + true_rectangle_height):
        selected = True
    elif (false_rectangle_x <= event.x <= false_rectangle_x + false_rectangle_width and
          false_rectangle_y <= event.y <= false_rectangle_y + false_rectangle_height):
        selected = False
    else:
        return

    # Validate the user's choice
    validate_user_input(selected)

# Function to validate the user's response
def validate_user_input(user_choice):
    if user_choice == root.correct_choice:
        result_label.config(text="Correct!")
    else:
        result_label.config(text="Incorrect!")


# Set up the main GUI window
root = tk.Tk()
root.title("Math CAPTCHA")

# Create a canvas to display the CAPTCHA image
canvas = tk.Canvas(root, width=STANDARD_SIZE[0], height=STANDARD_SIZE[1])
canvas.pack()
captcha_image_id = canvas.create_image(0, 0, anchor='nw')

# Bind the click event to the canvas
canvas.bind("<Button-1>", on_canvas_click)

# Create a label to show the result of the user's input
result_label = tk.Label(root, text="")
result_label.pack()

# Display the first CAPTCHA when the application starts
show_captcha()

# Start the Tkinter main event loop
root.mainloop()