import random
import math
import os
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk

def create_points_image(num_points, width, height, margin=50, dot_radius=20, min_distance=50):
    """
    Create an image with random numbered points that do not overlap.
    
    Parameters:
    - num_points: Number of points to generate.
    - width, height: Dimensions of the image.
    - margin: The margin within the image where points should not be placed.
    - dot_radius: The radius of the dots to be drawn.
    - min_distance: Minimum distance between any two points (center-to-center).
    
    Returns:
    - image: The generated image with points.
    - points: A list of (x, y) tuples for the point coordinates.
    """
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arialbd.ttf", 32)
    except IOError:
        font = ImageFont.load_default()

    points = []

    def is_valid_point(x, y, existing_points, min_distance):
        """Check if the new point (x, y) is valid and doesn't overlap with existing points."""
        for px, py in existing_points:
            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < min_distance:
                return False
        return True

    for i in range(num_points):
        while True:  # Keep trying to place the point until it doesn't overlap
            x = random.randint(margin, width - margin)
            y = random.randint(margin, height - margin)
            
            if is_valid_point(x, y, points, min_distance):
                points.append((x, y))
                
                # Draw larger circles (dots)
                draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=(255, 0, 0, 255))

                # Number the points starting from 1
                text = str(i + 1)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_x = x - (bbox[2] - bbox[0]) // 2
                text_y = y - (bbox[3] - bbox[1]) // 2
                draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
                
                break  # Exit the while loop once a valid point is found

    return image, points

# Load random background image from a directory and its subfolders, resizing it to 500x500
def load_random_image(image_directory):
    image_files = []
    for root, _, files in os.walk(image_directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    random_image_path = random.choice(image_files)
    img = Image.open(random_image_path)
    return img.resize((500, 500))

# Check if the user clicked near the current point (with a wider radius to allow imprecise tracing)
def is_near(point, pos, threshold=15):
    x, y = point
    px, py = pos
    return (x - threshold < px < x + threshold) and (y - threshold < py < y + threshold)

# Tkinter GUI for whiteboard-like tracing
class WhiteboardApp:
    def __init__(self, root, image_directory):
        self.root = root
        self.root.title("Whiteboard Tracing")

        # Load background and generate random points
        self.bg_image = load_random_image(image_directory)
        self.width, self.height = self.bg_image.size
        self.num_points = random.randint(3, 5)
        self.points_image, self.points = create_points_image(self.num_points, self.width, self.height)

        # Combine the background and points
        self.combined_image = Image.alpha_composite(self.bg_image.convert('RGBA'), self.points_image)

        # Display the image in Tkinter
        self.tk_image = ImageTk.PhotoImage(self.combined_image)
        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.pack()

        # Add label to display feedback
        self.feedback_label = tk.Label(root, text="Trace the points in order!.", font=("Arial", 16))
        self.feedback_label.pack(pady=10)

        # Mouse interaction variables
        self.traced_path = []
        self.mouse_movements = []
        self.is_drawing = False
        self.visited_order = []  # To track the order of visited points
        self.lines_connected = []  # To ensure lines are drawn between points
        self.tracing_active = True  # To control if tracing should continue

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Start a timer for 30 seconds
        self.root.after(15000, self.stop_tracing_due_to_timeout)  # 30000 ms = 30 seconds

    def on_click(self, event):
        if not self.tracing_active:
            return  # Do nothing if tracing is stopped
        if len(self.lines_connected) < self.num_points - 1:  # Stop drawing after all lines are connected
            self.is_drawing = True
            self.traced_path.append((event.x, event.y))
            self.mouse_movements.append((event.x, event.y))

            self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="blue")

    def on_draw(self, event):
        if not self.tracing_active or not self.is_drawing:
            return  # Do nothing if tracing is stopped or not in drawing mode
        if self.is_drawing and len(self.lines_connected) < self.num_points - 1:
            last_x, last_y = self.traced_path[-1]
            self.traced_path.append((event.x, event.y))
            self.canvas.create_line(last_x, last_y, event.x, event.y, fill="blue", width=3)
            self.mouse_movements.append((event.x, event.y))

    def on_release(self, event):
        if not self.tracing_active:
            return  # Do nothing if tracing is stopped
        self.is_drawing = False
        self.check_order()

    def check_order(self):
        """Check if the user visits the points in the correct order and connects them with a line."""
        self.visited_order = []  # Reset visited order
        correct_order = True

        # Check if the points are visited in the correct sequence
        for trace in self.traced_path:
            for i, point in enumerate(self.points):
                if is_near(point, trace, threshold=15):
                    if len(self.visited_order) == 0 and i == 0:  # Start at the first point
                        self.visited_order.append(i)
                    elif len(self.visited_order) > 0 and i == len(self.visited_order):  # Next point in order
                        self.visited_order.append(i)
                        if len(self.visited_order) > 1:
                            # Ensure lines are connected in the correct order
                            self.lines_connected.append((self.visited_order[-2], self.visited_order[-1]))
                        break

        # Check if all points are connected with lines in the correct sequence
        if len(self.visited_order) == self.num_points and len(self.lines_connected) == self.num_points - 1:
            self.stop_tracing("You traced all points correctly with lines!")
        else:
            self.feedback_label.config(text="Incorrect tracing! Make sure to connect all points in the correct order.")

    def stop_tracing_due_to_timeout(self):
        """Stop tracing if the time limit is reached."""
        if self.tracing_active:
            self.stop_tracing("Time is up! You didn't trace all points in time.")

    def stop_tracing(self, message):
        """Stop tracing and show a message."""
        self.tracing_active = False  # Disable further tracing
        self.feedback_label.config(text=message)

        # Optionally, unbind mouse events to completely stop interactions
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

# Main function to start the Tkinter app
if __name__ == "__main__":
    import sys
    if sys.version_info >= (3, 7):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    root = tk.Tk()

    # Set the path to your image dataset directory
    image_directory = r"C:\Users\kavin_1xozkcy\OneDrive\BTech-CSECS\Semesters\Sem-05\SIH\activeCaptcha\background_images\images\Pebbles"

    app = WhiteboardApp(root, image_directory)
    root.mainloop()
