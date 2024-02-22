import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from bacteria_counter_processor import BacteriaCounterProcessor


def to_odd_value(value):
    # Ensure the value is odd
    return int(value) + (1 if int(value) % 2 == 0 else 0)


class BacteriaCounterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bacteria Colony Counter")

        # Default parameters
        self.min_area = 10
        self.max_area = 400
        self.blur_kernel_size = 21
        self.adaptive_threshold_block_size = 19
        self.adaptive_threshold_C = 1
        self.morph_kernel_size = 3
        self.morph_iterations = 4
        self.distance_transform_threshold = 0.3
        self.distance_transform_mask_size = 3
        self.canvas_width = 1280
        self.canvas_height = 720

        self.image_path = ""
        self.image_loaded = False

        # Create the main frame
        self.main_frame = tk.Frame(self.root, padx=20, pady=20)
        self.main_frame.pack(side="left")

        # Create the results frame
        self.results_frame = tk.Frame(self.root, padx=20, pady=20)
        self.results_frame.pack(side="right")

        # "Select Image" field
        self.file_label = tk.Label(self.main_frame, text="Select Image:")
        self.file_label.grid(row=0, column=0, sticky="w")
        self.file_button = tk.Button(
            self.main_frame, text="Browse", command=self.browse_image)
        self.file_button.grid(row=0, column=1, sticky="w")

        # Configuration title
        self.config_title_label = tk.Label(
            self.main_frame, text="Configuration", font=("Helvetica", 16))
        self.config_title_label.grid(
            row=1, column=0, columnspan=2, pady=(10, 5), sticky="w")

        # Configuration options
        self.config_frame = tk.Frame(self.main_frame)
        self.config_frame.grid(row=2, column=0, columnspan=2, sticky="w")

        # Min area slider
        self.min_area_label = tk.Label(self.config_frame, text="Min Area:")
        self.min_area_label.grid(row=0, column=0, sticky="w")
        self.min_area_slider = tk.Scale(self.config_frame, from_=0, to=500,
                                        orient="horizontal", resolution=2, command=self.refresh_image, length=300)
        self.min_area_slider.set(self.min_area)
        self.min_area_slider.grid(row=0, column=1, sticky="w")

        # Max area slider
        self.max_area_label = tk.Label(self.config_frame, text="Max Area:")
        self.max_area_label.grid(row=1, column=0, sticky="w")
        self.max_area_slider = tk.Scale(self.config_frame, from_=0, to=1000,
                                        orient="horizontal", resolution=2, command=self.refresh_image, length=300)
        self.max_area_slider.set(self.max_area)
        self.max_area_slider.grid(row=1, column=1, sticky="w")

        # Blur kernel size slider
        self.blur_kernel_label = tk.Label(
            self.config_frame, text="Blur Kernel Size:")
        self.blur_kernel_label.grid(row=2, column=0, sticky="w")

        self.blur_kernel_slider = tk.Scale(
            self.config_frame, from_=1, to=99,
            orient="horizontal", command=self.update_kernel_size, length=300
        )
        self.blur_kernel_slider.set(self.blur_kernel_size)
        self.blur_kernel_slider.grid(row=2, column=1, sticky="w")

        # Adaptive threshold block size slider
        self.adaptive_threshold_block_label = tk.Label(
            self.config_frame, text="Adaptive Threshold Block Size:")
        self.adaptive_threshold_block_label.grid(
            row=3, column=0, sticky="w")
        self.adaptive_threshold_block_slider = tk.Scale(self.config_frame, from_=3, to=99,
                                                        orient="horizontal", command=self.update_block_size, length=300)
        self.adaptive_threshold_block_slider.set(
            self.adaptive_threshold_block_size)
        self.adaptive_threshold_block_slider.grid(row=3, column=1, sticky="w")

        # Adaptive threshold C slider
        self.adaptive_threshold_C_label = tk.Label(
            self.config_frame, text="Adaptive Threshold C:")
        self.adaptive_threshold_C_label.grid(row=4, column=0, sticky="w")
        self.adaptive_threshold_C_slider = tk.Scale(self.config_frame, from_=0, to=100,
                                                    orient="horizontal", command=self.refresh_image, length=300)
        self.adaptive_threshold_C_slider.set(self.adaptive_threshold_C)
        self.adaptive_threshold_C_slider.grid(row=4, column=1, sticky="w")

        # Morphological kernel size slider
        self.morph_kernel_label = tk.Label(
            self.config_frame, text="Morphological Kernel Size:")
        self.morph_kernel_label.grid(row=5, column=0, sticky="w")
        self.morph_kernel_slider = tk.Scale(self.config_frame, from_=1, to=15,
                                            orient="horizontal", command=self.refresh_image, length=300)
        self.morph_kernel_slider.set(self.morph_kernel_size)
        self.morph_kernel_slider.grid(row=5, column=1, sticky="w")

        # Morphological iterations slider
        self.morph_iterations_label = tk.Label(
            self.config_frame, text="Morphological Iterations:")
        self.morph_iterations_label.grid(row=6, column=0, sticky="w")
        self.morph_iterations_slider = tk.Scale(self.config_frame, from_=0, to=10,
                                                orient="horizontal", command=self.refresh_image, length=300)
        self.morph_iterations_slider.set(self.morph_iterations)
        self.morph_iterations_slider.grid(row=6, column=1, sticky="w")

        # Distance transform threshold slider
        self.dist_transform_threshold_label = tk.Label(
            self.config_frame, text="Distance Transform Threshold:")
        self.dist_transform_threshold_label.grid(
            row=7, column=0, sticky="w")
        self.dist_transform_threshold_slider = tk.Scale(self.config_frame, from_=0.1, to=1.0,
                                                        orient="horizontal", resolution=0.05, command=self.refresh_image, length=300)
        self.dist_transform_threshold_slider.set(
            self.distance_transform_threshold)
        self.dist_transform_threshold_slider.grid(row=7, column=1, sticky="w")

        # Distance transform distance radio buttons
        self.dist_transform_distance_label = tk.Label(
            self.config_frame, text="Distance Transform Mask Size:")
        self.dist_transform_distance_label.grid(
            row=8, column=0, sticky="w")

        self.distance_transform_mask_size_var = tk.IntVar()
        self.dist_transform_distance_0 = tk.Radiobutton(
            self.config_frame, text="0", variable=self.distance_transform_mask_size_var, value=0, command=self.refresh_image)
        self.dist_transform_distance_3 = tk.Radiobutton(
            self.config_frame, text="3", variable=self.distance_transform_mask_size_var, value=3, command=self.refresh_image)
        self.dist_transform_distance_5 = tk.Radiobutton(
            self.config_frame, text="5", variable=self.distance_transform_mask_size_var, value=5, command=self.refresh_image)

        self.dist_transform_distance_0.grid(row=8, column=1, sticky="w")
        self.dist_transform_distance_3.grid(row=8, column=2, sticky="w")
        self.dist_transform_distance_5.grid(row=8, column=3, sticky="w")

        # Results title
        self.results_title_label = tk.Label(
            self.results_frame, text="Results", font=("Helvetica", 16))
        self.results_title_label.grid(
            row=0, column=0, columnspan=2, pady=(20, 5), sticky="w")

        # Bacteria count display
        self.count_label = tk.Label(
            self.results_frame, text="Bacteria Count:", pady=10)
        self.count_label.grid(row=1, column=0, sticky="w")
        self.count_input = tk.Entry(self.results_frame, state='readonly')
        self.count_input.grid(row=1, column=1, sticky="w")

        # Image display canvas
        self.canvas = tk.Canvas(self.results_frame)
        self.canvas.grid(row=5, column=0, columnspan=2)

        self.processor = BacteriaCounterProcessor()

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tiff;*.tif")])
        if file_path:
            self.image_path = file_path
            self.image_loaded = True
            # Refresh the image with the default slider values
            self.refresh_image(0)

    def update_kernel_size(self, value):
        # Ensure that the selected value is odd
        new_value = int(value)
        if new_value % 2 == 0:
            new_value += 1

        # Update the slider value
        self.blur_kernel_slider.set(new_value)
        self.blur_kernel_size = new_value

        self.refresh_image()

    def update_block_size(self, value):
        # Ensure that the selected value is odd
        new_value = int(value)
        if new_value % 2 == 0:
            new_value += 1

        # Update the slider value
        self.adaptive_threshold_block_slider.set(new_value)
        self.adaptive_threshold_block_size = new_value

        self.refresh_image()

    def refresh_image(self, event=None):
        if self.image_loaded:
            # Get input values
            self.min_area = self.min_area_slider.get()
            self.max_area = self.max_area_slider.get()
            self.blur_kernel_size = self.blur_kernel_slider.get()
            self.adaptive_threshold_block_size = self.adaptive_threshold_block_slider.get()
            self.adaptive_threshold_C = self.adaptive_threshold_C_slider.get()
            self.morph_kernel_size = self.morph_kernel_slider.get()
            self.morph_iterations = self.morph_iterations_slider.get()
            self.distance_transform_threshold = self.dist_transform_threshold_slider.get()
            self.distance_transform_mask_size = self.distance_transform_mask_size_var.get()

            # Process image
            count, image_with_contours = self.processor._perform_count(
                self.image_path, self.min_area, self.max_area, self.blur_kernel_size,
                self.adaptive_threshold_block_size, self.adaptive_threshold_C,
                self.morph_kernel_size, self.morph_iterations,
                self.distance_transform_threshold, self.distance_transform_mask_size)

            # Convert the processed image to display in Tkinter canvas
            image_with_contours = cv2.cvtColor(
                image_with_contours, cv2.COLOR_BGR2RGB)
            image_with_contours = Image.fromarray(image_with_contours)

            # Resize image to fit canvas
            ratio = min(self.canvas_width / image_with_contours.width,
                        self.canvas_height / image_with_contours.height)
            new_width = int(image_with_contours.width * ratio)
            new_height = int(image_with_contours.height * ratio)
            image_with_contours = image_with_contours.resize(
                (new_width, new_height))

            photo = ImageTk.PhotoImage(image_with_contours)

            # Display the processed image with contours
            self.canvas.config(width=self.canvas_width,
                               height=self.canvas_height)
            self.canvas.create_image((self.canvas_width - new_width) // 2,
                                     (self.canvas_height - new_height) // 2, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep reference to prevent garbage collection

            # Update the count input
            self.count_input.config(state='normal')
            self.count_input.delete(0, 'end')
            self.count_input.insert(0, str(count))
            self.count_input.config(state='readonly')
        else:
            # Hide the canvas if no image is loaded
            self.canvas.config(width=0, height=0)
