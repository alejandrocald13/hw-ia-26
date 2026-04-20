import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2

from image_processor import (
    resize_to_fit,
    rotate_image,
    apply_rgb_balance,
    apply_gaussian_blur,
    apply_sobel_x,
    apply_sobel_y,
    create_selection_mask,
    paint_region,
)
from utils import cv_to_tk


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class ImageProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Procesador de Imágenes")
        self.geometry("1450x820")
        self.minsize(1250, 720)

        self.original_image = None
        self.processed_image = None
        self.display_original = None
        self.display_processed = None

        self.canvas_w = 500
        self.canvas_h = 420

        self.scale_x = 1.0
        self.scale_y = 1.0

        self.selection_start = None
        self.selection_end = None
        self.selection_shape = tk.StringVar(value="rect")

        self._build_ui()

    def _build_ui(self):
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(top_frame, text="Cargar imagen", command=self.load_image).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(top_frame, text="Restaurar", command=self.reset_image).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(top_frame, text="Guardar resultado", command=self.save_image).pack(side="left", padx=5, pady=5)

        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)

        right_frame = ctk.CTkScrollableFrame(main_frame, width=360)
        right_frame.pack(side="right", fill="y", padx=10, pady=10)

        # Panel antes y después
        preview_frame = ctk.CTkFrame(left_frame)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

        original_frame = ctk.CTkFrame(preview_frame)
        original_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        processed_frame = ctk.CTkFrame(preview_frame)
        processed_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(original_frame, text="Antes (original)", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        self.original_canvas = tk.Canvas(original_frame, width=self.canvas_w, height=self.canvas_h, bg="white", highlightthickness=1)
        self.original_canvas.pack(padx=10, pady=10)

        ctk.CTkLabel(processed_frame, text="Después (procesada)", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=5)
        self.processed_canvas = tk.Canvas(processed_frame, width=self.canvas_w, height=self.canvas_h, bg="white", highlightthickness=1)
        self.processed_canvas.pack(padx=10, pady=10)

        self.original_canvas.bind("<Button-1>", self.on_mouse_down)
        self.original_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # ----------------------------
        # Controles
        # ----------------------------
        ctk.CTkLabel(right_frame, text="Filtros globales", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", pady=(10, 5))

        self.global_r = self._create_slider(right_frame, "Rojo global", 0, 255, 255)
        self.global_g = self._create_slider(right_frame, "Verde global", 0, 255, 255)
        self.global_b = self._create_slider(right_frame, "Azul global", 0, 255, 255)

        self.blur_value = self._create_slider(right_frame, "Blur gaussiano", 0, 20, 0)

        self.sobel_x_intensity = self._create_slider(right_frame, "Sobel X", 0, 10, 0)
        self.sobel_y_intensity = self._create_slider(right_frame, "Sobel Y", 0, 10, 0)

        ctk.CTkLabel(right_frame, text="Rotación", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", pady=(20, 5))
        self.angle_slider = self._create_slider(right_frame, "Ángulo libre", -180, 180, 0)

        quick_angles_frame = ctk.CTkFrame(right_frame)
        quick_angles_frame.pack(fill="x", pady=5)

        ctk.CTkButton(quick_angles_frame, text="90°", width=70, command=lambda: self.set_angle(90)).pack(side="left", padx=4, pady=5)
        ctk.CTkButton(quick_angles_frame, text="180°", width=70, command=lambda: self.set_angle(180)).pack(side="left", padx=4, pady=5)
        ctk.CTkButton(quick_angles_frame, text="270°", width=70, command=lambda: self.set_angle(270)).pack(side="left", padx=4, pady=5)
        ctk.CTkButton(quick_angles_frame, text="-90°", width=70, command=lambda: self.set_angle(-90)).pack(side="left", padx=4, pady=5)

        ctk.CTkLabel(right_frame, text="Selección de región", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", pady=(20, 5))

        shape_frame = ctk.CTkFrame(right_frame)
        shape_frame.pack(fill="x", pady=5)

        ctk.CTkRadioButton(shape_frame, text="Rectángulo", variable=self.selection_shape, value="rect", command=self.auto_update).pack(anchor="w", padx=10, pady=5)
        ctk.CTkRadioButton(shape_frame, text="Círculo", variable=self.selection_shape, value="circle", command=self.auto_update).pack(anchor="w", padx=10, pady=5)

        coords_frame = ctk.CTkFrame(right_frame)
        coords_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(coords_frame, text="X1").grid(row=0, column=0, padx=5, pady=5)
        self.x1_entry = ctk.CTkEntry(coords_frame, width=70)
        self.x1_entry.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(coords_frame, text="Y1").grid(row=0, column=2, padx=5, pady=5)
        self.y1_entry = ctk.CTkEntry(coords_frame, width=70)
        self.y1_entry.grid(row=0, column=3, padx=5, pady=5)

        ctk.CTkLabel(coords_frame, text="X2").grid(row=1, column=0, padx=5, pady=5)
        self.x2_entry = ctk.CTkEntry(coords_frame, width=70)
        self.x2_entry.grid(row=1, column=1, padx=5, pady=5)

        ctk.CTkLabel(coords_frame, text="Y2").grid(row=1, column=2, padx=5, pady=5)
        self.y2_entry = ctk.CTkEntry(coords_frame, width=70)
        self.y2_entry.grid(row=1, column=3, padx=5, pady=5)

        ctk.CTkButton(coords_frame, text="Usar coordenadas", command=self.use_manual_coordinates).grid(
            row=2, column=0, columnspan=4, padx=5, pady=8
        )

        self.apply_filters_to_selection = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            right_frame,
            text="Aplicar filtros globales solo a la selección",
            variable=self.apply_filters_to_selection,
            command=self.auto_update
        ).pack(anchor="w", pady=10)

        ctk.CTkLabel(right_frame, text="Color de selección", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", pady=(20, 5))

        self.sel_r = self._create_slider(right_frame, "Rojo selección", 0, 255, 255)
        self.sel_g = self._create_slider(right_frame, "Verde selección", 0, 255, 0)
        self.sel_b = self._create_slider(right_frame, "Azul selección", 0, 255, 0)
        self.sel_alpha = self._create_slider(right_frame, "Transparencia selección", 0, 100, 45)

        self.paint_selection_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            right_frame,
            text="Pintar selección",
            variable=self.paint_selection_var,
            command=self.auto_update
        ).pack(anchor="w", pady=5)

        ctk.CTkButton(right_frame, text="Limpiar selección", command=self.clear_selection).pack(fill="x", pady=15)

    def _create_slider(self, parent, text, from_, to, default):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=5)

        label_frame = ctk.CTkFrame(frame, fg_color="transparent")
        label_frame.pack(fill="x", padx=5, pady=(5, 0))

        ctk.CTkLabel(label_frame, text=text).pack(side="left")
        value_label = ctk.CTkLabel(label_frame, text=str(default), width=50)
        value_label.pack(side="right")

        steps = int(to - from_) if isinstance(from_, int) and isinstance(to, int) else 100
        slider = ctk.CTkSlider(frame, from_=from_, to=to, number_of_steps=steps)
        slider.pack(fill="x", padx=10, pady=5)
        slider.set(default)

        def update_label(value):
            value_label.configure(text=str(int(float(value))))
            self.auto_update()

        slider.configure(command=update_label)
        return slider

    def auto_update(self):
        if self.original_image is not None:
            self.apply_filters()

    def set_angle(self, angle):
        self.angle_slider.set(angle)
        self.auto_update()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")]
        )

        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen.")
            return

        self.original_image = image
        self.processed_image = image.copy()
        self.clear_selection(redraw=False)
        self.show_images()
        self.auto_update()

    def reset_image(self):
        if self.original_image is None:
            return

        self.processed_image = self.original_image.copy()

        self.global_r.set(255)
        self.global_g.set(255)
        self.global_b.set(255)
        self.blur_value.set(0)
        self.sobel_x_intensity.set(0)
        self.sobel_y_intensity.set(0)
        self.angle_slider.set(0)

        self.sel_r.set(255)
        self.sel_g.set(0)
        self.sel_b.set(0)
        self.sel_alpha.set(45)

        self.paint_selection_var.set(False)
        self.apply_filters_to_selection.set(False)

        self.clear_selection(redraw=False)
        self.show_images()
        self.auto_update()

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Aviso", "No hay imagen procesada para guardar.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg"), ("JPEG", "*.jpeg")]
        )

        if not file_path:
            return

        cv2.imwrite(file_path, self.processed_image)
        messagebox.showinfo("Éxito", "Imagen guardada correctamente.")

    def get_selection_mask(self, image_shape):
        if self.selection_start is None or self.selection_end is None:
            return None

        return create_selection_mask(
            self.selection_shape.get(),
            image_shape,
            self.selection_start,
            self.selection_end
        )

    def apply_filters(self):
        if self.original_image is None:
            return

        result = self.original_image.copy()
        mask = self.get_selection_mask(result.shape)

        filter_mask = mask if self.apply_filters_to_selection.get() else None

        result = apply_rgb_balance(
            result,
            int(self.global_r.get()),
            int(self.global_g.get()),
            int(self.global_b.get()),
            filter_mask
        )

        result = apply_gaussian_blur(
            result,
            int(self.blur_value.get()),
            filter_mask
        )

        result = apply_sobel_x(
            result,
            int(self.sobel_x_intensity.get()),
            filter_mask
        )

        result = apply_sobel_y(
            result,
            int(self.sobel_y_intensity.get()),
            filter_mask
        )

        if self.paint_selection_var.get() and mask is not None:
            result = paint_region(
                result,
                (
                    int(self.sel_r.get()),
                    int(self.sel_g.get()),
                    int(self.sel_b.get()),
                ),
                mask,
                alpha=float(self.sel_alpha.get()) / 100.0
            )

        angle = float(self.angle_slider.get())
        if angle != 0:
            result = rotate_image(result, angle)

        self.processed_image = result
        self.show_images()

    def show_images(self):
        if self.original_image is not None:
            self.display_original = resize_to_fit(self.original_image, self.canvas_w, self.canvas_h)
            tk_img_original = cv_to_tk(self.display_original)

            self.original_canvas.delete("all")
            x = self.canvas_w // 2
            y = self.canvas_h // 2
            self.original_canvas.create_image(x, y, image=tk_img_original)
            self.original_canvas.image = tk_img_original

            oh, ow = self.original_image.shape[:2]
            dh, dw = self.display_original.shape[:2]
            self.scale_x = ow / dw
            self.scale_y = oh / dh

            self.draw_selection()

        if self.processed_image is not None:
            self.display_processed = resize_to_fit(self.processed_image, self.canvas_w, self.canvas_h)
            tk_img_processed = cv_to_tk(self.display_processed)

            self.processed_canvas.delete("all")
            x = self.canvas_w // 2
            y = self.canvas_h // 2
            self.processed_canvas.create_image(x, y, image=tk_img_processed)
            self.processed_canvas.image = tk_img_processed

    def draw_selection(self):
        if self.selection_start is None or self.selection_end is None or self.display_original is None:
            return

        dw = self.display_original.shape[1]
        dh = self.display_original.shape[0]

        offset_x = (self.canvas_w - dw) // 2
        offset_y = (self.canvas_h - dh) // 2

        x1 = int(self.selection_start[0] / self.scale_x) + offset_x
        y1 = int(self.selection_start[1] / self.scale_y) + offset_y
        x2 = int(self.selection_end[0] / self.scale_x) + offset_x
        y2 = int(self.selection_end[1] / self.scale_y) + offset_y

        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(self.sel_r.get()),
            int(self.sel_g.get()),
            int(self.sel_b.get())
        )

        # Tkinter Canvas no maneja alpha real en fill, así que usamos un stipple
        # para simular transparencia.
        if self.selection_shape.get() == "rect":
            self.original_canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="red",
                width=2,
                fill=color_hex,
                stipple="gray25"
            )
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = max(abs(x2 - x1), abs(y2 - y1)) // 2
            self.original_canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                outline="red",
                width=2,
                fill=color_hex,
                stipple="gray25"
            )

    def canvas_to_image_coords(self, event_x, event_y):
        if self.display_original is None or self.original_image is None:
            return None

        dw = self.display_original.shape[1]
        dh = self.display_original.shape[0]

        offset_x = (self.canvas_w - dw) // 2
        offset_y = (self.canvas_h - dh) // 2

        local_x = event_x - offset_x
        local_y = event_y - offset_y

        if local_x < 0 or local_y < 0 or local_x >= dw or local_y >= dh:
            return None

        img_x = int(local_x * self.scale_x)
        img_y = int(local_y * self.scale_y)
        return img_x, img_y

    def on_mouse_down(self, event):
        coords = self.canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return

        self.selection_start = coords
        self.selection_end = coords
        self.update_entries_from_selection()
        self.auto_update()

    def on_mouse_drag(self, event):
        coords = self.canvas_to_image_coords(event.x, event.y)
        if coords is None or self.selection_start is None:
            return

        self.selection_end = coords
        self.update_entries_from_selection()
        self.auto_update()

    def on_mouse_up(self, event):
        coords = self.canvas_to_image_coords(event.x, event.y)
        if coords is None or self.selection_start is None:
            return

        self.selection_end = coords
        self.update_entries_from_selection()
        self.auto_update()

    def update_entries_from_selection(self):
        if self.selection_start is None or self.selection_end is None:
            return

        x1, y1 = self.selection_start
        x2, y2 = self.selection_end

        self._set_entry(self.x1_entry, x1)
        self._set_entry(self.y1_entry, y1)
        self._set_entry(self.x2_entry, x2)
        self._set_entry(self.y2_entry, y2)

    def use_manual_coordinates(self):
        if self.original_image is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        try:
            x1 = int(self.x1_entry.get())
            y1 = int(self.y1_entry.get())
            x2 = int(self.x2_entry.get())
            y2 = int(self.y2_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Las coordenadas deben ser números enteros.")
            return

        self.selection_start = (x1, y1)
        self.selection_end = (x2, y2)
        self.auto_update()

    def clear_selection(self, redraw=True):
        self.selection_start = None
        self.selection_end = None

        self._set_entry(self.x1_entry, "")
        self._set_entry(self.y1_entry, "")
        self._set_entry(self.x2_entry, "")
        self._set_entry(self.y2_entry, "")

        if redraw:
            self.auto_update()
        else:
            self.show_images()

    def _set_entry(self, entry, value):
        entry.delete(0, "end")
        entry.insert(0, str(value))


if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()