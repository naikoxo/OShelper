import app_nlu_version

from tkinter import *
from PIL import ImageTk, Image
import time
from datetime import datetime


window = Tk()
window.title('Голосовой помощник')
window.geometry('800x450')
window.wm_attributes("-topmost", 1)

# Константы
MIN_WIDTH, MIN_HEIGHT = 400, 225
INIT_MIC_SIZE = 50
INIT_FONT_SIZE = 16

# Загрузка изображений
try:
    bg_image = Image.open("background.jpg")
    mic_image = Image.open("mic.png").resize((INIT_MIC_SIZE, INIT_MIC_SIZE), Image.LANCZOS)
except Exception as e:
    print(f"Ошибка загрузки изображений: {e}")
    exit()

# Canvas с правильным порядком элементов
canvas = Canvas(window)
canvas.pack(fill=BOTH, expand=True)

# Переменные
bg_photo = None
mic_photo = None
mic_obj = None
canvas_time = None
last_resize_time = 0

def update_time():
    current_time = datetime.now().strftime("%H:%M:%S")
    if canvas_time:
        canvas.itemconfig(canvas_time, text=current_time)
    window.after(1000, update_time)

def smart_resize():
    global bg_photo, mic_photo, mic_obj, canvas_time, last_resize_time
    
    # Дебаунс ресайза
    current_time = time.time() * 1000
    if current_time - last_resize_time < 100:
        return
    last_resize_time = current_time
    
    width = max(window.winfo_width(), MIN_WIDTH)
    height = max(window.winfo_height(), MIN_HEIGHT)
    
    # 1. Сначала фон (нижний слой)
    if not bg_photo or abs(bg_photo.width() - width) > 50 or abs(bg_photo.height() - height) > 50:
        bg_resized = bg_image.resize((width, height), Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_resized)
        canvas.delete("bg")
        canvas.create_image(0, 0, image=bg_photo, anchor=NW, tags="bg")
    
    # 2. Затем микрофон (средний слой)
    scale = min(width/800, height/450) * 0.9
    mic_size = int(INIT_MIC_SIZE * scale)
    
    if not mic_photo or abs(mic_photo.width() - mic_size) > 5:
        mic_resized = mic_image.resize((mic_size, mic_size), Image.LANCZOS)
        mic_photo = ImageTk.PhotoImage(mic_resized)
        canvas.delete("mic")
        mic_obj = canvas.create_image(width//2, height//2, image=mic_photo, anchor=CENTER, tags="mic")
        canvas.tag_bind(mic_obj, '<Button-1>', mic_click)
    
    # 3. И только потом часы (верхний слой)
    font_size = max(int(INIT_FONT_SIZE * scale), 8)
    if not canvas_time:
        canvas_time = canvas.create_text(20, 20, text=datetime.now().strftime("%H:%M:%S"), 
                                       font=("Arial", font_size), fill="white",  # Яркий цвет для теста
                                       anchor="nw", tags="time")
        canvas.tag_raise(canvas_time)  # Гарантируем, что часы поверх всего
        update_time()
    else:
        canvas.itemconfig(canvas_time, font=("Arial", font_size))
        canvas.tag_raise(canvas_time)  # Поднимаем часы на верхний слой при каждом ресайзе

def mic_click(event):
    if __name__ == '__main__':
   
        app_nlu_version.some_func()
    


def on_resize(event):
    window.after(50, smart_resize)

# Инициализация
window.bind('<Configure>', on_resize)
smart_resize()








window.mainloop()


