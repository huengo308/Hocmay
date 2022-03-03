import pygame #pip install pygame
import numpy
import pickle
from tkinter import messagebox
from tkinter import *
import sys

class ScaleData:
    def __init__(self, mouse_pos, thick_x, thick_y):
        self.mouse_pos = mouse_pos
        self.width = 28
        self.height = 28
        self.grid_width =500
        self.grid_height=500
        self.thick_x = thick_x
        self.thick_y = thick_y
    def scale_img_data(self):
        scale_grid = []
        scale_grid_row=[]
        differ_x = self.width / self.grid_width
        differ_y= self.height / self.grid_height
        scale_width, scale_height = round(self.thick_x*differ_x), round(self.thick_y*differ_y)

        for y in range(self.height):
            for x in range (self.width):
                scale_grid_row.append(0)
            scale_grid.append(numpy.array(scale_grid_row))
            scale_grid_row.clear()
        scale_grid=numpy.array(scale_grid)

        for mouse_pos in self.mouse_pos:
            x_pos, y_pos = mouse_pos
            scale_x, scale_y = round(x_pos*differ_x), round(y_pos*differ_y)

            for h in range(scale_height):
                for w in range(scale_width):
                    scale_grid[scale_y+h][scale_x+w]=255
        return scale_grid

def mouse_input():
    is_clicked = pygame.mouse.get_pressed(num_buttons=5)
    mouse_pos = pygame.mouse.get_pos()
    if is_clicked[0]:
        return mouse_pos
    
class MnistGui:
    def __init__(self):
        pygame.display.set_caption("Nhóm 17")
        self.width = self.height = 500
        self.bg_color = (255, 255, 255)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.cursor_pos = []
        self.thickness_x = self.thickness_y = 50
        self.root = Tk()
        self.root.eval(f"tk::PlaceWindow {self.root.winfo_toplevel()} center")
        self.root.withdraw()
    
    def run(self):
        #chạy toàn bộ vòng lặp chính
        while True:
            self.mouse_pos = mouse_input()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    try:
                        scaled_data = numpy.array([ScaleData(self.cursor_pos, thick_x=self.thickness_x, thick_y=self.thickness_y).scale_img_data()])
                        n_samples, nx, ny = scaled_data.shape
                        scaled_data = scaled_data.reshape((n_samples, nx*ny))
                        model_file = open("mnist.pickle", "rb")
                        model = pickle.load(model_file)

                        predict = model.predict(scaled_data)
                        messagebox.showinfo("Dự đoán",f"Số dự đoán là {predict[0]}")
                    except Exception:
                        messagebox.showerror("Error","Hệ thống không bắt được số, vui lòng thử lại!")
                    self.root.quit()
                    self.cursor_pos.clear()
            if self.mouse_pos is not None:
                self.cursor_pos.append((self.mouse_pos[0], self.mouse_pos[1]))
            self.draw()
            self.screen.fill(self.bg_color)

    def draw(self):
        try:
            color = (0,0,0)
            for cursor_pos in self.cursor_pos:
                pygame.draw.rect(self.screen, color, pygame.Rect(cursor_pos[0], cursor_pos[1], self.thickness_x, self.thickness_y))
            pygame.display.update()
        except Exception as e:
            print (e)
MnistGui().run()